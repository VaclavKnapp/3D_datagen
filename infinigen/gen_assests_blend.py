import argparse
import importlib
import logging
import math
import os
import random
import re
import subprocess
import traceback
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import bpy
import gin
import numpy as np
import submitit

# NOTE: logging config has to be before imports that use logging
logging.basicConfig(
    format="[%(asctime)s.%(msecs)03d] [%(module)s] [%(levelname)s] | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

import infinigen
from infinigen.assets.lighting import (
    hdri_lighting,
    holdout_lighting,
    sky_lighting,
    three_point_lighting,
)

# from infinigen.core.rendering.render import enable_gpu
from infinigen.assets.utils.decorate import read_base_co, read_co
from infinigen.assets.utils.misc import assign_material, subclasses
from infinigen.core import init, surface
from infinigen.core.init import configure_cycles_devices
from infinigen.core.placement import AssetFactory, density
from infinigen.core.tagging import tag_system
# noinspection PyUnresolvedReferences
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.test_utils import load_txt_list
from infinigen.tools import export

logger = logging.getLogger(__name__)

OBJECTS_PATH = infinigen.repo_root() / "infinigen/assets/objects"
assert OBJECTS_PATH.exists(), OBJECTS_PATH


def apply_texture_from_path(obj: bpy.types.Object, texture_path: Path):


    if not texture_path.exists() or not texture_path.is_file():
        logger.warning(f"Texture file not found at: {texture_path}")
        return


    mat = bpy.data.materials.new("ImportedTextureMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links


    for node in nodes:
        if node.type == "BSDF_PRINCIPLED":
            principled = node
        else:
            nodes.remove(node)


    tex_node = nodes.new('ShaderNodeTexImage')
    try:
        tex_node.image = bpy.data.images.load(filepath=str(texture_path), check_existing=True)
    except Exception as e:
        logger.error(f"Failed to load texture at {texture_path}: {e}")
        return


    tex_node.location = (-300, 200)
    principled.location = (0, 200)


    links.new(tex_node.outputs["Color"], principled.inputs["Base Color"])


    obj.data.materials.clear()
    obj.data.materials.append(mat)


def find_factory_in_objects(factory_name: str):
    """
    Search in `infinigen/assets/objects` modules to find the class or function
    named factory_name and return it if found. Otherwise raise an error.
    """
    for subdir in sorted(list(OBJECTS_PATH.iterdir())):
        clsname = subdir.name.split(".")[0].strip()
        with gin.unlock_config():
            module = importlib.import_module(f"infinigen.assets.objects.{clsname}")
        if hasattr(module, factory_name):
            logger.info(f"Found {factory_name} in {subdir}")
            return getattr(module, factory_name)
        logger.debug(f"{factory_name} not found in {subdir}")
    raise ModuleNotFoundError(f"{factory_name} not Found.")


def build_scene_asset(args, factory_name, idx):
    fac = find_factory_in_objects(factory_name)

    if args.dryrun:
        return

    with FixedSeed(idx):
        fac = fac(idx)
        try:
            if args.spawn_placeholder:
                ph = fac.spawn_placeholder(idx, (0, 0, 0), (0, 0, 0))
                asset = fac.spawn_asset(idx, placeholder=ph)
            else:
                asset = fac.spawn_asset(idx)
        except Exception as e:
            traceback.print_exc()
            print(f"{fac}.spawn_asset({idx=}) FAILED!! {e}")
            raise e

        fac.finalize_assets(asset)

        # If user requested 'fire', apply fire simulation
        if args.fire:
            from infinigen.assets.fluid.fluid import set_obj_on_fire
            set_obj_on_fire(
                asset,
                0,
                resolution=args.fire_res,
                simulation_duration=args.fire_duration,
                noise_scale=2,
                add_turbulence=True,
                adaptive_domain=False,
            )
            bpy.context.scene.frame_set(args.fire_duration)
            bpy.context.scene.frame_end = args.fire_duration
            bpy.data.worlds["World"].node_tree.nodes["Background.001"].inputs[
                1
            ].default_value = 0.04
            bpy.context.scene.view_settings.exposure = -1

        bpy.context.view_layer.objects.active = asset
        parent = asset

        # If the parent is an EMPTY, pick the largest child mesh
        if asset.type == "EMPTY":
            meshes = [o for o in asset.children_recursive if o.type == "MESH"]
            if not meshes:
                return asset
            sizes = []
            for m in meshes:
                co = read_co(m)
                sizes.append((np.amax(co, 0) - np.amin(co, 0)).sum())
            i = np.argmax(np.array(sizes))
            asset = meshes[i]

        # If we are allowed to modify geometry (apply transforms, etc.)
        if not args.no_mod:
            # Remove drivers if any
            if parent.animation_data is not None:
                for d in parent.animation_data.drivers.values():
                    parent.driver_remove(d.data_path)

            co = read_co(asset)  # co is Nx3 array
            x_min, x_max = np.amin(co, axis=0), np.amax(co, axis=0)

            # Center bounding box at origin
            center_xyz = (x_min + x_max) / 2.0
            parent.location = -center_xyz
            butil.apply_transform(parent, loc=True)


            if not args.no_ground:
                bpy.ops.mesh.primitive_grid_add(size=5, x_subdivisions=400, y_subdivisions=400)
                plane = bpy.context.active_object
                plane.location[-1] = x_min[-1]
                plane.is_shadow_catcher = True
                material = bpy.data.materials.new("plane")
                material.use_nodes = True
                material.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (
                    0.015, 0.009, 0.003, 1
                )
                assign_material(plane, material)

        if args.texture_path:
            apply_texture_from_path(asset, args.texture_path)

    return asset


def build_scene_surface(args, factory_name, idx):
    """
    Builds a surface-based scene (e.g. a plane or sphere),
    and if `--texture_path` is specified, applies that texture.
    """
    try:
        # Attempt to load the scatter module
        with gin.unlock_config():
            scatter = importlib.import_module(
                f"infinigen.assets.scatters.{factory_name}"
            )
        if not hasattr(scatter, "apply"):
            raise ValueError(f"{scatter} has no apply()")

        if args.dryrun:
            return

        # Build a plane with default material
        bpy.ops.mesh.primitive_grid_add(size=10, x_subdivisions=400, y_subdivisions=400)
        plane = bpy.context.active_object
        material = bpy.data.materials.new("plane")
        material.use_nodes = True
        material.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (
            0.015, 0.009, 0.003, 1
        )
        assign_material(plane, material)

        # Apply scatter
        if isinstance(scatter, type):
            scatter = scatter(idx)
        scatter.apply(plane, selection=density.placement_mask(0.15, 0.45))
        asset = plane

    except ModuleNotFoundError:
        # If not a scatter, try to interpret it as a material
        try:
            with gin.unlock_config():
                try:
                    template = importlib.import_module(
                        f"infinigen.assets.materials.{factory_name}"
                    )
                except ImportError:
                    # Fallback search
                    found_template = None
                    for subdir in os.listdir("infinigen/assets/materials"):
                        mod_name = subdir.split(".")[0]
                        module = importlib.import_module(
                            f"infinigen.assets.materials.{mod_name}"
                        )
                        if hasattr(module, factory_name):
                            found_template = getattr(module, factory_name)
                            break
                    if not found_template:
                        raise Exception(f"{factory_name} not Found.")
                    template = found_template

                if args.dryrun:
                    return

                # Build a sphere, or use make_sphere() if present
                if hasattr(template, "make_sphere"):
                    asset = template.make_sphere()
                else:
                    bpy.ops.mesh.primitive_ico_sphere_add(radius=0.8, subdivisions=9)
                    asset = bpy.context.active_object

                # If the template is a class, instantiate it
                if isinstance(template, type):
                    template = template(idx)

                template.apply(asset)

        except ModuleNotFoundError:
            raise Exception(f"{factory_name} not Found.")

    # ---- Apply a PNG texture to the surface if specified ----
    if args.texture_path:
        apply_texture_from_path(asset, args.texture_path)

    return asset


def build_and_save_asset(payload: dict):
    """
    Receives a dict containing 'fac', 'args', and 'idx'. 
    Builds the corresponding scene or surface, then saves as .blend.
    """
    factory_name = payload["fac"]
    args = payload["args"]
    idx = payload["idx"]

    logger.info(f"Building scene for {factory_name} {idx}")

    # Override idx if user explicitly sets --seed
    if args.seed > 0:
        idx = args.seed

    path = args.output_folder / factory_name
    blend_file = path / f"object_{idx:03d}.blend"

    if blend_file.exists() and args.skip_existing:
        print(f"Skipping {path}")
        return

    path.mkdir(exist_ok=True, parents=True)

    # Clear previous scene
    butil.clear_scene()

    # Build
    if "Factory" in factory_name:
        asset = build_scene_asset(args, factory_name, idx)
    else:
        asset = build_scene_surface(args, factory_name, idx)

    if args.dryrun:
        return

    # Configure devices/render
    configure_cycles_devices()

    # Save the .blend file with autopack
    butil.save_blend(str(blend_file), autopack=True)


def mapfunc(f, its, args):
    if args.n_workers == 1:
        return [f(i) for i in its]
    elif not args.slurm:
        with Pool(args.n_workers) as p:
            return list(p.imap(f, its))
    else:
        executor = submitit.AutoExecutor(folder=args.output_folder / "logs")
        executor.update_parameters(
            name=args.output_folder.name,
            timeout_min=60,
            cpus_per_task=2,
            mem_gb=8,
            slurm_partition=os.environ.get("INFINIGEN_SLURMPARTITION", "default"),
            slurm_array_parallelism=args.n_workers,
        )
        jobs = executor.map_array(f, its)
        for j in jobs:
            print(f"Job finished {j.wait()}")


def main(args):
    bpy.context.window.workspace = bpy.data.workspaces["Geometry Nodes"]

    init.apply_gin_configs(
        ["infinigen_examples/configs_indoor", "infinigen_examples/configs_nature"],
        skip_unknown=True,
    )
    surface.registry.initialize_from_gin()

    if args.debug is not None:
        for name in logging.root.manager.loggerDict:
            if not name.startswith("infinigen"):
                continue
            if len(args.debug) == 0 or any(name.endswith(x) for x in args.debug):
                logging.getLogger(name).setLevel(logging.DEBUG)

    init.configure_blender()

    if args.gpu:
        init.configure_render_cycles()


    if ".txt" in args.factories[0]:
        name = args.factories[0].split(".")[-2].split("/")[-1]
    else:
        name = "_".join(args.factories)

    if args.output_folder is None:
        args.output_folder = Path(os.getcwd()) / "outputs"

    path = Path(args.output_folder) / name
    path.mkdir(exist_ok=True, parents=True)

    factories = list(args.factories)

    # Expand special tokens if needed
    if "ALL_ASSETS" in factories:
        factories += [f.__name__ for f in subclasses(AssetFactory)]
        factories.remove("ALL_ASSETS")
        logger.warning(
            "ALL_ASSETS is deprecated. Use -f tests/assets/list_nature_meshes.txt or similar."
        )
    if "ALL_SCATTERS" in factories:
        factories += [p.stem for p in Path("infinigen/assets/scatters").iterdir()]
        factories.remove("ALL_SCATTERS")
    if "ALL_MATERIALS" in factories:
        factories += [p.stem for p in Path("infinigen/assets/materials").iterdir()]
        factories.remove("ALL_MATERIALS")
        logger.warning(
            "ALL_MATERIALS is deprecated. Use -f tests/assets/list_nature_materials.txt or similar."
        )


    has_txt = ".txt" in factories[0]
    if has_txt:
        factories = [
            f.split(".")[-1] for f in load_txt_list(factories[0], skip_sharp=False)
        ]


    if not args.postprocessing_only:
        for fac in factories:
            targets = [
                {"args": args, "fac": fac, "idx": idx} for idx in range(args.n_images)
            ]
            mapfunc(build_and_save_asset, targets, args)

    if args.dryrun:
        return


def snake_case(s):
    return "_".join(
        re.sub(
            r"([A-Z][a-z]+)",
            r" \1",
            re.sub(r"([A-Z]+)", r" \1", s.replace("-", " "))
        ).split()
    ).lower()


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_folder", type=Path, default=None)
    parser.add_argument(
        "-f",
        "--factories",
        default=[],
        nargs="+",
        help="List factories/surface scatters/materials or path to .txt file",
    )
    parser.add_argument(
        "-n", "--n_images", default=1, type=int, help="Number of scenes to create"
    )
    parser.add_argument(
        "-m",
        "--margin",
        default=0.01,
        help="Margin between the asset boundary and the image edge when adjusting camera",
    )
    parser.add_argument(
        "-R", "--resolution", default="1024x1024", type=str, help="Image resolution"
    )
    parser.add_argument("-p", "--samples", default=200, type=int, help="Cycles samples")
    parser.add_argument("-l", "--lighting", default=0, type=int, help="Lighting seed")
    parser.add_argument(
        "-Z",
        "--cam_zoff",
        "--z_offset",
        type=float,
        default=0.0,
        help="Additional offset on the camera's Z-axis for look-at positions",
    )
    parser.add_argument(
        "-s",
        "--save_blend",
        action="store_true",
        default=True,
        help="Whether to save the .blend file",
    )
    parser.add_argument(
        "-e", "--elevation", default=60, type=float, help="Sun elevation angle"
    )
    parser.add_argument(
        "--cam_dist",
        default=0,
        type=float,
        help="Distance from the camera to the look-at position",
    )
    parser.add_argument(
        "-a", "--cam_angle", default=(-30, 0, 45), type=float, nargs="+",
        help="Camera rotation in XYZ"
    )
    parser.add_argument(
        "-O", "--offset", default=(0, 0, 0), type=float, nargs="+",
        help="Asset location offset"
    )
    parser.add_argument(
        "-c", "--cam_center", default=1, type=int, help="Camera rotation around center"
    )
    parser.add_argument(
        '--background_image_folder',
        type=Path,
        default=None,
        help='Folder of background images to use as environment textures'
    )
    parser.add_argument(
        '--texture_path',
        type=Path,
        default=None,
        help='Path to a .png texture image to apply to the generated mesh'
    )
    parser.add_argument(
        "-r", "--render", default="none", choices=["image", "video", "none"],
        help="Render mode"
    )
    parser.add_argument(
        "-b",
        "--best_ratio",
        default=9/16,
        type=float,
        help="Aspect ratio for asset grid"
    )
    parser.add_argument("-F", "--fire", action="store_true", help="Set the object on fire")
    parser.add_argument("-I", "--fire_res", default=100, type=int, help="Fire resolution")
    parser.add_argument("-U", "--fire_duration", default=30, type=int, help="Fire duration")
    parser.add_argument(
        "-t", "--film_transparent", default=1, type=int,
        help="Use transparent background"
    )
    parser.add_argument(
        "-E", "--frame_end", type=int, default=120, help="End frame (for video rendering)"
    )
    parser.add_argument(
        "-g", "--gpu", action="store_true", help="Use GPU for rendering"
    )
    parser.add_argument("-C", "--cycles", type=float, default=1, help="Video cycles")
    parser.add_argument(
        "-A", "--scale_reference", action="store_true", help="Add a scale reference object"
    )
    parser.add_argument(
        "-S", "--skip_existing", action="store_true", help="Skip existing .blend files"
    )
    parser.add_argument(
        "-P", "--postprocessing_only", action="store_true", help="Only run postprocessing"
    )
    parser.add_argument(
        "-D", "--seed", type=int, default=-1, help="Fixed random seed override"
    )
    parser.add_argument(
        "-N", "--no-mod", action="store_true", help="Do not modify geometry after creation"
    )
    parser.add_argument("-H", "--hdri", action="store_true", help="Add HDRI lighting")
    parser.add_argument(
        "-T", "--three_point", action="store_true", help="Add 3-point lighting"
    )
    parser.add_argument(
        "-G", "--no_ground", action="store_true", help="Do not create ground plane"
    )
    parser.add_argument(
        "-W", "--spawn_placeholder", action="store_true", help="Spawn placeholder object"
    )
    parser.add_argument("-z", "--zoom", action="store_true", help="Zoom first figure")

    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--slurm", action="store_true")

    parser.add_argument(
        "--export",
        type=str,
        default=None,
        choices=export.FORMAT_CHOICES,
        help="Export format for geometry"
    )
    parser.add_argument("--export_texture_res", type=int, default=1024)
    parser.add_argument(
        "-d", "--debug", type=str, nargs="*", default=None,
        help="Debug logging for specified modules"
    )
    parser.add_argument("--dryrun", action="store_true", help="Dry run for testing")

    return init.parse_args_blender(parser)


if __name__ == "__main__":
    args = make_args()

    args.no_mod = args.no_mod or args.fire
    args.film_transparent = args.film_transparent and not args.hdri
    args.save_blend = True
    args.render = "none"

    with FixedSeed(1):
        main(args)
