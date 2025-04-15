import argparse
import importlib
import logging
import os
import traceback
from pathlib import Path
from multiprocessing import Pool

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
from infinigen.assets.utils.decorate import read_base_co, read_co
from infinigen.assets.utils.misc import assign_material, subclasses
from infinigen.core import init, surface
from infinigen.core.init import configure_cycles_devices
from infinigen.core.placement import AssetFactory
# noinspection PyUnresolvedReferences
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.test_utils import load_txt_list

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

        # Always center the object at origin
        # Remove drivers if applicable
        if parent.animation_data is not None:
            for d in parent.animation_data.drivers.values():
                parent.driver_remove(d.data_path)

        # Read coordinates and center object at origin
        co = read_co(asset)  # co is Nx3 array
        x_min, x_max = np.amin(co, axis=0), np.amax(co, axis=0)
        center_xyz = (x_min + x_max) / 2.0
        parent.location = -center_xyz
        butil.apply_transform(parent, loc=True)

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

        # Build a plane with reduced subdivisions
        bpy.ops.mesh.primitive_grid_add(size=10, x_subdivisions=100, y_subdivisions=100)
        plane = bpy.context.active_object
        material = bpy.data.materials.new("plane")
        material.use_nodes = True
        material.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (
            0.015, 0.009, 0.003, 1
        )
        assign_material(plane, material)

        # Apply scatter with reduced density if possible
        if isinstance(scatter, type):
            scatter = scatter(idx)
        scatter.apply(plane)
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

                # Build a sphere with fewer subdivisions
                if hasattr(template, "make_sphere"):
                    asset = template.make_sphere()
                else:
                    bpy.ops.mesh.primitive_ico_sphere_add(radius=0.8, subdivisions=4)
                    asset = bpy.context.active_object

                # If the template is a class, instantiate it
                if isinstance(template, type):
                    template = template(idx)

                template.apply(asset)

        except ModuleNotFoundError:
            raise Exception(f"{factory_name} not Found.")

    # Apply a PNG texture to the surface if specified
    if args.texture_path:
        apply_texture_from_path(asset, args.texture_path)

    return asset


def configure_low_complexity_render():
    """Configure render settings for low complexity/fast rendering"""
    # Set cycles render samples to 1/4 of 8192 (based on user's log)
    bpy.context.scene.cycles.samples = 2048
    
    # Reduce other complexity settings
    bpy.context.scene.cycles.max_bounces = 4
    bpy.context.scene.cycles.diffuse_bounces = 2
    bpy.context.scene.cycles.glossy_bounces = 2
    bpy.context.scene.cycles.transmission_bounces = 4
    bpy.context.scene.cycles.volume_bounces = 0
    bpy.context.scene.cycles.transparent_max_bounces = 4
    
    # Use simpler denoising
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.cycles.denoiser = 'OPENIMAGEDENOISE'  # Simpler denoiser
    
    # Lower subdivision levels
    for obj in bpy.data.objects:
        for mod in obj.modifiers:
            if mod.type == 'SUBSURF':
                mod.levels = max(1, mod.levels // 2)  # Cut subdivision in half
                mod.render_levels = max(1, mod.render_levels // 2)


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

    # Configure for low complexity rendering
    configure_low_complexity_render()
    
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

    # Use simplified configuration when possible
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
        '--texture_path',
        type=Path,
        default=None,
        help='Path to a .png texture image to apply to the generated mesh'
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
        "-W", "--spawn_placeholder", action="store_true", help="Spawn placeholder object"
    )
    parser.add_argument(
        "--simplify", action="store_true", default=True, 
        help="Use simplified geometry and render settings"
    )

    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--slurm", action="store_true")
    parser.add_argument(
        "-d", "--debug", type=str, nargs="*", default=None,
        help="Debug logging for specified modules"
    )
    parser.add_argument("--dryrun", action="store_true", help="Dry run for testing")

    return init.parse_args_blender(parser)


if __name__ == "__main__":
    args = make_args()
    
    # Set defaults - no ground, simplified rendering
    args.no_ground = True
    
    with FixedSeed(1):
        main(args)
