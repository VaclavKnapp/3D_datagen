#!/bin/bash

# Usage:
#   ./render_files.sh /path/to/input /path/to/output /path/to/background \
#       [--obj] [--complex] [--4_lights | --6_lights | --8_lights | --16_lights] [--random_dim]

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 /path/to/input /path/to/output /path/to/bg [--obj] [--complex] [--4_lights|--6_lights|--8_lights|--16_lights] [--random_dim]"
    exit 1
fi

input_dir="$1"
output_dir="$2"
bg_dir="$3"

shift 3


obj=false
light_setup="6_lights"    
random_dim=true
complex=true

while [ "$#" -gt 0 ]; do
    case "$1" in
        --obj)
            obj=true
            ;;
        --complex)
            complex=true
            ;;
        --4_lights|--6_lights|--8_lights|--16_lights)
            light_setup="${1#--}"  
            ;;
        --random_dim)
            random_dim=true
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done

if [ ! -d "$input_dir" ]; then
    echo "Error: input directory '$input_dir' does not exist."
    exit 1
fi

if [ ! -d "$bg_dir" ]; then
    echo "Error: background directory '$bg_dir' does not exist."
    exit 1
fi


mkdir -p "$output_dir"

python_script=$(mktemp)
trap 'rm -f "$python_script"' EXIT

cat << 'EOF' > "$python_script"
import bpy
import random
import math
import os
import sys
import json
import mathutils


argv = sys.argv
argv = argv[argv.index("--") + 1:]  # get all args after '--'

model_path     = None
output_dir     = None
bg_dir         = None
factory_name   = None
is_obj         = False
complex_mode   = False
random_dim     = False
light_setup    = "4_lights"  # default
num_renders    = 10          # how many images per file


if len(argv) < 3:
    print("Error: Not enough arguments passed to the python script.")
    sys.exit(1)

model_path   = argv[0]
output_dir   = argv[1]
bg_dir       = argv[2]

factory_name = None
# If there's a 4th argument, it's the "factory_name"
if len(argv) >= 4 and not argv[3].startswith("--"):
    factory_name = argv[3]
    extra_args_start_idx = 4
else:
    extra_args_start_idx = 3

# Parse the rest
i = extra_args_start_idx
while i < len(argv):
    arg = argv[i]
    if arg == "--obj":
        is_obj = True
    elif arg == "--complex":
        complex_mode = True
    elif arg == "--random_dim":
        random_dim = True
    elif arg in ("--4_lights", "--6_lights", "--8_lights", "--16_lights"):
        light_setup = arg.replace("--", "")  # e.g. "4_lights"
    else:
        print(f"Unknown option in python script: {arg}")
    i += 1


if factory_name is None:
    factory_name = "unknown_factory"

def spherical_to_cartesian(radius, theta, phi):
    """
    Convert spherical coords (radius, theta, phi) to cartesian (x, y, z).
    theta: 0=top (z-axis), pi=bottom
    phi: 0 along x-axis, pi=180, etc.
    """
    x = radius * math.sin(theta) * math.cos(phi)
    y = radius * math.sin(theta) * math.sin(phi)
    z = radius * math.cos(theta)
    return (x, y, z)

def generate_sphere_points(n_points, radius):
    """
    A fairly even distribution of n_points on a sphere (using 'Fibonacci' sphere).
    Returns a list of (x, y, z).
    """
    points = []
    offset = 2.0 / n_points
    inc = math.pi * (3.0 - math.sqrt(5.0))  # golden angle
    for i in range(n_points):
        y = (i * offset - 1) + (offset / 2)
        r = math.sqrt(1 - y*y)
        phi = i * inc
        x = math.cos(phi) * r
        z = math.sin(phi) * r
        points.append((x*radius, y*radius, z*radius))
    return points

def look_at(obj_camera, target_point):
    """
    Make the camera 'obj_camera' look at 'target_point' (a Vector).
    """
    direction = target_point - obj_camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    obj_camera.rotation_euler = rot_quat.to_euler()

def get_or_load_image(image_path):
    """
    If an image is already loaded in bpy.data, reuse it; otherwise load it fresh.
    """
    name = os.path.basename(image_path)
    if name in bpy.data.images:
        return bpy.data.images[name]
    else:
        return bpy.data.images.load(image_path)


if is_obj:

    bpy.ops.import_scene.obj(filepath=model_path)
else:

    pass

bpy.ops.object.select_all(action='DESELECT')

for obj in bpy.context.scene.objects:
    if obj.type == 'MESH':
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        # Move origin to object's geometric center
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        # Then place that origin at (0, 0, 0)
        obj.location = (0, 0, 0)
        obj.select_set(False)


world = bpy.context.scene.world
if not world:
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world

world.use_nodes = True
node_tree = world.node_tree
nodes = node_tree.nodes
links = node_tree.links
nodes.clear()

env_tex_node = nodes.new(type='ShaderNodeTexEnvironment')
bg_node = nodes.new(type='ShaderNodeBackground')
out_node = nodes.new(type='ShaderNodeOutputWorld')

links.new(env_tex_node.outputs['Color'], bg_node.inputs['Color'])
links.new(bg_node.outputs['Background'], out_node.inputs['Surface'])


if bpy.context.scene.camera is None:
    bpy.ops.object.camera_add()
    bpy.context.scene.camera = bpy.context.object

cam = bpy.context.scene.camera


bpy.ops.object.select_all(action='DESELECT')
for o in bpy.context.scene.objects:
    if o.type == 'LIGHT':
        o.select_set(True)
bpy.ops.object.delete()


base_radius = 10.0         
light_radius = base_radius 

n_lights = None
if light_setup in ("4_lights", "6_lights", "8_lights", "16_lights"):
    n_lights = int(light_setup.split("_")[0])  
else:

    n_lights = None

lights_created = []
if n_lights:

    positions = generate_sphere_points(n_lights, light_radius)
    for pos in positions:
        bpy.ops.object.light_add(type='POINT', location=pos)
        lamp = bpy.context.object
        max_energy = 1800.0
        min_energy = max_energy * 0.2  # 20%
        if random_dim:
            lamp.data.energy = random.uniform(min_energy, max_energy)
        else:
            lamp.data.energy = max_energy
        lights_created.append(lamp)
else:

    key_pos  = spherical_to_cartesian(light_radius, math.radians(60), math.radians(30))
    bpy.ops.object.light_add(type='POINT', location=key_pos)
    key_light = bpy.context.object
    key_light.data.energy = 10000
    lights_created.append(key_light)

    # Fill light
    fill_pos = spherical_to_cartesian(light_radius, math.radians(70), math.radians(150))
    bpy.ops.object.light_add(type='POINT', location=fill_pos)
    fill_light = bpy.context.object
    fill_light.data.energy = 5000
    lights_created.append(fill_light)

    # Back light
    back_pos = spherical_to_cartesian(light_radius, math.radians(110), math.radians(-90))
    bpy.ops.object.light_add(type='POINT', location=back_pos)
    back_light = bpy.context.object
    back_light.data.energy = 8000
    lights_created.append(back_light)


bg_files = [os.path.join(bg_dir, f) for f in os.listdir(bg_dir)
            if f.lower().endswith(".png") or f.lower().endswith(".jpg")]
if not bg_files:
    print(f"No background images found in {bg_dir}. Aborting.")
    sys.exit(1)


scene = bpy.context.scene
scene.render.resolution_x = 518
scene.render.resolution_y = 518



min_angle = math.radians(15)
previous_camera_vectors = []

def generate_camera_position(radius, prev_positions, min_angle):
    """
    Attempt random positions on a sphere, each at least 'min_angle' away
    from previously used directions.
    """
    max_attempts = 10000
    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        # "complex" region or entire sphere:
        if complex_mode:
            # Focus near equator around some angles
            theta0 = math.pi / 2  # 90 deg
            phi0   = math.pi      # 180 deg
            delta_theta = math.pi / 3.2
            delta_phi   = math.pi / 3.2
            theta_min = max(0, theta0 - delta_theta)
            theta_max = min(math.pi, theta0 + delta_theta)
            phi_min   = phi0 - delta_phi
            phi_max   = phi0 + delta_phi
            theta = random.uniform(theta_min, theta_max)
            phi   = random.uniform(phi_min,   phi_max)
        else:
            theta = random.uniform(0, math.pi)
            phi   = random.uniform(0, 2 * math.pi)

        # Convert to unit vector
        x_u, y_u, z_u = spherical_to_cartesian(1, theta, phi)
        new_vec = mathutils.Vector((x_u, y_u, z_u))

        # Check angle from previous
        acceptable = True
        for vec in prev_positions:
            dot_prod = new_vec.dot(vec)
            dot_prod = max(min(dot_prod, 1.0), -1.0)
            angle = math.acos(dot_prod)
            if angle < min_angle:
                acceptable = False
                break
        if acceptable:
            # scale up
            x, y, z = spherical_to_cartesian(radius, theta, phi)
            return (x, y, z), new_vec
    print("WARNING: Could not find acceptable camera position after many attempts.")
    return None, None


dataset_json_path = os.path.join(output_dir, "dataset.json")
if os.path.exists(dataset_json_path):
    with open(dataset_json_path, "r") as f:
        dataset_entries = json.load(f)
else:
    dataset_entries = []

# For naming
object_name = os.path.splitext(os.path.basename(model_path))[0]


for i in range(num_renders):

    chosen_bg = random.choice(bg_files)
    env_tex_node.image = get_or_load_image(chosen_bg)


    camera_radius = base_radius * 0.95  
    camera_loc, camera_dir = generate_camera_position(
        camera_radius, previous_camera_vectors, min_angle
    )
    if not camera_loc:

        break
    previous_camera_vectors.append(camera_dir)

    cam.location = camera_loc
    look_at(cam, mathutils.Vector((0, 0, 0)))


    subfolder = os.path.join(output_dir, os.path.basename(factory_name), object_name)
    os.makedirs(subfolder, exist_ok=True)

    out_path = os.path.join(subfolder, f"{i:03d}.png")
    scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)


    extrinsic = cam.matrix_world.inverted()


    extrinsic_4x4 = []
    for row_i in range(4):
        row = []
        for col_i in range(4):
            row.append(extrinsic[row_i][col_i])
        extrinsic_4x4.append(row)


    dataset_entries.append({
        "image_path": out_path,
        "extrinsic_matrix": extrinsic_4x4
    })


bpy.ops.object.select_all(action='DESELECT')
for lamp in lights_created:
    lamp.select_set(True)
bpy.ops.object.delete()


with open(dataset_json_path, "w") as f:
    json.dump(dataset_entries, f, indent=2)

print("Done rendering", model_path)
EOF


if [ "$obj" = true ]; then
    echo "[INFO] Processing *.obj files in $input_dir"

    shopt -s nullglob
    obj_files=("$input_dir"/*.obj)
    if [ ${#obj_files[@]} -eq 0 ]; then
        echo "No .obj files found in $input_dir"
        exit 0
    fi

    for obj_file in "${obj_files[@]}"; do

        object_name=$(basename "$obj_file" .obj)
        factory_name=$(dirname "$obj_file")


        blender_cmd=(
            "/home/vaclav_knapp/blender-3.6.19-linux-x64/blender"
            "-b"
            "--factory-startup"
            "--python" "$python_script"
            "--"
            "$obj_file"           # 1) model path
            "$output_dir"         # 2) output dir
            "$bg_dir"             # 3) bg dir
            "$factory_name"       # 4) factory_name
            "--obj"               # indicates we must import .obj
        )


        if [ "$complex" = true ]; then
            blender_cmd+=("--complex")
        fi
        if [ "$random_dim" = true ]; then
            blender_cmd+=("--random_dim")
        fi
        if [ -n "$light_setup" ]; then
            blender_cmd+=("--$light_setup")
        fi

        echo "[INFO] Rendering OBJ: $obj_file"
        "${blender_cmd[@]}"
    done

else
    echo "[INFO] Processing *.blend files in $input_dir"

    shopt -s nullglob
    blend_files=("$input_dir"/*.blend)
    if [ ${#blend_files[@]} -eq 0 ]; then
        echo "No .blend files found in $input_dir"
        exit 0
    fi

    for blend_file in "${blend_files[@]}"; do

        object_name=$(basename "$blend_file" .blend)
        factory_name=$(dirname "$blend_file")

        blender_cmd=(
            "/home/vaclav_knapp/blender-3.6.19-linux-x64/blender"
            "-b"
            "$blend_file"
            "--python" "$python_script"
            "--"
            "$blend_file"       # 1) model path
            "$output_dir"       # 2) output dir
            "$bg_dir"           # 3) bg dir
            "$factory_name"     # 4) factory_name
            # no --obj here
        )


        if [ "$complex" = true ]; then
            blender_cmd+=("--complex")
        fi
        if [ "$random_dim" = true ]; then
            blender_cmd+=("--random_dim")
        fi
        if [ -n "$light_setup" ]; then
            blender_cmd+=("--$light_setup")
        fi

        echo "[INFO] Rendering BLEND: $blend_file"
        "${blender_cmd[@]}"
    done
fi

echo "Rendering complete! Check $output_dir for results."
