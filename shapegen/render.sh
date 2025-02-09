#!/bin/bash


light_setup="16_lights"
random_dim=true
complex=true


if [ "$#" -lt 3 ]; then
    echo "Usage: ./render_blend_files.sh /path/to/blend/folder /path/to/output/folder /path/to/background/folder [options]"
    exit 1
fi


blend_dir="$1"
output_dir="$2"
bg_dir="$3"


shift 3


while [ "$#" -gt 0 ]; do
    case "$1" in
        --complex)
            complex=true
            ;;
        --4_lights)
            light_setup="4_lights"
            ;;
        --6_lights)
            light_setup="6_lights"
            ;;
        --8_lights)
            light_setup="8_lights"
            ;;
        --16_lights)
            light_setup="16_lights"
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


if [ ! -d "$blend_dir" ]; then
    echo "Error: Blend directory '$blend_dir' does not exist."
    exit 1
fi

if [ ! -d "$bg_dir" ]; then
    echo "Error: Background images directory '$bg_dir' does not exist."
    exit 1
fi


mkdir -p "$output_dir"




python_script=$(mktemp)


trap 'rm -f "$python_script"' EXIT


cat <<'EOL' > "$python_script"
import bpy
import random
import math
import os
import sys
import mathutils


argv = sys.argv
argv = argv[argv.index("--") + 1:]  # get all args after '--'


object_name = None
output_dir = None
bg_dir = None
light_setup = "16_lights"  # default
random_dim = False
complex = False


i = 0
while i < len(argv):
    arg = argv[i]
    if arg == '--complex':
        complex = True
        i += 1
    elif arg == '--random_dim':
        random_dim = True
        i += 1
    elif arg in ('--4_lights', '--6_lights', '--8_lights', '--16_lights'):
        light_setup = arg.lstrip('--')
        i += 1
    else:
        if object_name is None:
            object_name = arg
        elif output_dir is None:
            output_dir = arg
        elif bg_dir is None:
            bg_dir = arg
        else:
            print(f"Unknown argument: {arg}")
            sys.exit(1)
        i +=1


if object_name is None or output_dir is None or bg_dir is None:
    print("Usage: blender -b <file.blend> --python <script.py> -- <object_name> <output_dir> <bg_dir> [--complex] [--4_lights | --6_lights | --8_lights | --16_lights] [--random_dim]")
    sys.exit(1)


bg_files = [os.path.join(bg_dir, f) for f in os.listdir(bg_dir) if f.lower().endswith('.png')]

if not bg_files:
    print("No background images found in", bg_dir)
    sys.exit(1)


def get_or_load_image(image_path):
    image_name = os.path.basename(image_path)
    if image_name in bpy.data.images:
        return bpy.data.images[image_name]
    else:
        return bpy.data.images.load(image_path)


def spherical_to_cartesian(radius, theta, phi):
    x = radius * math.sin(theta) * math.cos(phi)
    y = radius * math.sin(theta) * math.sin(phi)
    z = radius * math.cos(theta)
    return (x, y, z)


def look_at(obj_camera, point):
    direction = point - obj_camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    obj_camera.rotation_euler = rot_quat.to_euler()


def generate_sphere_points(n_points, radius):
    points = []
    offset = 2.0 / n_points
    increment = math.pi * (3.0 - math.sqrt(5.0))  # golden angle

    for i in range(n_points):
        y = ((i * offset) - 1) + (offset / 2)
        r = math.sqrt(1 - y * y)
        phi = i * increment
        x = math.cos(phi) * r
        z = math.sin(phi) * r
        points.append((x * radius, y * radius, z * radius))
    return points


world = bpy.context.scene.world
world.use_nodes = True
node_tree = world.node_tree
nodes = node_tree.nodes
links = node_tree.links


nodes.clear()


env_tex_node = nodes.new(type='ShaderNodeTexEnvironment')


bg_node = nodes.new(type='ShaderNodeBackground')


output_node = nodes.new(type='ShaderNodeOutputWorld')


links.new(env_tex_node.outputs['Color'], bg_node.inputs['Color'])
links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])


if bpy.context.scene.camera is None:
    bpy.ops.object.camera_add()
    bpy.context.scene.camera = bpy.context.object


bpy.ops.object.select_by_type(type='LIGHT')
bpy.ops.object.delete()


bpy.context.scene.render.resolution_x = 518
bpy.context.scene.render.resolution_y = 518


min_angle = math.radians(35)


previous_camera_vectors = []

def generate_camera_position(radius, previous_positions, min_angle):
    max_attempts = 10000  
    attempts = 0
    while attempts < max_attempts:
        attempts += 1

        if complex:
            theta0 = math.pi / 2  
            phi0 = math.pi       
            delta_theta = math.pi / 2.3  
            delta_phi = math.pi / 2.3    
            theta_min = max(0, theta0 - delta_theta)
            theta_max = min(math.pi, theta0 + delta_theta)
            phi_min = phi0 - delta_phi
            phi_max = phi0 + delta_phi
            theta = random.uniform(theta_min, theta_max)
            phi = random.uniform(phi_min, phi_max)
        else:
            theta = random.uniform(0, math.pi)
            phi = random.uniform(0, 2 * math.pi)


        x_unit, y_unit, z_unit = spherical_to_cartesian(1, theta, phi)
        new_vector = mathutils.Vector((x_unit, y_unit, z_unit))


        acceptable = True
        for vec in previous_positions:
            dot_prod = new_vector.dot(vec)
            dot_prod = max(min(dot_prod, 1.0), -1.0)  
            angle = math.acos(dot_prod)
            if angle < min_angle:
                acceptable = False
                break
        if acceptable:

            x, y, z = spherical_to_cartesian(radius, theta, phi)
            return (x, y, z), new_vector

    print(f"Could not find acceptable camera position after {max_attempts} attempts")
    return None, None


for i in range(10):

    bg_image_path = random.choice(bg_files)

    bg_image = get_or_load_image(bg_image_path)

    env_tex_node.image = bg_image


    base_radius = 10  
    camera_radius = base_radius * 0.37  

    camera_position, new_vector = generate_camera_position(camera_radius, previous_camera_vectors, min_angle)
    if camera_position is None:
        break  

    bpy.context.scene.camera.location = camera_position
    look_at(bpy.context.scene.camera, mathutils.Vector((0, 0, 0)))  


    previous_camera_vectors.append(new_vector)


    if light_setup in ('4_lights', '6_lights', '8_lights', '16_lights'):
        n_lights = int(light_setup.split('_')[0])  # Extract number of lights
        light_radius = base_radius
        light_positions = generate_sphere_points(n_lights, light_radius)
        lights = []
        for pos in light_positions:
            bpy.ops.object.light_add(type='POINT', location=pos)
            light = bpy.context.object
            max_energy = 2000  
            min_energy = max_energy * 0.30  
            if random_dim:
                light.data.energy = random.uniform(min_energy, max_energy)
            else:
                light.data.energy = max_energy
            lights.append(light)
    else:

        light_radius = base_radius  


        key_theta = math.radians(60)  
        key_phi = math.radians(30)    
        key_light_position = spherical_to_cartesian(light_radius, key_theta, key_phi)
        bpy.ops.object.light_add(type='POINT', location=key_light_position)
        key_light = bpy.context.object
        key_light.data.energy = 10000  


        fill_theta = math.radians(70)
        fill_phi = math.radians(150)
        fill_light_position = spherical_to_cartesian(light_radius, fill_theta, fill_phi)
        bpy.ops.object.light_add(type='POINT', location=fill_light_position)
        fill_light = bpy.context.object
        fill_light.data.energy = 5000  

        back_theta = math.radians(110)
        back_phi = math.radians(-90)
        back_light_position = spherical_to_cartesian(light_radius, back_theta, back_phi)
        bpy.ops.object.light_add(type='POINT', location=back_light_position)
        back_light = bpy.context.object
        back_light.data.energy = 8000  


    output_file = os.path.join(output_dir, object_name, f"{i:03d}.png")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)


    bpy.context.scene.render.filepath = output_file
    bpy.ops.render.render(write_still=True)


    if light_setup in ('4_lights', '6_lights', '8_lights', '16_lights'):
        for light in lights:
            bpy.data.objects.remove(light, do_unlink=True)
    else:
        bpy.data.objects.remove(key_light, do_unlink=True)
        bpy.data.objects.remove(fill_light, do_unlink=True)
        bpy.data.objects.remove(back_light, do_unlink=True)

EOL


for blend_file in "$blend_dir"/*.blend; do

    if [ ! -e "$blend_file" ]; then
        echo "No .blend files found in $blend_dir"
        continue
    fi


    object_name=$(basename "$blend_file" .blend)


    blender_args=("/home/vaclav_knapp/blender-3.6.19-linux-x64/blender" "-b" "$blend_file" "--python" "$python_script" "--" "$object_name" "$output_dir" "$bg_dir")

    if [ "$complex" = true ]; then
        blender_args+=("--complex")
    fi

    if [ "$light_setup" != "" ]; then
        blender_args+=("--$light_setup")
    fi

    if [ "$random_dim" = true ]; then
        blender_args+=("--random_dim")
    fi


    "${blender_args[@]}"

done

echo "Rendering complete! All files saved to $output_dir"
