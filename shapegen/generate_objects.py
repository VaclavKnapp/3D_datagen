import os
import sys
import argparse
import pickle
import bpy
import numpy as np
from mathutils import Vector

def generate_object(params, clear=1):
    print('STARTING')
    if clear:
        delete_everything()

    bpy.ops.mesh.shape_generator(
        random_seed=params['seed'],
        mirror_x=0,
        mirror_y=0,
        mirror_z=0,
        favour_vec=params['axis_preferences'],
        amount=params['n_extrusions'],
        is_subsurf=(params['n_subdivisions'] > 0),
        subsurf_subdivisions=params['n_subdivisions'],
        big_shape_num=params['n_bigshapes'],
        medium_shape_num=params['n_medshapes'],
        medium_random_scatter_seed=params['med_locseed'],
        big_random_scatter_seed=params['big_locseed'],
        big_random_seed=params['big_shapeseed'],
        medium_random_seed=params['med_shapeseed'],

    )
    print('OVER')

    for obj in bpy.data.objects:
        obj.select_set(False)

def scale_and_center_object(xyz=(0,0,0)):
    """
    Scales the object so that it fits within a unit sphere and centers it at the specified location.
    """
    def bbox(ob):
        return (Vector(b) for b in ob.bound_box)
    def bbox_center(ob):
        return sum(bbox(ob), Vector()) / 8
    def mesh_radius(ob):
        o = bbox_center(ob)
        return max((v.co - o).length for v in ob.data.vertices)
    

    obj = bpy.context.selected_objects[0]
    scale_factor = 1 / mesh_radius(obj)
    obj.scale *= scale_factor
    obj.location = xyz

def delete_everything():
    # Delete all collections
    for coll in bpy.data.collections:
        bpy.data.collections.remove(coll)
    # Delete all objects
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj)

if __name__ == '__main__':

    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    
    parser = argparse.ArgumentParser(description="Generate random shapes with blocky and smooth options.")
    parser.add_argument("-n_extrusions", type=int,
                        help="Number of extrusions (default: 9)")
    parser.add_argument("-o", "--output", type=str,
                        default=os.path.join(os.getcwd(), "test"),
                        help="Output directory (default: <cwd>/test)")
    parser.add_argument("-n_shapes", type=int, default=10,
                        help="Number of random shapes to generate per smoothness type (default: 10)")
    args = parser.parse_args(argv)
    
    print("\nUsing parameters:")
    print(f"  n_extrusions: {args.n_extrusions}")
    print(f"  output dir:   {args.output}")
    print(f"  n_shapes:     {args.n_shapes}")
    print("\n")
    

    print('-------LOADING SHAPE GENERATOR------')
    addon_file_path = os.path.join(os.getcwd(), 'shape_generator', 'operators.py')
    if os.path.exists(addon_file_path):
        try:
            bpy.ops.preferences.addon_install(filepath=addon_file_path)
            bpy.ops.preferences.addon_enable(module='shape_generator')
            bpy.ops.wm.save_userpref()
            print('-------FINISHED LOADING SHAPE GENERATOR------')
        except Exception as e:
            print(f"Error installing addon: {e}")
    else:
        print(f"Addon file not found at {addon_file_path}. Please ensure the addon is installed.")
    

    smooth_output = os.path.join(args.output, str(args.n_extrusions), "smooth")
    blocky_output = os.path.join(args.output, str(args.n_extrusions), "blocky")
    os.makedirs(smooth_output, exist_ok=True)
    os.makedirs(blocky_output, exist_ok=True)
    

    stimulus_info = {}
    

    for smoothness in ['blocky', 'smooth']:
        shape_count = 0
        while shape_count < args.n_shapes:

            params = {
                'axis_preferences': [
                    np.random.uniform(0, 0.2),
                    np.random.uniform(0.6, 1),
                    np.random.uniform(0, 1)  
                ],
                'n_bigshapes': 1,
                'n_medshapes': 0,
                'big_shapeseed': np.random.randint(1000),
                'big_locseed': np.random.randint(1000),
                'med_locseed': np.random.randint(1000),
                'med_shapeseed': np.random.randint(1000),
                'n_extrusions': int(args.n_extrusions),
                'seed': np.random.randint(1000)
            }
            

            if smoothness == 'blocky':
                subdivisions = 0
            else:
                subdivisions = 5
            params['n_subdivisions'] = subdivisions
            

            params['brotations'] = [np.random.uniform(0, 0.2)]
            params['mrotations'] = [np.random.uniform(0, 0.2)]
            
            print(f"Generating {smoothness} shape {shape_count+1}/{args.n_shapes} with params:")
            print(params)
            

            delete_everything()
            generate_object(params)
            

            collections = bpy.data.collections

            med_coll = [coll for coll in collections if 'Medium' in coll.name]
            med_objects = med_coll[0].objects[:] if med_coll else []
            

            big_coll = [coll for coll in collections if coll.name == 'Generated Shape Collection']
            big_objects = big_coll[0].objects[:] if big_coll else []
            

            brotation = params['brotations'][0]
            for obj in big_objects:
                if obj.name != 'Generated Shape':
                    obj.rotation_euler.x -= brotation * np.pi
            

            mrotation = params['mrotations'][0]
            for obj in med_objects:
                obj.rotation_euler.x -= mrotation * np.pi
                obj.location.x += 0.3
            

            for obj in bpy.data.objects:
                obj.select_set(True)
            bpy.ops.object.join()
            bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='MEDIAN')
            bpy.ops.object.location_clear(clear_delta=False)
            scale_and_center_object()
            

            shape_name = f"shape_{smoothness}_{shape_count:03d}"
            if smoothness == 'smooth':
                save_path = os.path.join(smooth_output, shape_name + '.blend')
            else:
                save_path = os.path.join(blocky_output, shape_name + '.blend')
            bpy.ops.wm.save_as_mainfile(filepath=save_path)
            

            stimulus_info[shape_name] = params
            
            shape_count += 1

    print("\nGeneration complete!")
    print("Stimulus info for all objects:")
    print(stimulus_info)

