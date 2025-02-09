
import pickle, bpy, os, numpy as np 
from mathutils import Vector

def generate_object(params, clear=1): 
    print('STARTING')     
    if clear: delete_everything() 

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

    [i.select_set(False) for i in bpy.data.objects]

def scale_and_center_object(xyz=(0,0,0)):

    def bbox(ob):
        return (Vector(b) for b in ob.bound_box)
    def bbox_center(ob):
        return sum(bbox(ob), Vector()) / 8
    def mesh_radius(ob):
        o = bbox_center(ob)
        return max((v.co - o).length for v in ob.data.vertices)

    object = bpy.context.selected_objects[0]

    scale_factor = 1 / mesh_radius(object)
    object.scale *= scale_factor

    obj = bpy.context.active_object
    obj.location = (xyz[0], xyz[1], xyz[2])
    
def delete_everything(): 

    for collection in bpy.data.collections:
        bpy.data.collections.remove(collection)

    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj)

if __name__ == '__main__': 
    print('\n\n\n\n-------LOADING SHAPE GENERATOR------\n\n\n\n')     

    addon_file_path = os.path.join(os.getcwd(), 'shape_generator', 'operators.py')
    if os.path.exists(addon_file_path):
        try:
            bpy.ops.preferences.addon_install(filepath=addon_file_path)
            bpy.ops.preferences.addon_enable(module='shape_generator')
            bpy.ops.wm.save_userpref()
            print('\n\n\n-------FINISHED LOADING SHAPE GENERATOR\n\n\n\n')
        except Exception as e:
            print(f"Error installing addon: {e}")
    else:
        print(f"Addon file not found at {addon_file_path}. Please ensure the addon is installed.")
    print('\n\n\n-------FINISHED LOADING SHAPE GENERATOR\n\n\n\n')  

    num_sets = 2  # Number of different parameter sets you want to generate

    stimulus_info = {} 
    this_folder = os.path.split(os.getcwd())[1]
    object_folder = os.path.join(os.getcwd(), 'test')
    save_location = object_folder
    save_string = 'set%02d_%s_var%02d_n-extrusions:%s'  
    save_objects = 1 

    for set_num in range(num_sets):

        random_seed = np.random.randint(1000)
        np.random.seed(random_seed) 


        base_params = {
            'axis_preferences': [
                np.random.uniform(0, 0.2), 
                np.random.uniform(0.6, 1), 
                np.random.uniform(0.6, 1)
            ],
            'n_bigshapes': 1,
            'n_medshapes': 0,
            'big_shapeseed': np.random.randint(1000), 
            'big_locseed': 0,
            'med_locseed': np.random.randint(1000), 
            'med_shapeseed': np.random.randint(1000),
            'n_extrusions': 9,
            'seed': random_seed,

        }

        print(f'Generating set {set_num+1}/{num_sets} with base parameters:', base_params)

        for smoothness in ['blocky', 'smooth']:
            if smoothness == 'blocky':
                subdivisions = 0
            else:
                subdivisions = 5

            brotations = [0, .2]  
            mrotations = [0, .2]  

            variation_num = 0

            for brotation in brotations:
                for mrotation in mrotations:
                    params = base_params.copy()
                    params['n_subdivisions'] = subdivisions
                    params['brotations'] = [brotation]
                    params['mrotations'] = [mrotation]


                    params['axis_preferences'] = [
                        np.random.uniform(0, 0.2), 
                        np.random.uniform(0.6, 1), 
                        np.random.uniform(0.6, 1)
                    ]
                    params['big_shapeseed'] = np.random.randint(1000)
                    params['med_shapeseed'] = np.random.randint(1000)
                    params['seed'] = np.random.randint(1000)
                    params['med_locseed'] = np.random.randint(1000)
                    params['big_locseed'] = np.random.randint(1000)


                    delete_everything()

                    generate_object(params)


                    collections = bpy.data.collections
                    med_coll = [i.name for i in collections if 'Medium' in i.name]
                    med_objects = [i.name for i in collections[med_coll[0]].objects] if med_coll else []
                    
                    big_coll = [i.name for i in collections if 'Generated Shape Collection' == i.name]
                    big_objects = [i.name for i in collections[big_coll[0]].objects] if big_coll else []


                    for i_object in [i for i in big_objects if i != 'Generated Shape']: 
                        i_rotation = brotation * np.pi
                        bpy.data.objects[i_object].rotation_euler.x -= i_rotation


                    for i_object in med_objects: 
                        i_rotation = mrotation * np.pi
                        bpy.data.objects[i_object].rotation_euler.x -= i_rotation
                        bpy.data.objects[i_object].location.x += .3

                    for obj in bpy.data.objects:
                        obj.select_set(True)
                    bpy.ops.object.join()
                    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='MEDIAN')
                    bpy.ops.object.location_clear(clear_delta=False)

                    scale_and_center_object()


                    object_name = save_string % (
                        set_num, smoothness, variation_num, params['n_extrusions']
                    )
                    save_name = os.path.join(save_location, object_name + '.blend')
                    if save_objects: 
                        bpy.ops.wm.save_as_mainfile(filepath=save_name)


                    stimulus_info[object_name] = params

                    variation_num += 1
