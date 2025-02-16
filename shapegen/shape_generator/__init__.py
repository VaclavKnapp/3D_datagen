bl_info = {
    "name": "Random 3D Shape Generator",
    "author": "Mark Kingsnorth",
    "version": (1, 7, 9),
    "blender": (2, 80, 0),
    "location": "View3D > Mesh > Shape Generator",
    "description": "Creates a set of random extrusions to create different shapes",
    "warning": "",
    "wiki_url": "",
    "category": "Add Mesh",
    }

# To support reload properly, try to access a package var,
# if it's there, reload everything
if "bpy" in locals():
    import imp
    imp.reload(operators)
    print("Reloaded shape_generator files")
else:
    from . import operators

    print("Imported shape_generator")


import bpy
import os
import shutil
from . import properties, panel
from bpy.app.handlers import persistent

@persistent
def frame_change_post(dummy):

    if hasattr(bpy.context, 'active_object') and bpy.context.active_object and bpy.context.active_object.shape_generator_collection:
        bpy.ops.mesh.shape_generator_update()
        

def load_presets():
    """Load preset files if they have not been already"""
    presets_folder = bpy.utils.user_resource('SCRIPTS', "presets")
    my_presets = os.path.join(presets_folder, 'operator', 'mesh.shape_generator')
    if not os.path.isdir(my_presets):
        my_bundled_presets = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'presets')

        # makedirs() will also create all the parent folders (like "object")
        os.makedirs(my_presets)

        # Get a list of all the files in your bundled presets folder
        files = os.listdir(my_bundled_presets)

        # Copy them
        [shutil.copy2(os.path.join(my_bundled_presets, f), my_presets) for f in files]

def register():

    operators.register()
    properties.register()
    panel.register()

    operators.append_to_menu()

    load_presets()

    bpy.app.handlers.frame_change_post.append(frame_change_post)

def unregister():    
    bpy.app.handlers.frame_change_post.remove(frame_change_post)

    operators.remove_from_menu()

    operators.unregister()
    properties.unregister()
    panel.unregister()

if __name__ == "__main__":
    register()