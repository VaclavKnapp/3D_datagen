import bpy
import random
import numpy as np
from bpy.props import (
        BoolProperty,
        StringProperty
        )

from .utils import uvcalc_smart_project
from .utils import generate
from .properties import ShapeGeneratorConfig, small_shape_collection_name, medium_shape_collection_name

from .ui import draw_properties

import os
import time

preset_dir = 'operator/mesh.shape_generator'


class SG_OT_AddShapeGenerator(bpy.types.Operator, ShapeGeneratorConfig):
    """Add a generated shape"""
    bl_idname = "mesh.shape_generator"
    bl_label = "Shape Generator"
    bl_options = {'REGISTER', 'UNDO', 'PRESET'}

    def draw(self, context):
        """Layout the addon options"""
        layout = self.layout
        draw_properties(layout, self)


    def invoke(self, context, event):
        """Invoke the addon"""
        return self.execute(context)

    def execute(self, context):
        context.view_layer.objects.active = None

        new_objs = generate.generate_shapes(self, context)

        if not len(new_objs):
            return {'CANCELLED'}

        return {'FINISHED'}

class SG_OT_UpdateShapeGenerator(bpy.types.Operator):
    """Update a generated shape"""
    bl_idname = "mesh.shape_generator_update"
    bl_label = "Shape Generator Update"
    bl_options = {'INTERNAL', 'UNDO'}

    @classmethod
    def poll(self, context):
        return context.active_object and context.active_object.shape_generator_collection

    def invoke(self, context, event):
        """Invoke the addon"""
        return self.execute(context)

    def execute(self, context):

        collection = context.active_object.shape_generator_collection
        
        if collection == None:
            return

        # Get objects in the collection if they are meshes
        meshes = set()
        for obj in [o for o in collection.all_objects if o.type == 'MESH']:
            # Store the internal mesh
            meshes.add( obj.data )
            # Delete the object
            bpy.data.objects.remove( obj )

        # Look at meshes that are orphean after objects removal
        for mesh in [m for m in meshes if m.users == 0]:
            # Delete the meshes
            bpy.data.meshes.remove( mesh )

        # delete any sub collections made by the parent.
        for collection_to_remove in collection.children:
            bpy.data.collections.remove(collection_to_remove)


        # regenerate the shape.
        config = collection.shape_generator_properties
        config.auto_update = False
        try:
            generate.generate_shapes(config, context, collection)
        finally:
            config.auto_update = True

        return {'FINISHED'}

class SG_OT_DeleteShapeGenerator(bpy.types.Operator):
    """Delete a generated shape"""
    bl_idname = "mesh.shape_generator_delete"
    bl_label = "Shape Generator Delete"
    bl_options = {'INTERNAL', 'UNDO'}

    @classmethod
    def poll(self, context):
        return context.active_object and context.active_object.shape_generator_collection

    def invoke(self, context, event):
        """Invoke the addon"""
        return self.execute(context)

    def execute(self, context):

        collection = context.active_object.shape_generator_collection
        
        if collection == None:
            return

        # Get objects in the collection if they are meshes
        meshes = set()
        for obj in [o for o in collection.all_objects if o.type == 'MESH']:
            # Store the internal mesh
            meshes.add( obj.data )
            # Delete the object
            bpy.data.objects.remove( obj )

        # Look at meshes that are orphean after objects removal
        for mesh in [m for m in meshes if m.users == 0]:
            # Delete the meshes
            bpy.data.meshes.remove( mesh )

        # delete any sub collections made by the parent.
        for collection_to_remove in collection.children:
            bpy.data.collections.remove(collection_to_remove)

        # remove the main collection.
        bpy.data.collections.remove(collection)
    

        return {'FINISHED'}

def get_config(context):
    return context.active_object.shape_generator_collection.shape_generator_properties

class SG_OT_Iterator(bpy.types.Operator):
    """Start Shape Generator Iterator"""
    bl_idname = "mesh.shapegenerator_iterator"
    bl_label = "Shape Generator Iterator Run"
    bl_options = {"INTERNAL", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.shape_generator_collection and bpy.data.filepath

    def execute(self, context):
        file_path = context.scene.shape_generator_iterator.file_path
        start_seed = context.scene.shape_generator_iterator.start_seed
        end_seed = context.scene.shape_generator_iterator.end_seed

        config = get_config(context)

        old_master_seed = config.random_seed
        old_file_path = context.scene.render.filepath
        old_render_engine = context.scene.render.engine

        wm = bpy.context.window_manager
        wm.progress_begin(0, abs(end_seed - start_seed) + 1)

        try:

            if context.scene.shape_generator_iterator.render_engine != 'SAME':
                # temporarily set the render engine.
                context.scene.render.engine = context.scene.shape_generator_iterator.render_engine 

        
            if not os.path.exists(file_path):
                os.makedirs(file_path)

            ack_path = os.path.join(file_path, 'running.ack')
            open(ack_path, 'a').close()

            # bpy.ops.wm.save_mainfile(filepath=bpy.data.filepath)

            i = 0
            for seed in range(start_seed, end_seed + 1):
                start = time.time()

                # first check if we should abort because the .ack file is no longer there.
                try:
                    f = open (ack_path)
                except IOError:
                    # abort the run if there is an error opening the file.
                    report_message = 'ITERATE run aborted.'
                    self.report({'INFO'}, report_message)
                    break
                finally:
                    f.close()

                print('Commence Iterator with SEED: ', seed)

                # perform the next iteraton.
                config.random_seed = seed
                context.scene.render.filepath = os.path.join(file_path, 'shape_generator_' + str(seed))
                bpy.ops.render.render(write_still = True)

                end = time.time()

                print('Shape Generator Time Taken: ', str(end-start))

                i += 1
                wm.progress_update(i)

        finally:
            config.random_seed = old_master_seed
            context.scene.render.filepath = old_file_path
            context.scene.render.engine = old_render_engine
            wm.progress_end()

        return {'FINISHED'}

class SG_OT_BakeShapeGenerator(bpy.types.Operator):
    """Bake a generated shape"""
    bl_idname = "mesh.shape_generator_bake"
    bl_label = "Shape Generator Bake"
    bl_description = "Join all generated shapes together and disable properties"
    bl_options = {'INTERNAL', 'UNDO'}

    @classmethod
    def poll(self, context):
        return context.active_object and context.active_object.shape_generator_collection

    def invoke(self, context, event):
        """Invoke the addon"""
        return self.execute(context)

    def execute(self, context):
        bpy.ops.object.mode_set(mode='OBJECT')

        # for all objects in the shape generator collection, disable their properties and join them together.
        original_collection = context.active_object.shape_generator_collection
        all_objects = original_collection.all_objects
        if len(all_objects):
            # get first object created (assumes this is the main object))
            for obj in bpy.data.objects:
                obj.select_set(False)

            bool_objects = []
            for obj in all_objects:
                obj.select_set(True)
                context.view_layer.objects.active = obj
                obj.shape_generator_collection = None
                for mod in obj.modifiers:
                    if mod.type == 'BOOLEAN':
                        bool_objects.append(mod.object)
                    bpy.ops.object.modifier_apply(modifier=mod.name)
                obj.select_set(False)

            # remove any boolean targets.
            for obj in bool_objects:
                try:
                    old_data = obj.data
                    bpy.data.objects.remove(obj)
                    bpy.data.meshes.remove(old_data)
                except ReferenceError:
                    pass

            for obj in all_objects:
                try:
                    obj.select_set(True)
                except ReferenceError:
                    pass
            
            #This should be the first (main) object added.
            context.view_layer.objects.active = all_objects[0]

            #perform join operation
            bpy.ops.object.join()

            # remove any sub collections that have been created
            for collection_to_remove in original_collection.children:
                bpy.data.collections.remove(collection_to_remove)


            return {'FINISHED'}
        return {'CANCELLED'}

#
# Preset classes taken from \2.92\scripts\startup\bl_operators
#


class SG_OT_ExecuteShapeGeneratorPreset(bpy.types.Operator):
    """Apply a preset"""
    bl_idname = "script.execute_preset_shape_gen"
    bl_label = "Execute a Python Preset"
    bl_options = {"INTERNAL"}

    filepath: StringProperty(
        subtype='FILE_PATH',
        options={'SKIP_SAVE'},
    )
    menu_idname: StringProperty(
        name="Menu ID Name",
        description="ID name of the menu this was called from",
        options={'SKIP_SAVE'},
    )

    def execute(self, context):
        from os.path import basename, splitext
        filepath = self.filepath

        # change the menu title to the most recently chosen option
        preset_class = getattr(bpy.types, 'SG_PT_presets')
        preset_class.bl_label = bpy.path.display_name(basename(filepath))

        ext = splitext(filepath)[1].lower()

        if ext not in {".py", ".xml"}:
            self.report({'ERROR'}, "unknown filetype: %r" % ext)
            return {'CANCELLED'}

        if hasattr(preset_class, "reset_cb"):
            preset_class.reset_cb(context)



        obj = context.active_object

        cfg = obj.shape_generator_collection.shape_generator_properties

        if ext == ".py":
            old_auto_update = context.active_object.shape_generator_collection.shape_generator_properties.auto_update
            context.active_object.shape_generator_collection.shape_generator_properties.auto_update = False
            try:
                # bpy.utils.execfile(filepath)
                count = 0
                with open(filepath) as fp:
                    Lines = fp.readlines()
                    for line in Lines:
                        count += 1
                        if line.startswith("op."):
                            prop=line.strip()[(len('op.')):]
                            param, value = prop.split(" = ",1)
                            execLine = "context.active_object.shape_generator_collection.shape_generator_properties." + param + " = " + value
                            exec(execLine)
                

            except Exception as ex:
                self.report({'ERROR'}, "Failed to execute the preset: " + repr(ex))

            finally:
                context.active_object.shape_generator_collection.shape_generator_properties.auto_update = old_auto_update
                bpy.ops.mesh.shape_generator_update()
        elif ext == ".xml":
            import rna_xml
            rna_xml.xml_file_run(context,
                                 filepath,
                                 preset_class.preset_xml_map)

        if hasattr(preset_class, "post_cb"):
            preset_class.post_cb(context)

        return {'FINISHED'}


class AddPresetBase:
    """Base preset class, only for subclassing
    subclasses must define
     - preset_values
     - preset_subdir """
    # bl_idname = "script.preset_base_add"
    # bl_label = "Add a Python Preset"

    # only because invoke_props_popup requires. Also do not add to search menu.
    bl_options = {'REGISTER', 'INTERNAL'}

    name: StringProperty(
        name="Name",
        description="Name of the preset, used to make the path name",
        maxlen=64,
        options={'SKIP_SAVE'},
    )
    remove_name: BoolProperty(
        default=False,
        options={'HIDDEN', 'SKIP_SAVE'},
    )
    remove_active: BoolProperty(
        default=False,
        options={'HIDDEN', 'SKIP_SAVE'},
    )

    @staticmethod
    def as_filename(name):  # could reuse for other presets

        # lazy init maketrans
        def maketrans_init():
            cls = AddPresetBase
            attr = "_as_filename_trans"

            trans = getattr(cls, attr, None)
            if trans is None:
                trans = str.maketrans({char: "_" for char in " !@#$%^&*(){}:\";'[]<>,.\\/?"})
                setattr(cls, attr, trans)
            return trans

        name = name.lower().strip()
        name = bpy.path.display_name_to_filepath(name)
        trans = maketrans_init()
        # Strip surrounding "_" as they are displayed as spaces.
        return name.translate(trans).strip("_")

    def execute(self, context):
        import os
        from bpy.utils import is_path_builtin

        if hasattr(self, "pre_cb"):
            self.pre_cb(context)

        preset_menu_class = getattr(bpy.types, self.preset_menu)

        is_xml = getattr(preset_menu_class, "preset_type", None) == 'XML'
        is_preset_add = not (self.remove_name or self.remove_active)

        if is_xml:
            ext = ".xml"
        else:
            ext = ".py"

        name = self.name.strip() if is_preset_add else self.name

        if is_preset_add:
            if not name:
                return {'FINISHED'}

            # Reset preset name
            wm = bpy.data.window_managers[0]
            if name == wm.preset_name:
                wm.preset_name = 'New Preset'

            filename = self.as_filename(name)

            target_path = os.path.join("presets", self.preset_subdir)
            target_path = bpy.utils.user_resource('SCRIPTS',
                                                  target_path,
                                                  create=True)

            if not target_path:
                self.report({'WARNING'}, "Failed to create presets path")
                return {'CANCELLED'}

            filepath = os.path.join(target_path, filename) + ext

            if hasattr(self, "add"):
                self.add(context, filepath)
            else:
                print("Writing Preset: %r" % filepath)

                if is_xml:
                    import rna_xml
                    rna_xml.xml_file_write(context,
                                           filepath,
                                           preset_menu_class.preset_xml_map)
                else:

                    def rna_recursive_attr_expand(value, rna_path_step, level):
                        if isinstance(value, bpy.types.PropertyGroup):
                            for sub_value_attr in value.bl_rna.properties.keys():
                                if sub_value_attr == "rna_type":
                                    continue
                                sub_value = getattr(value, sub_value_attr)
                                rna_recursive_attr_expand(sub_value, "%s.%s" % (rna_path_step, sub_value_attr), level)
                        elif type(value).__name__ == "bpy_prop_collection_idprop":  # could use nicer method
                            file_preset.write("%s.clear()\n" % rna_path_step)
                            for sub_value in value:
                                file_preset.write("item_sub_%d = %s.add()\n" % (level, rna_path_step))
                                rna_recursive_attr_expand(sub_value, "item_sub_%d" % level, level + 1)
                        else:
                            # convert thin wrapped sequences
                            # to simple lists to repr()
                            try:
                                value = value[:]
                            except:
                                pass

                            file_preset.write("%s = %r\n" % (rna_path_step, value))

                    file_preset = open(filepath, 'w', encoding="utf-8")
                    file_preset.write("import bpy\n")

                    if hasattr(self, "preset_defines"):
                        for rna_path in self.preset_defines:
                            exec("op = context.active_object.shape_generator_collection.shape_generator_properties")
                            file_preset.write("%s\n" % rna_path)
                        file_preset.write("\n")

                    for rna_path in self.preset_values:
                        value = eval(rna_path)
                        rna_recursive_attr_expand(value, rna_path, 1)

                    file_preset.close()

            preset_menu_class.bl_label = bpy.path.display_name(filename)

        else:
            if self.remove_active:
                name = preset_menu_class.bl_label

            # fairly sloppy but convenient.
            filepath = bpy.utils.preset_find(name,
                                             self.preset_subdir,
                                             ext=ext)

            if not filepath:
                filepath = bpy.utils.preset_find(name,
                                                 self.preset_subdir,
                                                 display_name=True,
                                                 ext=ext)

            if not filepath:
                return {'CANCELLED'}

            # Do not remove bundled presets
            if is_path_builtin(filepath):
                self.report({'WARNING'}, "Unable to remove default presets")
                return {'CANCELLED'}

            try:
                if hasattr(self, "remove"):
                    self.remove(context, filepath)
                else:
                    os.remove(filepath)
            except Exception as e:
                self.report({'ERROR'}, "Unable to remove preset: %r" % e)
                import traceback
                traceback.print_exc()
                return {'CANCELLED'}

            # XXX, stupid!
            preset_menu_class.bl_label = "Presets"

        if hasattr(self, "post_cb"):
            self.post_cb(context)

        return {'FINISHED'}

    def check(self, _context):
        self.name = self.as_filename(self.name.strip())

    def invoke(self, context, _event):
        if not (self.remove_active or self.remove_name):
            wm = context.window_manager
            return wm.invoke_props_dialog(self)
        else:
            return self.execute(context)



class SG_OT_AddShapeGenPreset(AddPresetBase, bpy.types.Operator):
    bl_idname = 'mesh.add_shape_gen_preset'
    bl_label = 'Manage Preset'
    bl_description = "Manage the addition or removal of a preset"
    preset_menu = 'SG_MT_ShapeGenPresets'

    # Common variable used for all preset values
    preset_defines = [
                        'op = bpy.context.active_operator'
                     ]

    # Properties to store in the preset
    preset_values = [
                    'op.random_seed',
                    'op.amount',
                    'op.min_extrude',
                    'op.max_extrude',
                    'op.min_taper',
                    'op.max_taper',
                    'op.min_rotation',
                    'op.max_rotation',
                    'op.min_slide',
                    'op.max_slide',
                    'op.favour_vec',
                    'op.mirror_x',
                    'op.mirror_y',
                    'op.mirror_z',
                    'op.is_bevel',
                    'op.bevel_width',
                    'op.bevel_profile',
                    'op.bevel_method',
                    'op.bevel_segments',
                    'op.bevel_clamp_overlap',
                    'op.is_subsurf',
                    'op.subsurf_type',
                    'op.is_adaptive',
                    'op.adaptive_dicing',
                    'op.subsurf_subdivisions',
                    'op.subdivide_edges',
                    'op.shade_smooth',
                    'op.auto_smooth',
                    'op.flip_normals',
                    'op.prevent_ovelapping_faces',
                    'op.overlap_check_limit',
                    'op.align',
                    'op.location',
                    'op.rotation',
                    'op.scale',
                    'op.random_transform_seed',
                    'op.number_to_create',
                    'op.randomize_location',
                    'op.number_to_create',
                    'op.randomize_rotation',
                    'op.randomize_scale',
                    'op.start_rand_location',
                    'op.start_rand_rotation',
                    'op.start_rand_scale',
                    'op.end_rand_location',
                    'op.end_rand_rotation',
                    'op.end_rand_scale',
                    'op.uv_projection_limit',
                    'op.uv_island_margin',
                    'op.uv_area_weight',
                    'op.uv_stretch_to_bounds',
                    'op.material_to_use',
                    'op.is_boolean',
                    'op.is_parent_booleans',
                    'op.boolean_main_obj_index',
                    'op.main_obj_bool_operation',
                    'op.bool_solver',
                    'op.fast_overlap_threshold',
                    'op.exact_self',
                    'op.bool_hide_viewport',
                    'op.bool_hide_render',
                    'op.bool_display_type',
                    'op.big_random_seed',
                    'op.medium_random_seed',
                    'op.small_random_seed',
                    'op.medium_random_scatter_seed',
                    'op.small_random_scatter_seed',
                    'op.medium_shape_num',
                    'op.small_shape_num',
                    'op.big_shape_scale',
                    'op.medium_shape_scale',
                    'op.small_shape_scale',
                    'op.use_medium_shape_bool',
                    'op.medium_shape_bool_operation',
                    'op.use_small_shape_bool',
                    'op.small_shape_bool_operation',
                    'op.use_colors',
                    'op.big_shape_color',
                    'op.medium_shape_color',
                    'op.small_shape_color'
                    ]

    # Directory to store the presets
    preset_subdir = preset_dir

classes = [SG_OT_AddShapeGenPreset, SG_OT_AddShapeGenerator, SG_OT_UpdateShapeGenerator, SG_OT_DeleteShapeGenerator, SG_OT_ExecuteShapeGeneratorPreset, SG_OT_Iterator, SG_OT_BakeShapeGenerator]

def append_to_menu():
    """lets add ourselves to the main header"""
    bpy.types.VIEW3D_MT_mesh_add.append(shape_generator_menu_func)

def remove_from_menu():
    bpy.types.VIEW3D_MT_mesh_add.remove(shape_generator_menu_func)


def shape_generator_menu_func(self, context):
    self.layout.operator(SG_OT_AddShapeGenerator.bl_idname, icon='MOD_SOLIDIFY')

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()
