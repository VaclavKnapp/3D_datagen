import bpy
from bpy.utils import register_class, unregister_class
from bpy.types import Panel, Menu
from .ui import draw_properties
from .operators import preset_dir

from bl_ui.utils import PresetPanel


class SG_MT_ShapeGenPresets(Menu): 
    bl_label = 'My Presets' 
    preset_subdir = preset_dir
    preset_operator = 'script.execute_preset_shape_gen' 
    draw = Menu.draw_preset

class SG_PT_presets(PresetPanel, Panel):
    bl_label = 'My Presets'
    preset_subdir = preset_dir
    preset_operator = 'script.execute_preset_shape_gen'
    preset_add_operator = 'mesh.add_shape_gen_preset'

def can_show(context):
    if context.active_object and len([o for o in context.selected_objects if o == context.active_object]):
        collection = context.active_object.shape_generator_collection
        if collection:
            return True
    return False


class SG_PT_ui(Panel):
    bl_space_type = 'VIEW_3D'
    bl_label = 'Shape Generator'
    bl_region_type = 'UI'
    bl_category = 'Shape Generator'
    bl_idname = "SG_PT_Panel"
    

    def draw_header_preset(self, context):
        if can_show(context):
            SG_PT_presets.draw_panel_header(self.layout)

    def draw(self, context):
        layout = self.layout
        if can_show(context):
            collection = context.active_object.shape_generator_collection
            draw_properties(layout, collection.shape_generator_properties)
        else:
            layout.label(text="Select a Generated Shape.")


class SG_PT_UI_PT_IteratorPanel(Panel):
    """Properties panel for add-on operators."""
    bl_idname = "SG_PT_Panel_Iterator"
    bl_label = "Iterator"
    bl_category = "SHAPEGEN"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_parent_id = 'SG_PT_Panel'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        """Draw all options for the user to input to."""
        layout = self.layout

        if can_show(context):
            col = layout.column()
            col.label(text='Seed Range')
            row = col.row(align=True)
            row.prop(context.scene.shape_generator_iterator, 'start_seed', text="")
            row.prop(context.scene.shape_generator_iterator, 'end_seed', text="")

            col.label(text='File Path')
            col.prop(context.scene.shape_generator_iterator, 'file_path', text="")

            col.label(text='Render Engine')
            col.prop(context.scene.shape_generator_iterator, 'render_engine', text="")

            col.separator()
            col.operator('mesh.shapegenerator_iterator', text='Start')

            if not bpy.data.filepath:
                col_warn = col.column()
                col_warn.alert = True
                col_warn.label(text="Save .blend file before proceeding")

class SG_PT_UI_PT_OperationsPanel(Panel):
    """Properties panel for add-on operators."""
    bl_idname = "SG_PT_Operations"
    bl_label = "Operations"
    bl_category = "SHAPEGEN"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_parent_id = 'SG_PT_Panel'

    def draw(self, context):
        """Draw all options for the user to input to."""
        layout = self.layout

        if can_show(context):
            col = layout.column()
            col.operator("mesh.shape_generator_bake", text="Bake Shape")
            col.operator("mesh.shape_generator_delete", text="Delete Shape")


classes = [SG_MT_ShapeGenPresets, SG_PT_presets,
    SG_PT_ui, SG_PT_UI_PT_IteratorPanel, SG_PT_UI_PT_OperationsPanel]


def register():
    for cls in classes:
        register_class(cls)


def unregister():
    for cls in classes:
        unregister_class(cls)