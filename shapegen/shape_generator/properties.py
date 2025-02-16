import bpy
from bpy.types import PropertyGroup
from bpy.props import (
        BoolProperty,
        BoolVectorProperty,
        FloatProperty,
        FloatVectorProperty,
        IntProperty,
        EnumProperty,
        StringProperty,
        PointerProperty
        )

from bpy_extras.object_utils import AddObjectHelper
from .utils import generate
import random
import numpy as np


bool_items = [
    ("INTERSECT" , "Intersect" , "Keep the part of the mesh that is common between all operands"),
    ("UNION", "Union", "Combine meshes in an additive way"),
    ("DIFFERENCE" , "Difference" , "Combine meshes in a subtractive way")
]

medium_shape_collection_name = 'Medium Shape Collection'
small_shape_collection_name = 'Small Shape Collection'

def update_obj(self, context):

    if self.auto_update and \
        self.is_property_group and \
        context.active_object  and \
        len([o for o in context.selected_objects if o == context.active_object]):
        bpy.ops.mesh.shape_generator_update()
        

    return None


class ShapeGeneratorConfig():
    
    is_property_group : BoolProperty(default=False)
    auto_update : BoolProperty(default=True)

    # Cosmetics
    ###########
    update_draw_only : BoolProperty(default=False, options={'SKIP_SAVE'})
    def update_draw(self, context):
        self.update_draw_only = True
    show_seed_panel : BoolProperty(default=False,update=update_draw, options={'SKIP_SAVE'})
    show_extrude_panel : BoolProperty(default=False,update=update_draw, options={'SKIP_SAVE'})
    show_bevel_panel : BoolProperty(default=False,update=update_draw, options={'SKIP_SAVE'})
    show_subdivide_panel : BoolProperty(default=False,update=update_draw, options={'SKIP_SAVE'})
    show_mirror_panel : BoolProperty(default=False,update=update_draw, options={'SKIP_SAVE'})
    show_material_panel : BoolProperty(default=False,update=update_draw, options={'SKIP_SAVE'})
    show_overlap_faces_panel : BoolProperty(default=False,update=update_draw, options={'SKIP_SAVE'})
    show_translation_panel : BoolProperty(default=False,update=update_draw, options={'SKIP_SAVE'})
    show_randomisation_panel : BoolProperty(default=False,update=update_draw, options={'SKIP_SAVE'})
    show_sizing_panel : BoolProperty(default=False,update=update_draw, options={'SKIP_SAVE'})
    show_bool_panel : BoolProperty(default=False,update=update_draw, options={'SKIP_SAVE'})
    show_other_options_panel : BoolProperty(default=False,update=update_draw, options={'SKIP_SAVE'})
    show_uv_projection_panel : BoolProperty(default=False,update=update_draw, options={'SKIP_SAVE'})


    random_seed : IntProperty(
        name="Random Seed",
        description="Seed for generating extrusions",
        default=1000,
        step=1,
        update=update_obj,
        options={'ANIMATABLE'}
        )

    amount : IntProperty(
        name="Amount",
        description="Number of extrusions",
        min=0,
        default=5,
        soft_max=100,
        update=update_obj
        )

    min_extrude : FloatProperty(
            name="Min Extrude",
            description="Minimum extrude length",
            default=.5,
            step=1,
        update=update_obj
            )

    max_extrude : FloatProperty(
            name="Max Extrude",
            description="Maximum extrude length",
            default=1.0,
            step=1,
        update=update_obj
            )
    min_taper : FloatProperty(
            name="Min Taper",
            description="Minimum taper on extrusion",
            min=0,
            default=.5,
            step=1,
        update=update_obj
            )
    max_taper : FloatProperty(
            name="Max Taper",
            description="Maximum taper on extrusion",
            min=0,
            default=1.0,
            step=1,
        update=update_obj
            )
    min_rotation : FloatProperty(
            name="Min Rotation",
            description="Minimum rotation of the extruded face",
            default=0,
            step=3,
        update=update_obj
            )
    max_rotation : FloatProperty(
            name="Max Rotation",
            description="Maximum rotation of the extruded face",
            default=20,
            step=3,
        update=update_obj
            )

    min_slide : FloatProperty(
            name="Min Slide",
            description="Minimum slide of the extruded face",
            default=0,
            step=1,
        update=update_obj
            )
    max_slide : FloatProperty(
            name="Max Slide",
            description="Maximum slide of the extruded face",
            default=0.1,
            step=1,
        update=update_obj
            )

    favour_vec : FloatVectorProperty(
            name="Favour",
            description="Favour",
            subtype='XYZ',
            min=0,max=1,
            default=[1,1,1],
        update=update_obj
            )

    mirror_x : BoolProperty(
            name="Mirror X",
            description="Mirror along X axis",
            default=True,
        update=update_obj
            )

    mirror_y : BoolProperty(
            name="Mirror Y",
            description="Mirror along Y axis",
            default=False,
        update=update_obj
            )

    mirror_z : BoolProperty(
            name="Mirror Z",
            description="Mirror along Z axis",
            default=False,
        update=update_obj
            )

    is_bevel : BoolProperty(
            name="Bevel Edges",
            description="Bevel edges option",
            default=False,
        update=update_obj
            )

    bevel_width : FloatProperty(
            name="Bevel Width",
            description="Bevel width",
            default=0.1,
            precision=4,
            step=1,
            min=0,
        update=update_obj
            )

    bevel_profile : FloatProperty(
            name="Bevel Profile",
            description="The profile shape (0.5 = round)",
            default=0.5,
            precision=2,
            step=1,
            min=0,
            max=1,
            subtype='FACTOR',
        update=update_obj
            )

    bevel_method : EnumProperty(
            name = "Width Method",
            description = "Declares how Width will be interpreted to determine the amount of bevel",
            default="OFFSET",
            items = [
                ("OFFSET" , "Offset" , "Amount is offset of new edges from original"),
                ("WIDTH", "Width", "Amount is width of new face"),
                ("DEPTH" , "Depth" , "Amount is perpendicular distance of original edge to bevel face"),
                ("PERCENT", "Percent", "Amount is percent of adjacent edge length")
            ],
        update=update_obj
        )

    bevel_segments : IntProperty(
            name="Bevel Segments",
            description="Number of segments in the bevel",
            default=1,
            min=1,
        update=update_obj
            )

    bevel_clamp_overlap : BoolProperty(
            name="Clamp overlap",
            description="Clamp the width to avoid overlap",
            default=False,
        update=update_obj
            )

    is_subsurf : BoolProperty(
            name="Subdivision Surface",
            description="Subdivision surface option",
            default=False,
        update=update_obj
            )

    subsurf_type : EnumProperty(
            name = "Subdivision Algorithm",
            description = "Subdivision Algorithm",
            default="CATMULL_CLARK",
            items = [
                ("CATMULL_CLARK", "Catmull-Clark", "Amount is width of new face"),
                ("SIMPLE" , "Simple" , "Amount is offset of new edges from original")
                
            ],
        update=update_obj
        )

    is_adaptive : BoolProperty(
            name="Adaptive Subdivision (Experimental)",
            description="Enable Adaptive Subdivision in CYCLES mode",
            default=False,
        update=update_obj
            )

    adaptive_dicing : FloatProperty(
            name="Dicing Scale",
            description="Multiplier for scene dicing rate (located in the Subdivision panel)",
            default=1.0,
            min=0,
        update=update_obj
            )

    subsurf_subdivisions : IntProperty(
            name="SubSurf Segments",
            description="Number of Subdivision Segments",
            default=2,
            min=1,
        update=update_obj
            )

    subdivide_edges : IntProperty(
            name="Subdivide Edges",
            description="Number of segments to subdivide the mesh by",
            default=0,
            min=0,
            max=10,
        update=update_obj
            )

    shade_smooth  : BoolProperty(
            name="Shade Smooth",
            description="Shade smooth the faces",
            default=False,
        update=update_obj
            )

    auto_smooth  : BoolProperty(
            name="Auto Smooth",
            description="Auto smooth the faces",
            default=False,
        update=update_obj
            )


    flip_normals : BoolProperty(
            name="Flip Normals",
            description="Flip all face normals",
            default=False,
        update=update_obj
            )

    prevent_ovelapping_faces  : BoolProperty(
            name="Prevent face overlaps",
            description="Attempt to stop faces from overlapping one another",
            default=True,
        update=update_obj
            )

    overlap_check_limit : IntProperty(
            name="Face check limit",
            description="Limit the number of checks the tool will do on other faces. Zero will mean the check will never stop.",
            default=0,
            min=0,
        update=update_obj
            )

    layers : BoolVectorProperty(
            name="Layers",
            description="Object Layers",
            size=20,
            options={'HIDDEN', 'SKIP_SAVE'},
        update=update_obj
            )

    # generic transform props
    align_items = (
            ('WORLD', "World", "Align the new object to the world"),
            ('VIEW', "View", "Align the new object to the view"),
            ('CURSOR', "3D Cursor", "Use the 3D cursor orientation for the new object")
    )
    align: EnumProperty(
            name="Align",
            items=align_items,
            default='WORLD',
            update=AddObjectHelper.align_update_callback,
            )

    location : FloatVectorProperty(
            name="Location",
            subtype='TRANSLATION',
        update=update_obj
            )
    rotation : FloatVectorProperty(
            name="Rotation",
            subtype='EULER',
        update=update_obj
            )
    scale : FloatVectorProperty(
            name="Scale",
            subtype='XYZ',
            default=[1,1,1],
            step=1,
        update=update_obj
            )

    #randomisatiion
    random_transform_seed : IntProperty(
        name="Random Transform Seed",
        description="Seed for generating random transformations",
        default=12345,
        step=1,
        update=update_obj
        )
    number_to_create : IntProperty(
        name="Number to create",
        description="Number of shapes to create",
        default=1,
        min=1,
        update=update_obj
        )
    randomize_location  : BoolProperty(
            name="Randomize Location",
            description="Randomise Location",
            default=False,
        update=update_obj
            )
    randomize_rotation  : BoolProperty(
            name="Randomize Rotation",
            description="Randomise Rotation",
            default=False,
        update=update_obj
            )
    randomize_scale  : BoolProperty(
            name="Randomize Scale",
            description="Randomise Scale",
            default=False,
        update=update_obj
            )

    start_rand_location : FloatVectorProperty(
            name="Start Location",
            subtype='TRANSLATION',
        update=update_obj
            )
    start_rand_rotation : FloatVectorProperty(
            name="Start Rotation",
            subtype='EULER',
        update=update_obj
            )
    start_rand_scale : FloatVectorProperty(
            name="Start Scale",
            subtype='XYZ',
            default=[1,1,1],
            step=1,
        update=update_obj
            )

    end_rand_location : FloatVectorProperty(
            name="End Location",
            subtype='TRANSLATION',
        update=update_obj
            )
    end_rand_rotation : FloatVectorProperty(
            name="End Rotation",
            subtype='EULER',
        update=update_obj
            )
    end_rand_scale : FloatVectorProperty(
            name="End Scale",
            subtype='XYZ',
            default=[1,1,1],
            step=1,
        update=update_obj
            )

    uv_projection_limit : FloatProperty(
            name="Angle Limit",
            description="For mapping UVs. Lower for more projection groups, higher for less distortion.",
            default=66.0,
            min=1,
            max=89,
        update=update_obj
            )

    uv_island_margin : FloatProperty(
            name="Island Margin",
            description="Margin to reduce bleed from adjacent islands.",
            default=0.0,
            min=0,
            max=1,
        update=update_obj
            )
    uv_area_weight : FloatProperty(
            name="Area Weight",
            description="Weight projections vector by faces with bigger areas.",
            default=0.0,
            min=0,
            max=1,
        update=update_obj
            )

    uv_stretch_to_bounds : BoolProperty(
            name="Stretch to UV Bounds",
            description="Stretch the final output to texture bounds.",
            default=True,
        update=update_obj
            )

    material_to_use : StringProperty(
            name="Material to use",
            description="Material to add to shape.",
        update=update_obj
    )

    is_boolean : BoolProperty(
        default=False,
        description="Apply a Boolean Operation to the randomly created objects",
        update=update_obj
    )

    def check_bool_index(self, context):
        if self.boolean_main_obj_index >= self.number_to_create:
            self.boolean_main_obj_index = 0
        update_obj(self, context)

    is_parent_booleans : BoolProperty(
        default=True,
        description="Parent the boolean objects to the main object",
        update=update_obj
    )

    boolean_main_obj_index : IntProperty(
        default=0,
        min=0,
        update=check_bool_index,
        description="Index for main object in boolean operation"
    )

    main_obj_bool_operation : EnumProperty(
        name = "Operation",
        description = "Boolean Operation to Perform",
        default="DIFFERENCE",
        items = bool_items,
        update=update_obj
    )

    bool_solver : EnumProperty(
        name = "Solver",
        description = "Method for calculating booleans",
        default="EXACT",
        items = [
            ("FAST" , "Fast" , "Simple solver for the best performance, without support for overlapping geometry"),
            ("EXACT", "Exact", "Advanced solver for the best result")
        ],
        update=update_obj
    )

    fast_overlap_threshold : FloatProperty(
        default=0.000001, 
        step=6, 
        name="Overlap Threshold", 
        description="Threshold for checking overlapping geometry",
        update=update_obj)

    exact_self : BoolProperty(
        default=False, 
        name="Self", 
        description="Allow self-intersection in operands",
        update=update_obj)


    bool_hide_viewport : BoolProperty(
        default=False, 
        description="Hide boolean object in viewport",
        update=update_obj
    )

    bool_hide_render : BoolProperty(
        default=True, 
        description="Hide boolean object from rendering",
        update=update_obj
    )

    bool_display_type : EnumProperty(
        name = "Display Type",
        description = "Display type for Boolean objects",
        default="WIRE",
        items = [
            ("TEXTURED" , "Textured" , "Display the object with textures (if textures are displayed in the viewport)"),
            ("SOLID", "Solid", "Display the object as solid (if solid drawing is enabled in the viewport)"),
            ("WIRE" , "Wire" , "Display the object as a wireframe"),
            ("BOUNDS", "Bounds", "Display the bounds of the object")
        ],
        update=update_obj
    )

    # 
    # Big/Medium/Small
    # ##################

    # seeds

    big_random_seed : IntProperty(
        name="Big Random Seed",
        description="Seed for generating big shapes",
        default=0,
        step=1,
        min=0,
        update=update_obj
        )

    medium_random_seed : IntProperty(
        name="Medium Random Seed",
        description="Seed for generating medium shapes",
        default=1,
        step=1,
        min=0,
        update=update_obj
        ) 

    small_random_seed : IntProperty(
        name="Small Random Seed",
        description="Seed for generating small shapes",
        default=2,
        step=1,
        min=0,
        update=update_obj
        )

    #scattering

    big_random_scatter_seed : IntProperty(
        name="Big Scatter Seed",
        description="Seed for randomly scattering big shapes across the object surface",
        default=0,
        step=1,
        min=0,
        update=update_obj
        ) 

    medium_random_scatter_seed : IntProperty(
        name="Medium Scatter Seed",
        description="Seed for randomly scattering medium shapes across the object surface",
        default=0,
        step=1,
        min=0,
        update=update_obj
        ) 

    small_random_scatter_seed : IntProperty(
        name="Small Scatter Seed",
        description="Seed for randomly scattering small shapes across the object surface",
        default=0,
        step=1,
        min=0,
        update=update_obj
        )


    #shape numbers
    big_shape_num : IntProperty(
        name="Big Shapes Number",
        description="Number of Big Shapes",
        default=1,
        step=1,
        min=1,
        update=update_obj
        ) 


    medium_shape_num : IntProperty(
        name="Medium Shapes Number",
        description="Number of Medium Shapes",
        default=0,
        step=1,
        min=0,
        update=update_obj
        ) 

    small_shape_num : IntProperty(
        name="Small Shapes Number",
        description="Number of Small Shapes",
        default=0,
        step=1,
        min = 0,
        update=update_obj
        )
    
    #shape numbers

    big_shape_scale : FloatProperty(
            name="Big Shape Scale",
            description="Scale of big shape.",
            default=1.0,
            min=0,
            update=update_obj
            )

    medium_shape_scale : FloatProperty(
            name="Medium Shape Scale",
            description="Scale of medium shape.",
            default=0.5,
            min=0,
            update=update_obj
            )

    small_shape_scale : FloatProperty(
            name="Small Shape Scale",
            description="Scale of small shape.",
            default=0.25,
            min=0,
            update=update_obj
            )

    # Booleans
    use_medium_shape_bool : BoolProperty(name="Medium Shapes: Apply Boolean", 
                                description="Apply a boolean operation to the medium shapes",
                                default=False,
                                update=update_obj)

    medium_shape_bool_operation : EnumProperty(
        name = "Operation",
        description = "Boolean Operation to Perform for Medium Shapes",
        default="DIFFERENCE",
        items = bool_items,
        update=update_obj
    )

    use_small_shape_bool : BoolProperty(name="Small Shapes: Apply Boolean", 
                                description="Apply a boolean operation to the small shapes",
                                default=False,
                                update=update_obj)

    small_shape_bool_operation : EnumProperty(
        name = "Operation",
        description = "Boolean Operation to Perform for Small Shapes",
        default="DIFFERENCE",
        items = bool_items,
        update=update_obj
    )


    #colors
    use_colors : BoolProperty(name="Use Coloring", 
                                description="Apply separate coloring to shapes",
                                default=False,
                                update=update_obj)

    big_shape_color : FloatVectorProperty(name="Big Shape Color", 
                                        subtype='COLOR', 
                                        default=[0.0,0.0,1.0, 1.0],
                                        size=4,
                                        min=0,
                                        max=1,
                                        update=update_obj)

    medium_shape_color : FloatVectorProperty(name="Medium Shape Color", 
                                        subtype='COLOR', 
                                        default=[1.0,1.0,0.0, 1.0],
                                        size=4,
                                        min=0,
                                        max=1,
                                        update=update_obj)

    small_shape_color : FloatVectorProperty(name="Small Shape Color", 
                                        subtype='COLOR', 
                                        default=[1.0,0.0,0.0, 1.0],
                                        size=4,
                                        min=0,
                                        max=1,
                                        update=update_obj)

    #materials
    bms_use_materials : BoolProperty(name="Use Materials", 
                                description="Apply separate materials to shapes",
                                default=False,
            update=update_obj)

    bms_medium_shape_material : StringProperty(
            name="Medium Shape Material",
            description="Material to add to medium shape.",
            update=update_obj
    )

    bms_small_shape_material : StringProperty(
            name="Small Shape Material",
            description="Material to add to small shape.",
            update=update_obj
    )

render_engines = []
def get_render_engines(self, context):
    """Get a list of the available render engines."""
    global render_engines
    render_engines = []
    render_engines.append(("SAME" , "Same as Scene" , ""))
    render_engines.append(("BLENDER_EEVEE" , "Eevee" , ""))
    render_engines.append(("BLENDER_WORKBENCH", "Workbench", ""))
    for render_engine_cls in bpy.types.RenderEngine.__subclasses__():
        render_engines.append((render_engine_cls.bl_idname, render_engine_cls.bl_label, ""))
    return render_engines

class ShapeGeneratorIterator(PropertyGroup):

    file_path: StringProperty(
            name = 'Folder Path',
            description = 'Folder Output Path',
            subtype = 'DIR_PATH',
            default = '/tmp\\')


    start_seed : IntProperty(
            name='Start Random Seed',
            description='Seed Value for generating and placing INSERTs',
            min=0,
            default=0
            )

    end_seed : IntProperty(
            name='End Random Seed',
            description='Seed Value for generating and placing INSERTs',
            min=0,
            default=0
            )

    render_engine : EnumProperty(
        name = "Render Engine",
        description = "Engine to use while rendering",
        default=0,
        items = get_render_engines
    )

class ShapeGeneratorConfigPropertyGroup(PropertyGroup, ShapeGeneratorConfig):
    is_property_group : BoolProperty(default=True)
    pass


classes = [ShapeGeneratorConfigPropertyGroup, ShapeGeneratorIterator]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Collection.shape_generator_properties = PointerProperty(name='Shape Generator', type=ShapeGeneratorConfigPropertyGroup)
    bpy.types.Object.shape_generator_collection = PointerProperty(name='Shape Generator', type=bpy.types.Collection)
    bpy.types.Scene.shape_generator_iterator = PointerProperty(name='Shape Generator Iterator', type=ShapeGeneratorIterator)
    

def unregister():
    del bpy.types.Scene.shape_generator_iterator
    del bpy.types.Object.shape_generator_collection
    del bpy.types.Collection.shape_generator_properties

    for cls in classes:
        bpy.utils.unregister_class(cls)