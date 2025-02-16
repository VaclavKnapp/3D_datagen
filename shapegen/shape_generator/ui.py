import bpy

def draw_properties(layout, self):
    draw_warnings(layout, self)


    col = layout.column()

    box = col.box()
    row = box.row()
    row.prop(self, "show_seed_panel",
        icon="TRIA_DOWN" if self.show_seed_panel else "TRIA_RIGHT",
        icon_only=True, emboss=False
    )
    row.label(text="Master Seed")

    if self.show_seed_panel:
        seed_panel = box.column()
        seed_panel.prop(self, "random_seed")

    box = col.box()
    row = box.row()
    row.prop(self, "show_extrude_panel",
        icon="TRIA_DOWN" if self.show_extrude_panel else "TRIA_RIGHT",
        icon_only=True, emboss=False
    )
    row.label(text="Extrusions")

    if self.show_extrude_panel:
        extrude_panel = box.column()
        extrude_panel.prop(self, "amount", text="Amount")
        extrude_panel.label(text="Extrude Length")
        extrude_panel.prop(self, "min_extrude", text="Min")
        extrude_panel.prop(self, "max_extrude", text="Max")
        extrude_panel.label(text="Taper")
        extrude_panel.prop(self, "min_taper", text="Min")
        extrude_panel.prop(self, "max_taper", text="Max")
        extrude_panel.label(text="Rotation")
        extrude_panel.prop(self, "min_rotation", text="Min")
        extrude_panel.prop(self, "max_rotation", text="Max")
        extrude_panel.label(text="Slide")
        extrude_panel.prop(self, "min_slide", text="Min")
        extrude_panel.prop(self, "max_slide", text="Max")
        extrude_panel.prop(self, "favour_vec", text="When choosing a face, favour")
        
    box = col.box()
    row = box.row()
    row.prop(self, "show_bevel_panel",
        icon="TRIA_DOWN" if self.show_bevel_panel else "TRIA_RIGHT",
        icon_only=True, emboss=False
    )
    row.label(text="Bevel")
    if self.show_bevel_panel:
        bevel_panel = box.column()
        bevel_panel.prop(self, "is_bevel")
        bevel_panel_col = bevel_panel.column()
        bevel_panel_col.enabled = self.is_bevel
        bevel_panel_col.prop(self, "bevel_width")
        bevel_panel_col.prop(self, "bevel_segments")
        bevel_panel_col.prop(self, "bevel_profile")
        bevel_panel_col.label(text="Width Method:")
        bevel_panel_col.row().prop(self, "bevel_method", expand=True)
        bevel_panel_col.separator()
        bevel_panel_col.prop(self, "bevel_clamp_overlap")

    box = col.box()
    row = box.row()
    row.prop(self, "show_subdivide_panel",
        icon="TRIA_DOWN" if self.show_subdivide_panel else "TRIA_RIGHT",
        icon_only=True, emboss=False
    )
    row.label(text="Subdivision")

    if self.show_subdivide_panel:

        subdivide_panel = box.column()
        subdivide_panel.prop(self, "is_subsurf")
        subdivide_panel_col = box.column()
        subdivide_panel_col.enabled = self.is_subsurf
        subdivide_panel_row = subdivide_panel_col.row(align=True)
        subdivide_panel_row.prop(self, "subsurf_type", expand=True)
        subdivide_panel_col.prop(self, "is_adaptive")
        if self.is_adaptive:
            subdivide_panel_col.prop(self, "adaptive_dicing")
        subdivide_panel_col.prop(self, "subsurf_subdivisions")
        subdivide_panel_edges_col = box.column()
        subdivide_panel_edges_col.separator()
        subdivide_panel_edges_col.prop(self, "subdivide_edges")


    box = col.box()
    row = box.row()
    row.prop(self, "show_mirror_panel",
        icon="TRIA_DOWN" if self.show_mirror_panel else "TRIA_RIGHT",
        icon_only=True, emboss=False
    )
    row.label(text="Mirror")

    if self.show_mirror_panel:
        mirror_panel = box.column()
        mirror_panel.prop(self, "mirror_x")
        mirror_panel.prop(self, "mirror_y")
        mirror_panel.prop(self, "mirror_z")



    box = col.box()
    row = box.row()
    row.prop(self, "show_material_panel",
        icon="TRIA_DOWN" if self.show_material_panel else "TRIA_RIGHT",
        icon_only=True, emboss=False
    )
    row.label(text="Materials")

    if self.show_material_panel:
        material_panel = box.column()
        material_panel.prop_search(self, "material_to_use", bpy.data, "materials")                



    box = col.box()
    row = box.row()
    row.prop(self, "show_overlap_faces_panel",
        icon="TRIA_DOWN" if self.show_overlap_faces_panel else "TRIA_RIGHT",
        icon_only=True, emboss=False
    )
    row.label(text="Overlapping Faces")

    if self.show_overlap_faces_panel:
        overlap_faces_panel = box.column()
        overlap_faces_panel.prop(self, "prevent_ovelapping_faces")
        check_limit_col = overlap_faces_panel.column()
        check_limit_col.enabled = self.prevent_ovelapping_faces
        check_limit_col.prop(self, "overlap_check_limit")



    box = col.box()
    row = box.row()
    row.prop(self, "show_translation_panel",
        icon="TRIA_DOWN" if self.show_translation_panel else "TRIA_RIGHT",
        icon_only=True, emboss=False
    )
    row.label(text="Transformation")

    if self.show_translation_panel:
        translation_panel = box.column()

        location_panel = translation_panel.column()
        location_panel.enabled = not self.randomize_location
        location_panel.prop(self, "location")

        rotation_panel = translation_panel.column()
        rotation_panel.enabled = not self.randomize_rotation
        rotation_panel.prop(self, "rotation")

        scale_panel = translation_panel.column()
        scale_panel.enabled = not self.randomize_scale
        scale_panel.prop(self, "scale")


    box = col.box()
    row = box.row()
    row.prop(self, "show_randomisation_panel",
        icon="TRIA_DOWN" if self.show_randomisation_panel else "TRIA_RIGHT",
        icon_only=True, emboss=False
    )
    row.label(text="Random Transform")

    if self.show_randomisation_panel:
        randomisation_panel = box.column()
        randomisation_panel.prop(self, "number_to_create")
        randomisation_panel.prop(self, "random_transform_seed")
        randomisation_panel.separator()
        randomisation_panel.prop(self, "randomize_location")
        end_rand_location_panel = randomisation_panel.column()
        end_rand_location_panel.enabled = self.randomize_location
        end_rand_location_panel.prop(self, "start_rand_location", text="Start Location")
        end_rand_location_panel.prop(self, "end_rand_location", text="End Location")
        end_rand_location_panel.separator()
        randomisation_panel.prop(self, "randomize_rotation")
        end_rand_rotation_panel = randomisation_panel.column()
        end_rand_rotation_panel.enabled = self.randomize_rotation
        end_rand_rotation_panel.prop(self, "start_rand_rotation", text="Start Rotation")
        end_rand_rotation_panel.prop(self, "end_rand_rotation", text="End Rotation")
        end_rand_rotation_panel.separator()
        randomisation_panel.prop(self, "randomize_scale")
        end_rand_scale_panel = randomisation_panel.column()
        end_rand_scale_panel.enabled = self.randomize_scale
        end_rand_scale_panel.prop(self, "start_rand_scale", text="Start Scale")
        end_rand_scale_panel.prop(self, "end_rand_scale", text="End Scale")
        end_rand_scale_panel.separator()
        randomisation_panel.prop(self, "is_boolean", text="Boolean Operation")
        boolean_panel = randomisation_panel.column()
        boolean_panel.enabled = self.is_boolean
        boolean_panel.prop(self, "boolean_main_obj_index", text="Main Object Index")
        boolean_panel.prop(self, "is_parent_booleans", text="Parent Booleans")
        boolean_panel_row = boolean_panel.row(align=True)
        boolean_panel_row.prop(self, "main_obj_bool_operation", text="Operation", expand=True)



    box = col.box()
    row = box.row()
    row.prop(self, "show_sizing_panel",
        icon="TRIA_DOWN" if self.show_sizing_panel else "TRIA_RIGHT",
        icon_only=True, emboss=False
    )
    row.label(text="Big/Medium/Small")

    if self.show_sizing_panel:
        sizing_panel = box.column()

        #shape numbers
        sizing_panel.prop(self, "big_shape_num")
        sizing_panel.prop(self, "medium_shape_num")
        sizing_panel.prop(self, "small_shape_num")
        sizing_panel.separator()


        # shape seeds
        sizing_panel.prop(self, "big_random_seed")
        sizing_panel.prop(self, "medium_random_seed")
        sizing_panel.prop(self, "small_random_seed")
        sizing_panel.separator()

        #scattering
        sizing_panel.prop(self, "big_random_scatter_seed")
        sizing_panel.prop(self, "medium_random_scatter_seed")
        sizing_panel.prop(self, "small_random_scatter_seed")
        sizing_panel.separator()
        
        #shape numbers
        sizing_panel.prop(self, "big_shape_scale")
        sizing_panel.prop(self, "medium_shape_scale")
        sizing_panel.prop(self, "small_shape_scale")
        sizing_panel.separator()

        # Booleans
        sizing_panel.prop(self, "use_medium_shape_bool")
        boolean_panel_row = sizing_panel.row(align=True)
        boolean_panel_row.enabled = self.use_medium_shape_bool
        boolean_panel_row.prop(self, "medium_shape_bool_operation", text="Operation", expand=True)

        sizing_panel.prop(self, "use_small_shape_bool")
        boolean_panel_row = sizing_panel.row(align=True)
        boolean_panel_row.enabled = self.use_small_shape_bool
        boolean_panel_row.prop(self, "small_shape_bool_operation", text="Operation", expand=True)

        
        #colors
        sizing_panel.prop(self, "use_colors")
        sizing_color_col = sizing_panel.column()
        sizing_color_col.enabled = self.use_colors
        sizing_color_col.prop(self, "big_shape_color")
        sizing_color_col.prop(self, "medium_shape_color")
        sizing_color_col.prop(self, "small_shape_color")
        sizing_panel.separator()

        #materials
        sizing_panel.prop(self, "bms_use_materials")
        sizing_color_col = sizing_panel.column()
        sizing_color_col.enabled = self.bms_use_materials
        sizing_color_col.prop_search(self, "material_to_use", bpy.data, "materials", text="Big")
        sizing_color_col.prop_search(self, "bms_medium_shape_material", bpy.data, "materials", text="Medium")
        sizing_color_col.prop_search(self, "bms_small_shape_material", bpy.data, "materials", text="Small")
        sizing_panel.separator()

    

    box = col.box()
    row = box.row()
    row.prop(self, "show_bool_panel",
        icon="TRIA_DOWN" if self.show_bool_panel else "TRIA_RIGHT",
        icon_only=True, emboss=False
    )
    row.label(text="Boolean Operation")

    if self.show_bool_panel:
        boolean_panel = box.column()            
        if bpy.app.version > (2, 90, 0):
            boolean_panel_row = boolean_panel.row(align=True)
            boolean_panel_row.prop(self, "bool_solver", text="Solver", expand=True)
            if self.bool_solver == "FAST":
                boolean_panel.prop(self, "fast_overlap_threshold")
            elif self.bool_solver == "EXACT":
                boolean_panel.prop(self, "exact_self")
        
        boolean_panel.separator()
        boolean_panel.prop(self, "bool_hide_viewport", text="Hide in Viewport")
        boolean_panel.prop(self, "bool_hide_render", text="Hide in Render")
        boolean_panel.prop(self, "bool_display_type", text="Display Type")


    box = col.box()
    row = box.row()
    row.prop(self, "show_uv_projection_panel",
        icon="TRIA_DOWN" if self.show_uv_projection_panel else "TRIA_RIGHT",
        icon_only=True, emboss=False
    )
    row.label(text="UV Smart Projection")

    if self.show_uv_projection_panel:
        other_options_panel = box.column()
        other_options_panel.prop(self, "uv_projection_limit")
        other_options_panel.prop(self, "uv_island_margin")
        other_options_panel.prop(self, "uv_area_weight")
        other_options_panel.prop(self, "uv_stretch_to_bounds")


    box = col.box()
    row = box.row()
    row.prop(self, "show_other_options_panel",
        icon="TRIA_DOWN" if self.show_other_options_panel else "TRIA_RIGHT",
        icon_only=True, emboss=False
    )
    row.label(text="Other Options")

    if self.show_other_options_panel:
        other_options_panel = box.column()
        other_options_panel.prop(self, "shade_smooth")
        if self.shade_smooth:
            other_options_panel.prop(self, "auto_smooth")
        other_options_panel.prop(self, "flip_normals")
        other_options_panel.prop_menu_enum(self, "align")


def is_high_computation(self):
    return self.amount > 20 or \
            self.number_to_create > 20 or \
            self.medium_shape_num > 10 or \
            self.small_shape_num > 10 or \
            self.subsurf_subdivisions > 3 or \
            self.subdivide_edges > 5


def draw_warnings(layout, self):
    if is_high_computation(self):
        col = layout.column()
        col.alert = True
        col.label(text="Warning: complex set-up detected.")