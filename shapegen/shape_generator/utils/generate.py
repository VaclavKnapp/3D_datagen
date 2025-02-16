import bpy
import bmesh
import numpy as np
import random
import sys
import mathutils
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
import math

from . import uvcalc_smart_project
from ..properties import medium_shape_collection_name, small_shape_collection_name

def point_on_triangle(pt1, pt2, pt3, rng):
    """Calculate random point on the triangle with vertices pt1, pt2 and pt3."""
    s, t, = sorted([rng.uniform(0,1), rng.uniform(0,1)])
    return Vector((s * pt1[0] + (t-s)*pt2[0] + (1-t)*pt3[0],
            s * pt1[1] + (t-s)*pt2[1] + (1-t)*pt3[1],
            s * pt1[2] + (t-s)*pt2[2] + (1-t)*pt3[2]))

def get_random_points_objs(context, objs, num_points, rng):
    random_points = []

    for i in range(0, num_points):
        # randomly pick an object.
        obj_idx = rng.choice(range(0, len(objs)))
        obj = objs[obj_idx]       
        random_points.extend(get_random_points(context, obj, 1, rng))
    return random_points

def get_random_points(context, obj, num_points, rng):

    bm = bmesh.new()
    try:
        if obj.mode == 'EDIT':
            bm = bmesh.from_edit_mesh(obj.data).copy()
        elif obj.mode == 'OBJECT':
            bm.from_mesh(obj.data)

            # bm.from_object(obj, depsgraph=context.evaluated_depsgraph_get(), deform=True, face_normals=True)
        # triangulate to easily get points
        input_faces = []
        input_faces = [f for f in bm.faces]
        result = bmesh.ops.triangulate(bm, faces=input_faces, quad_method='FIXED', ngon_method='EAR_CLIP')

        result_faces = result['faces']

        if len(result_faces) == 0:
            return []
        
        random_points = []
        for i in range(0, num_points):  
            # randomly get a face and then a point on that face.
            random_face_index = rng.choice(range(0, len(result_faces)))
            random_face = result_faces[random_face_index]
            if len(random_face.verts) >= 3:
                verts = random_face.verts
                random_point = point_on_triangle(verts[0].co, verts[1].co, verts[2].co, rng)
                random_points.append(obj.matrix_world @ random_point)

    finally:
        bm.free()
        obj.data.update()

    return random_points

    

shapes_collection_name = 'Generated Shapes'
def generate_shapes(self, context, collection = None):
    seed = self.random_seed + self.big_random_seed

    if collection == None:
        new_collection = bpy.data.collections.new('Generated Shape Collection')
        context.scene.collection.children.link(new_collection)

        # copy to incoming properties onto the collection so we can reference them to change the shape in the future.
        for pointer in dir(new_collection.shape_generator_properties):
            if '__' in pointer or pointer in {'bl_rna', 'rna_type', 'type', 'face_count', 'falloff_curve', 'vertex_indices', 'vertex_indices_set', 'is_property_group'}:
                continue

            self_attr = getattr(self, pointer, None)
            
            if self_attr != None:
                setattr(new_collection.shape_generator_properties, pointer, self_attr)

        for attr, value in self.__dict__.items():
            print(attr, value)
            if hasattr(new_collection, attr):
                setattr(new_collection, value)

    else:
        new_collection = collection

    new_objs = generate_shape_objs(self, context, seed, new_collection)

    if not len(new_objs):
        return []

    # if we also have big/medium/small objects, we might need to add more.
    first_main_obj = new_objs[0]

    random_points = get_random_points_objs(context, new_objs, self.big_shape_num - 1, np.random.RandomState(self.big_random_scatter_seed))
    for i in range(0, self.big_shape_num -1):
        new_new_objs = generate_shape_objs(self, context, seed + i + 1, new_collection)
        new_objs.extend(new_new_objs)
        for new_new_obj in new_new_objs:
            new_new_obj.location = random_points[i]
            # if mirrored, set mirror object to location of parent objects.
            for mod in [m for m in new_new_obj.modifiers if m.type=='MIRROR']:
                mod.mirror_object = first_main_obj

    #update the context to kick off object updates when translation applied
    context.view_layer.update()

    # if needed, color the object.
    if self.use_colors:
        for obj in new_objs:
            obj.color = self.big_shape_color

    if self.big_shape_scale != 1:
        for obj in new_objs:
            obj.scale *= self.big_shape_scale


    if self.medium_shape_num:
        new_objs.extend(
            add_supporting_shapes(self, 
            context,
            new_objs, 
            new_collection, 
            medium_shape_collection_name,
            self.random_seed + self.medium_random_seed, 
            self.medium_shape_num, 
            self.medium_random_scatter_seed, 
            self.medium_shape_scale, 
            self.use_colors, 
            self.medium_shape_color,
            self.bms_use_materials,
            self.bms_medium_shape_material,
            self.use_medium_shape_bool,
            self.medium_shape_bool_operation))
        context.view_layer.objects.active = first_main_obj

    if self.small_shape_num:
        new_objs.extend(
            add_supporting_shapes(self, 
            context,
            new_objs, 
            new_collection, 
            small_shape_collection_name,
            self.random_seed + self.small_random_seed, 
            self.small_shape_num, 
            self.small_random_scatter_seed, 
            self.small_shape_scale, 
            self.use_colors, 
            self.small_shape_color,
            self.bms_use_materials,
            self.bms_small_shape_material,
            self.use_small_shape_bool,
            self.small_shape_bool_operation))
        context.view_layer.objects.active = first_main_obj


    if self.is_subsurf and self.is_adaptive:
        context.scene.cycles.feature_set = 'EXPERIMENTAL'

        for new_obj in new_objs:
            new_obj.cycles.use_adaptive_subdivision = True
            new_obj.cycles.dicing_rate = self.adaptive_dicing
            mod = new_obj.modifiers.new('Subdivision', 'SUBSURF')
            mod.levels = self.subsurf_subdivisions
            mod.subdivision_type = self.subsurf_type
            mod.render_levels = self.subsurf_subdivisions
            mod.show_expanded = False

    # Go through all objects and select them.
    for new_obj in new_objs:
        new_obj.select_set(True)

    for obj in new_objs:
        obj.shape_generator_collection = new_collection

    return new_objs

def add_supporting_shapes(self, 
                        context, new_objs, 
                        new_collection, 
                        sub_collection_name,
                        seed, 
                        shape_num, 
                        scatter_seed, 
                        scale, 
                        use_colors, 
                        shape_color,
                        use_materials,
                        material,
                        apply_boolean,
                        boolean_operation):
    sub_seed = seed

    first_main_obj = new_objs[0]

    sub_collection = bpy.data.collections.new(sub_collection_name)
    new_collection.children.link(sub_collection)

    random_points = get_random_points_objs(context, new_objs, shape_num, np.random.RandomState(scatter_seed))

    new_objs_to_return = []

    for i in range(0, shape_num):
        sub_seed += i
        sub_objs = generate_shape_objs(self, context, sub_seed, sub_collection)

        for obj in sub_objs:
            #scale objects appropriately.
            obj.scale *= scale

            #relocate object.
            obj.location = random_points[i]

            # if mirrored, set mirror object to location of parent objects.
            for mod in [m for m in obj.modifiers if m.type=='MIRROR']:
                mod.mirror_object = first_main_obj

            # if needed, color the object.
            if use_colors:
                obj.color = shape_color

            if use_materials:
                if material is not None:
                    for mat in bpy.data.materials:
                        if mat.name == material:
                            obj.data.materials.clear()
                            obj.data.materials.append(mat)
                            break

        new_objs_to_return.extend(sub_objs)

    if apply_boolean:
        for obj in new_objs:
            apply_booleans(obj, new_objs_to_return, context,
                bool_operation=boolean_operation, 
                bool_solver=self.bool_solver, 
                exact_self=self.exact_self, 
                bool_hide_viewport=self.bool_hide_viewport, 
                bool_hide_render=self.bool_hide_render, 
                bool_display_type=self.bool_display_type, 
                fast_overlap_threshold=self.fast_overlap_threshold, 
                is_parent_booleans=False)

    #update the context to kick off object updates when translation applied
    context.view_layer.update()

    return new_objs_to_return

def generate_shape_objs(self, context, seed, collection):
    #initialise the random seed.
    random.seed(seed)

    random_transformer = np.random.RandomState(self.random_transform_seed)

    new_objs = []

    for shape_no in range(0, self.number_to_create):

        new_obj = generate_shape_obj(self, context, collection, random_transformer)

        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.object.mode_set(mode='OBJECT')

        new_objs.append(new_obj)

    if self.is_boolean and len(new_objs):
        # take the first object (for now) and boolean the others.
        main_obj = new_objs[self.boolean_main_obj_index]
        bool_objs = new_objs[:]
        del bool_objs[self.boolean_main_obj_index]

        apply_booleans(main_obj, bool_objs, context,
            bool_operation=self.main_obj_bool_operation, 
            bool_solver=self.bool_solver, 
            exact_self=self.exact_self, 
            bool_hide_viewport=self.bool_hide_viewport, 
            bool_hide_render=self.bool_hide_render, 
            bool_display_type=self.bool_display_type, 
            fast_overlap_threshold=self.fast_overlap_threshold, 
            is_parent_booleans=self.is_parent_booleans)

        context.view_layer.objects.active = main_obj

    return new_objs

def generate_shape_obj(self, context, new_collection, random_transformer):
    mesh = generate_shape_mesh(min_extrude = self.min_extrude,
                                            max_extrude = self.max_extrude,
                                            amount = self.amount,
                                            random = random,
                                            random_transformer = random_transformer,
                                            min_taper = self.min_taper,
                                            max_taper = self.max_taper,
                                            min_rotation = self.min_rotation,
                                            max_rotation = self.max_rotation,
                                            min_slide = self.min_slide,
                                            max_slide = self.max_slide,
                                            favour_vec = self.favour_vec,
                                            overlap_check_limit = self.overlap_check_limit,
                                            prevent_ovelapping_faces = self.prevent_ovelapping_faces,
                                            location = self.location,
                                            rotation = self.rotation,
                                            scale = self.scale,
                                            randomize_location = self.randomize_location,
                                            randomize_rotation = self.randomize_rotation,
                                            randomize_scale = self.randomize_scale,
                                            start_rand_location = self.start_rand_location,
                                            end_rand_location = self.end_rand_location,
                                            start_rand_rotation = self.start_rand_rotation,
                                            end_rand_rotation = self.end_rand_rotation,
                                            start_rand_scale = self.start_rand_scale,
                                            end_rand_scale = self.end_rand_scale,
                                            subdivide_edges = self.subdivide_edges,
                                            flip_normals = self.flip_normals,
                                            material=self.material_to_use)

    uvcalc_smart_project.smart_project(mesh,
                                        projection_limit = self.uv_projection_limit,
                                        island_margin = self.uv_island_margin,
                                        user_area_weight = self.uv_area_weight,
                                        stretch_to_bounds = self.uv_stretch_to_bounds)

    new_obj = bpy.data.objects.new("Generated Shape", mesh)

    new_collection.objects.link(new_obj)

    for selected_object in context.selected_objects:
        selected_object.select_set(False)
    new_obj.select_set(True)
    context.view_layer.objects.active = new_obj
    layer = context.view_layer
    layer.update()
    new_obj.location = context.scene.cursor.location

    decorate_object(new_obj, self)
    

    return new_obj


def decorate_object(new_obj, self):
    #apply optional modifiers.

    #apply bevel modifier if needed.
    if self.is_bevel:
        mod = new_obj.modifiers.new('Bevel', 'BEVEL')
        mod.width = self.bevel_width
        mod.segments = self.bevel_segments
        mod.show_expanded = False
        mod.profile = self.bevel_profile
        mod.offset_type = self.bevel_method
        mod.use_clamp_overlap = self.bevel_clamp_overlap

    #apply mirror modifier if needed.
    if self.mirror_x or self.mirror_y or self.mirror_z:
        mod = new_obj.modifiers.new('Mirror', 'MIRROR')
        mod.use_axis[0] = self.mirror_x
        mod.use_axis[1] = self.mirror_y
        mod.use_axis[2] = self.mirror_z
        mod.use_bisect_axis[0] = self.mirror_x
        mod.use_bisect_axis[1] = self.mirror_y
        mod.use_bisect_axis[2] = self.mirror_z
        mod.use_clip = True
        mod.show_expanded = False
        mod.show_on_cage = True



    #add a subsurf modifer if needed.
    if self.is_subsurf and not self.is_adaptive:
        mod = new_obj.modifiers.new('Subdivision', 'SUBSURF')
        mod.levels = self.subsurf_subdivisions
        mod.subdivision_type = self.subsurf_type
        mod.render_levels = self.subsurf_subdivisions
        mod.show_expanded = False



    #shade all the faces as smooth if needed.
    if self.shade_smooth:
        bpy.ops.object.shade_smooth()
        new_obj.data.use_auto_smooth = self.auto_smooth

    return new_obj


def generate_shape_mesh(*, min_extrude = 0.5,
                        max_extrude = 1,
                        amount=5,
                        random = random,
                        random_transformer = np.random.RandomState(12345),
                        min_taper=0.5,
                        max_taper=1,
                        min_rotation=0,
                        max_rotation=20,
                        min_slide=0,
                        max_slide=0,
                        favour_vec = Vector((1,1,1)),
                        overlap_check_limit = 0,
                        prevent_ovelapping_faces=False,
                        location=Vector((0,0,0)),
                        rotation=mathutils.Euler((0.0, math.radians(0), 0.0), 'XYZ'),
                        scale=Vector((1,1,1)),
                        randomize_location=False,
                        randomize_rotation=False,
                        randomize_scale=False,
                        start_rand_location=Vector((0,0,0)),
                        end_rand_location=Vector((0,0,0)),
                        start_rand_rotation=mathutils.Euler((0.0, math.radians(0), 0.0), 'XYZ'),
                        end_rand_rotation=mathutils.Euler((0.0, math.radians(0), 0.0), 'XYZ'),
                        start_rand_scale=Vector((1,1,1)),
                        end_rand_scale=Vector((1,1,1)),
                        subdivide_edges=0,
                        flip_normals=False,
                        material=None):



    mesh = bpy.data.meshes.new("Generated Shape")

    #first, add a simple box and scale it by min and max extrusion lengths
    verts_loc, faces = add_box(random.uniform(min_extrude, max_extrude),
                               random.uniform(min_extrude, max_extrude),
                               random.uniform(min_extrude, max_extrude),
                               )
    bm = bmesh.new()

    for v_co in verts_loc:
        bm.verts.new(v_co)
    bm.verts.ensure_lookup_table()
    for f_idx in faces:
        f = bm.faces.new([bm.verts[i] for i in f_idx])
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    #once first created, add a max score for the faces when choosing them.
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.faces.ensure_lookup_table()
    max_score = 0
    for f in bm.faces:
        score = calc_choice_score(favour_vec, f)
        if score > max_score:
            max_score = score
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    #the main part of the algorithm - create  extrusions on faces with different scales and orientations, optionally checking for overlaps.
    for i in range(0, amount):
        is_manifold = False
        count = 0
        count_limit_exceeded = False
        #perform an extrusion, but check of the mesh is still 'manifold', ie no overlapping faces.  If it is not, attempt again.
        while (not is_manifold) and (not count_limit_exceeded):
            count+=1
            #determine if we have exceeded the number of checks to prevent an infinite loop, configured by the user.
            if overlap_check_limit == 0:
                #zero will mean we never exceed a limit, could cause an infinite loop...but worth the risk.
                count_limit_exceeded = False
            else:
                count_limit_exceeded = (count > overlap_check_limit)

            # Get a BMesh representation
            bm = bmesh.new()   # create an empty BMesh
            bm.from_mesh(mesh)   # fill it in from a Mesh

            # determine a score threshold to choose the number of faces from.
            bm.faces.ensure_lookup_table()
            if max_score > 0:
                score_thres = random.uniform(0, max_score)
            else:
                score_thres = 0

            # determine a list of candidate faces to choose from.
            candidate_faces = []
            for f2 in bm.faces:
                if calc_choice_score(favour_vec, f2) > score_thres:
                    candidate_faces.append(f2)

            #either select from the candidate faces, or if there are no candidates, select a random one.
            if len(candidate_faces) > 0:
                f = random.choice(candidate_faces)
            else:
                f = random.choice(bm.faces)

            #perform a rotation on the extruded face depending on the supplied limits.
            orig_face_center = f.calc_center_median()
            orig_orth_vec = f.normal.orthogonal()
            #perform a random rotation of the orthoganol vector to ensure the face is rotated along a random orthoganal axis.
            orig_orth_vec.rotate(mathutils.Matrix.Rotation(math.radians(random.uniform(0,360)), 4, f.normal))

            #perform an inset operation as a basis for the extrusion.
            result = bmesh.ops.inset_individual(bm, faces=[f], thickness=0, depth = random.uniform(min_extrude, max_extrude), use_even_offset=True)

            bm.faces.ensure_lookup_table()
            bm.edges.ensure_lookup_table()
            bm.verts.ensure_lookup_table()

            #randomly scale the face of the extrusion
            scale_fac = random.uniform(min_taper, max_taper)
            scale_face(bm, f, scale_fac, scale_fac, scale_fac)

            #rotate the face only when it has been extruded.
            bmesh.ops.rotate(bm, cent=orig_face_center, matrix=mathutils.Matrix.Rotation(math.radians(random.uniform(min_rotation,max_rotation)), 4, orig_orth_vec), verts=f.verts)

            #slide...
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
            orig_face_center = f.calc_center_median()
            orig_orth_vec = f.normal.orthogonal()
            orig_orth_vec.rotate(mathutils.Matrix.Rotation(math.radians(random.uniform(0,360)), 4, f.normal))


            transform_face(bm, f, orig_orth_vec * random.uniform(min_slide, max_slide))

            #we now need to check the new faces for overlap.
            faces_to_check = result['faces']
            faces_to_check.append(f)


            #check for any overlapping faces
            is_manifold = True
            if prevent_ovelapping_faces:

                #for each newly crated or moved face, check whether the other edges overlap
                #or whether that face's edges overlap with th eother faces.
                for f2 in faces_to_check:
                    is_manifold &= not is_face_overlap_with_edges(f2, bm.edges)
                    if not is_manifold:
                        break
                    for f3 in bm.faces:
                        is_manifold &= not is_face_overlap_with_edges(f3, f2.edges)
                        if not is_manifold:
                            break
                #if not is_manifold:
                #    break

            if is_manifold:
                #determine the next maximum score.
                for f in faces_to_check:
                    score = calc_choice_score(favour_vec, f)
                    if score > max_score:
                        max_score = score

                #commit the mesh
                bm.to_mesh(mesh)
                bm.free()
                mesh.update()
            else:
                bm.free()

    #
    bm = bmesh.new()   # create an empty BMesh
    bm.from_mesh(mesh)   # fill it in from a Mesh


    #apply random transforms if necessary

    location2 = location.copy()
    random_location2 = start_rand_location.copy()
    random_location2[0] = random_transformer.uniform(start_rand_location[0], end_rand_location[0])
    random_location2[1] = random_transformer.uniform(start_rand_location[1], end_rand_location[1])
    random_location2[2] = random_transformer.uniform(start_rand_location[2], end_rand_location[2])
    if randomize_location:
        location2 = random_location2

    rotation2 = rotation.copy()
    random_rotation2 = start_rand_rotation.copy()
    random_rotation2[0] = random_transformer.uniform(start_rand_rotation[0], end_rand_rotation[0])
    random_rotation2[1] = random_transformer.uniform(start_rand_rotation[1], end_rand_rotation[1])
    random_rotation2[2] = random_transformer.uniform(start_rand_rotation[2], end_rand_rotation[2])
    if randomize_rotation:
        rotation2 = random_rotation2

    scale2 = scale.copy()
    random_scale2 = start_rand_scale.copy()
    random_scale2[0] = random_transformer.uniform(start_rand_scale[0], end_rand_scale[0])
    random_scale2[1] = random_transformer.uniform(start_rand_scale[1], end_rand_scale[1])
    random_scale2[2] = random_transformer.uniform(start_rand_scale[2], end_rand_scale[2])
    if randomize_scale:
        scale2 = random_scale2



    #transform the overall mesh.
    bmesh.ops.scale(bm, vec=scale2, verts=bm.verts)
    bmesh.ops.translate(bm,vec=location2, verts=bm.verts)
    bmesh.ops.rotate(bm, cent=Vector((0,0,0)), matrix=rotation2.to_matrix(), verts=bm.verts)

    bmesh.ops.subdivide_edges(bm, edges=bm.edges, cuts=subdivide_edges, use_grid_fill=True)



    #flip the normals if necessary.
    if flip_normals:
        bmesh.ops.reverse_faces(bm, faces=bm.faces)


    bm.to_mesh(mesh)
    bm.free()
    
    if material is not None:
        for mat in bpy.data.materials:
            if mat.name == material:
                mesh.materials.append(mat)
                break
    
    mesh.update()


    return mesh

x_dir = Vector((1,0,0))
y_dir = Vector((0,1,0))
z_dir = Vector((0,0,1))

def calc_choice_score(favour_vec,f):
    """calculate a score for how strongly a face should be chosen."""
    # do not bother if they are all equal.
    if favour_vec[0] == favour_vec[1] == favour_vec[2]:
        return 0

    #weight the score based on the direction of the face.
    x_dot = abs(x_dir.dot(f.normal))
    y_dot = abs(y_dir.dot(f.normal))
    z_dot = abs(z_dir.dot(f.normal))

    return x_dot * favour_vec[0] + y_dot * favour_vec[1] + z_dot * favour_vec[2]

def is_face_overlap_with_edges(f, edges):
    """determine if face f is overlapped by the edges"""
    center = f.calc_center_median()

    #check whether any of the edges overlap with the face.
    for e in edges:

        #do a preliminary check to see if the edge goes across the face plane
        pt1 = True if (e.verts[0].co - center).dot(f.normal) < 0 else False
        pt2 = True if (e.verts[1].co - center).dot(f.normal) < 0 else False

        #if one is negative and the other positive, we potentially have an overlap.
        if (pt1 ^ pt2):
            #get the coordinates of the intersection point between the face plane and the edge.
            intersect_line_plane_co = mathutils.geometry.intersect_line_plane(e.verts[0].co, e.verts[1].co, center, f.normal, False)
            if intersect_line_plane_co is not None:
                #now check if that point is inside the face.
                is_in_face = bmesh.geometry.intersect_face_point(f, intersect_line_plane_co)
                #if it is, check the edge wasn't actually connected to the face.  If it isn't,
                #return True, otherwise keep on searching.
                if is_in_face:
                    connected_edge = False
                    for v in f.verts:
                        if v.index == e.verts[0].index or v.index == e.verts[1].index:
                            connected_edge = True
                            break
                    if connected_edge:
                        #ignore any connected edges and keep searching.
                        continue
                    else:
                        #we have found an intersection - return true immediately.
                        return True
    return False

# Scales a face in local face space.
def scale_face(bm, face, scale_x, scale_y, scale_z):
    face_space = get_face_matrix(face)
    face_space.invert()
    bmesh.ops.scale(bm,
                    vec=Vector((scale_x, scale_y, scale_z)),
                    space=face_space,
                    verts=face.verts)

# Scales a face in local face space.
def transform_face(bm, face, transform_vec):
    bmesh.ops.transform(bm,
                    matrix=mathutils.Matrix.Translation(transform_vec),
                    verts=face.verts)

# Returns a rough 4x4 transform matrix for a face (doesn't handle
# distortion/shear) with optional position override.
def get_face_matrix(face, pos=None):
    x_axis = (face.verts[1].co - face.verts[0].co).normalized()
    z_axis = -face.normal
    y_axis = z_axis.cross(x_axis)
    if not pos:
        pos = face.calc_center_bounds()

    # Construct a 4x4 matrix from axes + position:
    # http://i.stack.imgur.com/3TnQP.png
    mat = Matrix()
    mat[0][0] = x_axis.x
    mat[1][0] = x_axis.y
    mat[2][0] = x_axis.z
    mat[3][0] = 0
    mat[0][1] = y_axis.x
    mat[1][1] = y_axis.y
    mat[2][1] = y_axis.z
    mat[3][1] = 0
    mat[0][2] = z_axis.x
    mat[1][2] = z_axis.y
    mat[2][2] = z_axis.z
    mat[3][2] = 0
    mat[0][3] = pos.x
    mat[1][3] = pos.y
    mat[2][3] = pos.z
    mat[3][3] = 1
    return mat

def add_box(width, height, depth):
    """
    This function takes inputs and returns vertex and face arrays.
    no actual mesh data creation is done here.
    """
    verts = [(+1.0, +1.0, -1.0),
             (+1.0, -1.0, -1.0),
             (-1.0, -1.0, -1.0),
             (-1.0, +1.0, -1.0),
             (+1.0, +1.0, +1.0),
             (+1.0, -1.0, +1.0),
             (-1.0, -1.0, +1.0),
             (-1.0, +1.0, +1.0),
             ]

    verts = [(+.5, +.5, -.5),
             (+.5, -.5, -.5),
             (-.5, -.5, -.5),
             (-.5, +.5, -.5),
             (+.5, +.5, +.5),
             (+.5, -.5, +.5),
             (-.5, -.5, +.5),
             (-.5, +.5, +.5),
             ]

    faces = [(0, 1, 2, 3),
             (4, 7, 6, 5),
             (0, 4, 5, 1),
             (1, 5, 6, 2),
             (2, 6, 7, 3),
             (4, 0, 3, 7),
            ]

    # apply size
    for i, v in enumerate(verts):
        verts[i] = v[0] * width, v[1] * depth, v[2] * height

    return verts, faces


def apply_booleans(main_obj, bool_objs, context,
                    bool_operation="DIFFERENCE", 
                    bool_solver="EXACT", 
                    exact_self=False, 
                    bool_hide_viewport=False, 
                    bool_hide_render=True, 
                    bool_display_type='WIRE', 
                    fast_overlap_threshold=0.000001, 
                    is_parent_booleans=True):
    try:
        i = 0
        for bool_obj in bool_objs:
            bool_mod = main_obj.modifiers.new(type="BOOLEAN", name="Shape Gen Boolean " + str(i))
            bool_mod.object = bool_obj
            bool_mod.operation = bool_operation

            if bpy.app.version > (2, 90, 0):
                bool_mod.solver = bool_solver
                if bool_solver == "FAST":
                    bool_mod.double_threshold = fast_overlap_threshold
                elif bool_solver == "EXACT":
                    bool_mod.use_self = exact_self

            bool_obj.hide_viewport = bool_hide_viewport
            bool_obj.hide_render = bool_hide_render
            bool_obj.display_type = bool_display_type

            if is_parent_booleans:
                bool_obj.parent = main_obj
            i+=1
    except IndexError:
        pass
