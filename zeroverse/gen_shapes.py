import sys
import numpy as np
import pickle
from pathlib import Path
import argparse
import sys
import cv2
import os
from glob import glob
import random
import shutil
import time
import uuid
import json
from datasets import load_dataset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from common import *
import bpy
import random

def find_texture_images(texture_folder):
    """Find all texture images in the specified folder"""
    if not texture_folder:
        return None
    
    import os
    import glob
    
    # Valid texture file extensions
    extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']
    
    # Find all files with valid extensions
    texture_images = []
    for ext in extensions:
        texture_images.extend(glob.glob(os.path.join(texture_folder, f'*{ext}')))
        texture_images.extend(glob.glob(os.path.join(texture_folder, f'*{ext.upper()}')))
    
    # Sort for deterministic results (with seed)
    texture_images.sort()
    
    if not texture_images:
        print(f"No texture images found in {texture_folder}")
        return None
    
    print(f"Found {len(texture_images)} texture images in {texture_folder}")
    return texture_images


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    print(f'Seed: {seed}')


class Shape(object):
    def __init__(self):
        self.points = []
        self.uvs = []
        self.faces = []
        self.facesUV = []
        self.matNames = []
        self.matStartId = []

    def genShape(self):
        self.points = np.reshape([], (-1,3)).astype(float)
        self.uvs = np.reshape([], (-1,2)).astype(float)
        self.faces = np.reshape([], (-1,3)).astype(int)
        self.facesUV = np.reshape([], (-1,3)).astype(int)
        self.matNames = []
        self.matStartId = []

    def permuteMatIds(self, ratio = 0.25):
        if len(self.matStartId) == 0:
            print("no mats")
            return
        newIds = [self.matStartId[0]]

        for i in range(1, len(self.matStartId)):
            neg = self.matStartId[i] - self.matStartId[i-1]
            negCount = -int(ratio * neg)

            if i != len(self.matStartId) - 1:
                pos = self.matStartId[i+1] - self.matStartId[i]
            else:
                pos = len(self.faces) - self.matStartId[i]

            posCount = int(ratio * pos)

            offset = np.random.permutation(posCount - negCount)[0] + negCount

            newIds.append(self.matStartId[i] + offset)
        self.matStartId = newIds

    def computeNormals(self):
        vec0 = self.points[self.faces[:,1]-1] - self.points[self.faces[:,0]-1]
        vec1 = self.points[self.faces[:,2]-1] - self.points[self.faces[:,1]-1]
        areaNormals = np.cross(vec0, vec1)
        self.normals = self.points.copy()
        vertFNs = np.zeros(len(self.points), int)
        vertFMaps = np.zeros((len(self.points), 200), int)
        for iF, face in enumerate(self.faces):
            for id in face:
                vertFMaps[id-1, vertFNs[id-1]] = iF
                vertFNs[id-1] += 1

        for i in range(len(self.points)):
            faceNormals = areaNormals[vertFMaps[i,:vertFNs[i]]]
            normal = np.average(faceNormals, axis=0)
            self.normals[i] = normalize(normal).reshape(-1)
        return self.normals

    def loadSimpleObj(self, filePath):

        self.points = []
        self.uvs = []
        self.faces = []
        self.facesUV = []
        self.matNames = []
        self.matStartId = []

        with open(filePath, "r") as f:
            #write v
            curFid = 0
            while True:
                lineStr = f.readline()
                if lineStr == "":
                    break
                if lineStr[:2] == "v ":
                    point = [float(val) for val in lineStr[2:-1].split(" ")]
                    self.points.append(point)
                if lineStr[:2] == "vt":
                    point = [float(val) for val in lineStr[3:-1].split(" ")]
                    self.uvs.append(point)
                if lineStr[:len("usemtl")] == "usemtl":
                    self.matStartId.append(curFid)
                    self.matNames.append(lineStr[len("usemtl "):-1])
                if lineStr[:2] == "f ":
                    curFid += 1
                    self.faces.append([])
                    self.facesUV.append([])
                    for oneTerm in lineStr[2:-1].split(" "):
                        iduvids = oneTerm.split("/")
                        self.faces[-1].append(int(iduvids[0]))
                        self.facesUV[-1].append(int(iduvids[1]))

        self.points = np.reshape(self.points, (-1, 3)).astype(float)
        self.uvs = np.reshape(self.uvs, (-1, 2)).astype(float)
        self.faces = np.reshape(self.faces, (-1, 3)).astype(int)
        self.facesUV = np.reshape(self.facesUV, (-1, 3)).astype(int)
        self.matStartId = np.reshape(self.matStartId, -1)

    
    def genObj(self, filePath, bMat=False, bComputeNormal=False, bScaleMesh=False,
            bMaxDimRange=[0.3, 0.5], texture_images=None, texture_rng=None):
        """Write a .blend file containing the generated object using Blender's API"""

        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)

        # Convert numpy arrays to lists for Blender
        verts = [tuple(point) for point in self.points]
        faces = [tuple([idx - 1 for idx in face]) for face in self.faces]
        uvs = [tuple(uv) for uv in self.uvs]
        faces_uv = [tuple(idx - 1 for idx in face_uv) for face_uv in self.facesUV]

        # Create a new mesh and object in Blender
        mesh = bpy.data.meshes.new('mesh')
        mesh.from_pydata(verts, [], faces)
        mesh.update()
        obj = bpy.data.objects.new('Object', mesh)
        bpy.context.collection.objects.link(obj)

        # Create UV map
        uv_layer = mesh.uv_layers.new(name='UVMap')

        # Assign UV coordinates to each loop
        for face_idx, face in enumerate(mesh.polygons):
            face_uv_indices = faces_uv[face_idx]
            for loop_idx, uv_idx in zip(face.loop_indices, face_uv_indices):
                uv_layer.data[loop_idx].uv = uvs[uv_idx]

        # Optionally compute normals
        if bComputeNormal:
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.shade_smooth()

        # Create materials and assign to faces
        if texture_images and texture_rng:
            # Create a mapping from face to subshape index
            face_to_subshape = np.zeros(len(self.faces), dtype=int)
            
            # Fill the face_to_subshape mapping
            for mat_idx in range(len(self.matNames)):
                # Get the corresponding subshape index
                if hasattr(self, 'materialSubShapeIdx') and mat_idx < len(self.materialSubShapeIdx):
                    subshape_idx = self.materialSubShapeIdx[mat_idx]
                else:
                    subshape_idx = 0  # Default if not specified
                    
                # Get the start and end indices for faces with this material
                start_idx = self.matStartId[mat_idx]
                if mat_idx < len(self.matStartId) - 1:
                    end_idx = self.matStartId[mat_idx + 1]
                else:
                    end_idx = len(self.faces)
                    
                # Assign subshape index to these faces
                for face_idx in range(start_idx, end_idx):
                    if face_idx < len(face_to_subshape):
                        face_to_subshape[face_idx] = subshape_idx
            
            # Get unique subshape indices
            unique_subshapes = sorted(set(face_to_subshape))
            num_subshapes = len(unique_subshapes)
            
            # Create texture mapping for each subshape
            subshape_texture_paths = {}
            
            # If there are fewer textures than sub-shapes, we'll need to reuse some
            if len(texture_images) < num_subshapes:
                # Make a copy to avoid changing the original
                available_textures = list(texture_images)
                
                # Randomly select textures for each sub-shape, without replacement if possible
                for subshape_idx in unique_subshapes:
                    if available_textures:
                        texture_image_path = texture_rng.choice(available_textures)
                        available_textures.remove(texture_image_path)
                    else:
                        # If we've used all textures, start reusing them
                        texture_image_path = texture_rng.choice(texture_images)
                    subshape_texture_paths[subshape_idx] = texture_image_path
            else:
                # We have enough textures for each sub-shape
                # Choose textures without replacement to ensure each subshape gets a unique texture
                selected_indices = texture_rng.choice(len(texture_images), num_subshapes, replace=False)
                for i, subshape_idx in enumerate(unique_subshapes):
                    subshape_texture_paths[subshape_idx] = texture_images[selected_indices[i]]
            
            # Create materials for each subshape
            materials = []
            subshape_to_material = {}  # Maps subshape index to material index
            
            # Create a material for each unique subshape
            for subshape_idx in unique_subshapes:
                texture_image_path = subshape_texture_paths[subshape_idx]
                
                # Create a new material
                material_index = len(materials)
                mat = bpy.data.materials.new(name=f"Material_Shape_{subshape_idx}")
                mat.use_nodes = True
                bsdf = mat.node_tree.nodes.get("Principled BSDF")
                if bsdf is None:
                    bsdf = mat.node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')

                # Add an image texture node
                tex_image_node = mat.node_tree.nodes.new('ShaderNodeTexImage')
                try:
                    tex_image_node.image = bpy.data.images.load(texture_image_path)
                except:
                    print(f"Failed to load image {texture_image_path}")
                    continue

                # Connect the image texture node to the BSDF node
                mat.node_tree.links.new(bsdf.inputs['Base Color'], tex_image_node.outputs['Color'])

                materials.append(mat)
                subshape_to_material[subshape_idx] = material_index

                # Assign material to the object
                obj.data.materials.append(mat)
            
            # Now assign material indices to faces based on subshape
            material_indices = [0] * len(mesh.polygons)
            for face_idx, subshape_idx in enumerate(face_to_subshape):
                if face_idx < len(material_indices):
                    material_indices[face_idx] = subshape_to_material[subshape_idx]
        else:
            # Create a default material
            mat = bpy.data.materials.new(name="Default_Material")
            obj.data.materials.append(mat)
            material_indices = [0] * len(mesh.polygons)

        # Now assign material indices to the mesh polygons
        for poly, mat_idx in zip(mesh.polygons, material_indices):
            poly.material_index = mat_idx

        # Save the Blender file
        bpy.ops.wm.save_as_mainfile(filepath=filePath)

        # Clean up: remove the object from the scene
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(mesh, do_unlink=True)

        max_dim = max(self.points.max(axis=0) - self.points.min(axis=0))
        # Adjust material_ids to match the number of materials
        material_ids = [0] * len(self.matNames)
        return max_dim, material_ids


    def genMultiObj(self, folderPath, bComputeNormal=False):
        if len(self.faces) == 0:
            print("no mesh")
            return False
        if bComputeNormal:
            self.computeNormals()

        if len(self.matStartId) == 0:
            print("No mats")
            return

        for im in range(len(self.matStartId)):
            if im == len(self.matStartId) - 1:
                endId = len(self.faces)
            else:
                endId = self.matStartId[im + 1]

            usePoints = np.zeros(len(self.points), int) - 1
            pointMap = np.zeros(len(self.points), int) - 1
            useTex = np.zeros(len(self.uvs), int) - 1
            texMap = np.zeros(len(self.uvs), int) - 1
            nP = 0
            nT = 0
            for i in range(self.matStartId[im], endId):
                for ii in range(3):
                    if pointMap[self.faces[i][ii]-1] == -1:
                        pointMap[self.faces[i][ii] - 1] = nP + 1
                        usePoints[nP] = self.faces[i][ii] - 1
                        nP += 1
                    if texMap[self.facesUV[i][ii]-1] == -1:
                        texMap[self.facesUV[i][ii] - 1] = nT + 1
                        useTex[nT] = self.facesUV[i][ii] - 1
                        nT += 1

            filePath = folderPath + "/%s.obj"%self.matNames[im]

            with open(filePath, "w") as f:
                #write v
                for ii in range(len(usePoints)):
                    if usePoints[ii] == -1:
                        break
                    f.write("v %f %f %f\n"%(self.points[usePoints[ii]][0], self.points[usePoints[ii]][1], self.points[usePoints[ii]][2]))

                if bComputeNormal:
                    for ii in range(len(usePoints)):
                        if usePoints[ii] == -1:
                            break
                        f.write("vn %f %f %f\n"%(self.normals[usePoints[ii]][0], self.normals[usePoints[ii]][1], self.normals[usePoints[ii]][2]))
                        # f.write("vn %f %f %f\n" % (0, 0, 1))
                # write uv
                for ii in range(len(useTex)):
                    if useTex[ii] == -1:
                        break
                    f.write("vt %f %f\n" % (self.uvs[useTex[ii]][0], self.uvs[useTex[ii]][1]))
                #write face
                # f.write("usemtl mat_%d\n"%matId)


                f.write("usemtl %s\n"%self.matNames[im])

                if not bComputeNormal:
                    for i in range(self.matStartId[im], endId):
                        f.write("f %d/%d %d/%d %d/%d\n" %
                                (pointMap[self.faces[i][0]-1], texMap[self.facesUV[i][0]-1],
                                 pointMap[self.faces[i][1]-1], texMap[self.facesUV[i][1]-1],
                                 pointMap[self.faces[i][2]-1], texMap[self.facesUV[i][2]-1]))
                else:
                    for i in range(self.matStartId[im], endId):
                        f.write("f %d/%d/%d %d/%d/%d %d/%d/%d\n" %
                                (pointMap[self.faces[i][0]-1], texMap[self.facesUV[i][0]-1], pointMap[self.faces[i][0]-1],
                                 pointMap[self.faces[i][1]-1], texMap[self.facesUV[i][1]-1], pointMap[self.faces[i][1]-1],
                                 pointMap[self.faces[i][2]-1], texMap[self.facesUV[i][2]-1], pointMap[self.faces[i][2]-1]))

        return True


    def genMatList(self, filePath):
        with open(filePath, "w") as f:
            for matname in self.matNames:
                f.write("%s\n"%matname)
    def genInfo(self, filePath):
        with open(filePath, "w") as f:
            minP = np.min(self.points, axis=0)
            maxP = np.max(self.points, axis=0)
            print(minP, maxP)
            f.write("%f %f %f\n" % (minP[0], minP[1], minP[2]))
            f.write("%f %f %f\n" % (maxP[0], maxP[1], maxP[2]))


    def translate(self, translation):
        self.points += translation

    def rotate(self, axis, degAngle):
        self.points = rotateVector(self.points, axis, np.deg2rad(degAngle))

    def reCenter(self):
        minP = np.min(self.points, 0)
        maxP = np.max(self.points, 0)

        center = 0.5*minP + 0.5*maxP

        self.translate(-center)

    def addShape(self, otherShape, shape_id):
        curPN = len(self.points)
        curUN = len(self.uvs)
        curFN = len(self.faces)
        if curPN == 0:
            self.points = np.copy(otherShape.points)
            
            # Just copy UVs for the first shape
            self.uvs = np.copy(otherShape.uvs)
            
            self.faces = np.copy(otherShape.faces)
            self.facesUV = np.copy(otherShape.facesUV)
            self.matNames += otherShape.matNames
            self.matStartId = np.append(self.matStartId, otherShape.matStartId + curFN).astype(int)
            num_new_mats = len(otherShape.matNames)
            self.materialSubShapeIdx = [shape_id] * num_new_mats  # Initialize the list
        else:
            self.points = np.vstack([self.points, otherShape.points])
            
            # Transform UVs to avoid overlaps between sub-shapes
            # Create a unique UV offset based on shape_id to avoid texture bleeding
            uv_offset_x = (shape_id % 5) * 0.2  # 5 columns of UV space
            uv_offset_y = (shape_id // 5) * 0.2  # Use integer division for rows
            uv_offset = np.array([uv_offset_x, uv_offset_y])
            
            # Copy and modify UVs for additional shapes
            new_uvs = otherShape.uvs.copy()
            
            # Scale UVs to fit within their allocated space (80% of their region)
            new_uvs = new_uvs * 0.18  # Scale down to 90% of the 0.2 space
            
            # Offset UVs to their allocated region
            new_uvs = new_uvs + uv_offset
            
            # Stack with existing UVs
            self.uvs = np.vstack([self.uvs, new_uvs])
            
            self.faces = np.vstack([self.faces, otherShape.faces + curPN])
            self.facesUV = np.vstack([self.facesUV, otherShape.facesUV + curUN])
            self.matNames += otherShape.matNames
            self.matStartId = np.append(self.matStartId, otherShape.matStartId + curFN).astype(int)
            num_new_mats = len(otherShape.matNames)
            
            # Add new shape_id entries to materialSubShapeIdx
            if not hasattr(self, 'materialSubShapeIdx'):
                self.materialSubShapeIdx = []
            self.materialSubShapeIdx += [shape_id] * num_new_mats

    def _addMorphCircle(self, center=(0,0,0), axisA = 1.0, axisB = 1.0, X=(1,0,0), Z=(0,0,1), circelRes = (50, 100), matName = "mat"):
        circelRes = (int(circelRes[0]), int(circelRes[1]))
        
        X = np.reshape(X,3)
        Z = np.reshape(Z,3)
        Y = np.cross(Z, X)
        startPId = len(self.points)
        startUId = len(self.uvs)
        startFaceId = len(self.faces)

        center = np.reshape(center, 3)
        points = []
        uvs = []
        points.append(center)
        uvs.append((0.5, 0.5))

        # create points
        for iy in range(1, circelRes[0]):
            for ix in range(circelRes[1]):
                ra = float(axisA) * iy / (circelRes[0] - 1)
                rb = float(axisB) * iy / (circelRes[0] - 1)


                phi = float(ix) /(circelRes[1]) * 2.0 * np.pi

                x = ra *  np.cos(phi)
                y = rb *  np.sin(phi)


                p = x*X + y*Y + center
                points.append(p)
        # create uvs
        for iy in range(1, circelRes[0]):
            for ix in range(circelRes[1]):
                ra = float(axisA) * iy / (circelRes[0] - 1)
                rb = float(axisB) * iy / (circelRes[0] - 1)

                phi = float(ix) / (circelRes[1]) * 2.0 * np.pi

                x = ra * np.cos(phi)
                y = rb * np.sin(phi)

                ux = float(iy) / (circelRes[0] - 1) * np.cos(phi)
                uy = float(iy) / (circelRes[0] - 1) * np.sin(phi)

                u = 0.5 * ux + 0.5
                v = 0.5 * uy + 0.5
                uvs.append((u, v))

        if startPId == 0:
            self.points = np.reshape(points, (-1, 3))
            self.uvs = np.rehsape(uvs, (-1, 2))
        else:
            self.points = np.vstack([self.points, points])
            self.uvs = np.vstack([self.uvs, uvs])

        # create faces
        tempFaces = []
        for iy in range(circelRes[0] - 1):
            for ix in range(circelRes[1]):
                if iy == 0:
                    curId = 1
                    rightId = 1
                    bottomId = 1 + ix + 1
                    if ix == circelRes[1] - 1:
                        rightBottomId = 1 + 1
                    else:
                        rightBottomId = 1 + ix + 1 + 1
                else:
                    curId = 1 + (iy - 1) * circelRes[1] + ix + 1
                    bottomId = 1 + (iy) * circelRes[1] + ix + 1
                    if ix == circelRes[1] - 1:
                        rightId = 1 + (iy - 1) * circelRes[1] + 1
                        rightBottomId = 1 + (iy) * circelRes[1] + 1
                    else:
                        rightId = 1 + (iy - 1) * circelRes[1] + ix + 1 + 1
                        rightBottomId = 1 + (iy) * circelRes[1] + ix + 1 + 1
                if iy != 0:
                    tempFaces.append((curId, rightBottomId, rightId))
                tempFaces.append((curId, bottomId, rightBottomId))

        tempFaces = np.reshape(tempFaces, (-1,3))

        if len(self.faces) == 0:
            self.faces = tempFaces.copy()
            self.facesUV = tempFaces.copy()
            self.matStartId = self.matStartId = np.asarray([0],int)
        else:
            self.faces = np.vstack([self.faces, tempFaces+startPId])
            self.facesUV = np.vstack([self.facesUV, tempFaces+startUId])
            self.matStartId = np.append(self.matStartId, [startFaceId])

        self.matNames.append(matName)


class HeightFieldCreator:
    def __init__(self, initSize=(5, 5), maxHeight=(-0.2, 0.2), bFixCorner=True, rng=None):
        self.initSize = initSize
        self.bFixCorner = bFixCorner
        self.initNum = self.initSize[0] * self.initSize[1]
        self.maxHeight = maxHeight
        self.heightField = None
        self.rng = rng if rng is not None else np.random  # Use the provided RNG or default to np.random

    def __initializeHeigthField(self):
        heights = self.rng.uniform(self.maxHeight[0], self.maxHeight[1], self.initNum)
        initHeightField = heights.reshape(self.initSize)
        self.initHeightField = initHeightField
        return initHeightField

    def genHeightField(self, targetSize = (36, 36)):
        halfSize = (int(targetSize[0]/6*5), int(targetSize[1]/6*5))
        if halfSize[0] < self.initSize[0] or halfSize[1] < self.initSize[1]:
            print("target size should be double as init size")
            return None
        initHeight = self.__initializeHeigthField()
        if self.bFixCorner:
            bounder = np.zeros((self.initSize[0]+2, self.initSize[1]+2))
            bounder[1:-1, 1:-1] = initHeight
            initHeight = bounder
        heightField_half = cv2.resize(initHeight, halfSize, interpolation=cv2.INTER_CUBIC)#
        if self.bFixCorner:
            bounder = np.zeros(halfSize)
            bounder[1:-1, 1:-1] = heightField_half[1:-1, 1:-1]
            initHeight = bounder
        heightField = cv2.resize(initHeight, targetSize)  #
        self.heightField = heightField
        self.targetSize = targetSize
        return heightField

    def genObj(self, filePath):
        if type(self.heightField) == type(None):
            print("no generated height fields")
            return False


        with open(filePath, "w") as f:
            #write v
            for iy in range(self.targetSize[0]):
                for ix in range(self.targetSize[1]):
                    f.write("v %f %f %f\n"%
                            (float(ix)/(self.targetSize[1]-1),
                             float(iy)/(self.targetSize[0]-1),
                             self.heightField[iy, ix]))
            #write f
            for iy in range(self.targetSize[0]-1):
                for ix in range(self.targetSize[1]-1):
                    curId = iy * self.targetSize[1] + ix + 1
                    rightId = iy * self.targetSize[1] + ix + 1 +1
                    bottomId = (iy+1) * self.targetSize[1] + ix+1
                    rightBottomId = (iy+1) * self.targetSize[1] + ix + 1+1
                    f.write("f %d %d %d\n"%
                            (curId, rightBottomId, rightId))
                    f.write("f %d %d %d\n" %
                            (curId, bottomId, rightBottomId))
        return True

class Ellipsoid(Shape):
    # meshRes: rows x columns, latitude res x longitude res
    def __init__(self, a = 1.0, b = 1.0, c = 1.0, meshRes = (50, 100)):
        super(Ellipsoid, self).__init__()
        if meshRes[1] % 2 != 0:
            print("WARN: longitude res is supposed to be even")
        self.axisA = a
        self.axisB = b
        self.axisC = c
        self.meshRes = meshRes

        self.numPoints = (self.meshRes[0] - 2) * self.meshRes[1] + 2


    def genShape(self, matName = "mat"):
        super(Ellipsoid, self).__init__()


        self.points.append((0,0,self.axisC))
        self.uvs.append((0,0))


        #create points
        for iy in range(1,self.meshRes[0]-1):
            for ix in range(self.meshRes[1]):
                v = float(iy) / (self.meshRes[0]-1)
                u = float(ix) / (self.meshRes[1]/2)

                theta = np.pi/2.0 - v * np.pi
                phi = u * np.pi

                x = self.axisA * np.cos(theta) * np.cos(phi)
                y = self.axisB * np.cos(theta) * np.sin(phi)
                z = self.axisC * np.sin(theta)

                self.points.append((x,y,z))
        self.points.append((0, 0, -self.axisC))

        #create uvs
        for iy in range(1, self.meshRes[0] - 1):
            for ix in range(self.meshRes[1]):
                v = float(iy) / (self.meshRes[0] - 1)
                u = float(ix) / (self.meshRes[1] / 2)
                if u > 1.0:
                    u = u - 1.0
                self.uvs.append((u,v))
        self.uvs.append((1.0,1.0))

        #create faces
        for iy in range(self.meshRes[0]-1):
            for ix in range(self.meshRes[1]):
                if iy == 0:
                    curId = 1
                    rightId = 1
                    bottomId = 1 + ix + 1
                    if ix == self.meshRes[1] - 1:
                        rightBottomId = 1 + 1
                    else:
                        rightBottomId = 1+ ix + 1 + 1
                elif iy == self.meshRes[0]-2:
                    curId = 1 + (iy-1) * self.meshRes[1] + ix + 1
                    bottomId = 1 + (iy) * self.meshRes[1] + 1
                    if ix == self.meshRes[1] - 1:
                        rightId = 1 + (iy-1) * self.meshRes[1] + 1
                    else:
                        rightId = 1 + (iy-1) * self.meshRes[1] + ix + 1 + 1
                    rightBottomId = 1 + (iy) * self.meshRes[1] + 1
                else:
                    curId = 1 + (iy - 1) * self.meshRes[1] + ix + 1
                    bottomId = 1 + (iy) * self.meshRes[1] + ix + 1
                    if ix == self.meshRes[1] - 1:
                        rightId = 1 + (iy - 1) * self.meshRes[1] + 1
                        rightBottomId = 1 + (iy) * self.meshRes[1] + 1
                    else:
                        rightId = 1 + (iy - 1) * self.meshRes[1] + ix + 1 + 1
                        rightBottomId = 1 + (iy) * self.meshRes[1]+ ix + 1 + 1
                if iy != 0:
                    self.faces.append((curId, rightBottomId, rightId))
                if iy != self.meshRes[0]-2:
                    self.faces.append((curId, bottomId, rightBottomId))

        self.points = np.reshape(self.points, (-1,3)).astype(float)
        self.uvs = np.reshape(self.uvs, (-1,2)).astype(float)
        self.faces = np.reshape(self.faces, (-1, 3)).astype(int)
        self.facesUV = np.copy(self.faces)
        self.matNames = [matName]
        self.matStartId = np.asarray([0],int)

    def applyHeightField(self, heightFields):
        if len(self.points) == 0:
            print("no points")
            return False
        if len(heightFields.shape) != 2 and len(heightFields.shape) != 3:
            print("wrong shape of heightfiels")
            return False
        if len(heightFields.shape) == 3:
            heightField = heightFields[0]


        for i,point in enumerate(self.points):
            uv = self.uvs[i]
            normal = np.reshape(point,-1) / (self.axisA, self.axisB, self.axisC)
            normal = normal / np.linalg.norm(normal)
            xy = uv * (heightField.shape[1], heightField.shape[0])
            h = subPix(heightField, xy[0], xy[1])#cv2.getRectSubPix(heightField, (1,1), (xy[0], xy[1]))

            self.points[i] = point + normal * h


class Cube(Shape):
    """faces:
    front: c = 1
    back: c = -1
    left: a = -1
    right: a = 1
    up: b = 1
    down: b = -1

    """
    def __init__(self, a=1.0, b=1.0, c=1.0, faceRes=(50, 50)):
        super(Cube,self).__init__()
        self.axisA = a
        self.axisB = b
        self.axisC = c
        self.faceRes = faceRes
        self.pointNumPerFace = faceRes[0] * faceRes[1]

        self.numPoints = (self.faceRes[0]) * self.faceRes[1] * 6


    def genShape(self, matName = "mat"):
        """ compute points, uvs, faces according to the parameters for the shape type """
        super(Cube, self).__init__()


        #uvs
        for iy in range(self.faceRes[0]):
            for ix in range(self.faceRes[1]):
                u = float(ix) / (self.faceRes[1] - 1)
                v = float(iy) / (self.faceRes[0] - 1)
                self.uvs.append((u,v))
        self.uvs = np.reshape(self.uvs, (-1,2))

        #face:
        #oneFace:
        oneFaces = []
        for iy in range(self.faceRes[0] - 1):
            for ix in range(self.faceRes[1] - 1):
                curId = iy * self.faceRes[1] + ix + 1
                rightId = iy * self.faceRes[1] + ix + 1 + 1
                bottomId = (iy + 1) * self.faceRes[1] + ix + 1
                rightBottomId = (iy + 1) * self.faceRes[1] + ix + 1 + 1

                oneFaces.append((curId, rightBottomId, rightId))
                oneFaces.append((curId, bottomId, rightBottomId))
        oneFaces = np.reshape(oneFaces, (-1,3)).astype(int)
        self.faces = np.vstack([oneFaces,
                                oneFaces + self.pointNumPerFace,
                                oneFaces + self.pointNumPerFace*2,
                                oneFaces + self.pointNumPerFace*3,
                                oneFaces + self.pointNumPerFace*4,
                                oneFaces + self.pointNumPerFace*5])
        self.facesUV = self.faces.copy()#np.row_stack([oneFaces, oneFaces, oneFaces, oneFaces, oneFaces, oneFaces])

        #points
        #front
        for uv in self.uvs:
            xy = uv * (self.axisA, -self.axisB) * 2.0 + (-self.axisA, self.axisB)
            point = (xy[0], xy[1], self.axisC)
            self.points.append(point)

        # back
        for uv in self.uvs:
            xy = uv * (self.axisA, self.axisB) * 2.0 + (-self.axisA, -self.axisB)
            point = (xy[0], xy[1], -self.axisC)
            self.points.append(point)

        #left
        for uv in self.uvs:
            zy = uv * (self.axisC, -self.axisB) * 2.0 + (-self.axisC, self.axisB)
            point = (-self.axisA, zy[1], zy[0])
            self.points.append(point)

        # right
        for uv in self.uvs:
            zy = uv * (-self.axisC, -self.axisB) * 2.0 + (self.axisC, self.axisB)
            point = (self.axisA, zy[1], zy[0])
            self.points.append(point)

        # up
        for uv in self.uvs:
            xz = uv * (self.axisA, self.axisC) * 2.0 + (-self.axisA, -self.axisC)
            point = (xz[0], self.axisB, xz[1])
            self.points.append(point)

        # down
        for uv in self.uvs:
            xz = uv * (self.axisA, -self.axisC) * 2.0 + (-self.axisA, self.axisC)
            point = (xz[0], -self.axisB, xz[1])
            self.points.append(point)

        self.uvs = np.reshape(np.vstack([self.uvs, self.uvs, self.uvs, self.uvs, self.uvs, self.uvs]), (-1, 2))

        self.points = np.reshape(self.points, (-1, 3)).astype(float)
        self.uvs = np.reshape(self.uvs, (-1, 2)).astype(float)
        self.faces = np.reshape(self.faces, (-1, 3)).astype(int)
        self.facesUV = np.reshape(self.facesUV, (-1, 3)).astype(int)
        self.matNames = ["%s_%d"%(matName, 0),
                         "%s_%d"%(matName, 1),
                         "%s_%d"%(matName, 2),
                         "%s_%d"%(matName, 3),
                         "%s_%d"%(matName, 4),
                         "%s_%d"%(matName, 5)]
        numFacePerFace = len(oneFaces)
        self.matStartId = np.asarray([0,
                                      numFacePerFace,
                                      numFacePerFace*2,
                                      numFacePerFace*3,
                                      numFacePerFace*4,
                                      numFacePerFace*5],int)


    def applyHeightField(self, heightFields):
        if len(self.points) == 0:
            print("no points")
            return False
        if len(heightFields.shape) != 2 and len(heightFields.shape) != 3:

            print("wrong shape of heightfiels")
            return False

        if len(heightFields.shape) == 2:
            newH = []
            for i in range(6):
                newH.append(heightFields)
            heightFields = newH
        else:
            if heightFields.shape[0] < 6:
                newH = []
                for i in range(6):
                    if i < heightFields.shape[0]:
                        newH.append(heightFields[i])
                    else:
                        newH.append(heightFields[-1])
                heightFields = newH

        # modify points
        # front
        heightField = heightFields[0]
        normal = np.asarray((0,0,1))
        offSet = 0
        for i in range(self.pointNumPerFace):
            uv = self.uvs[i]
            xy = uv * (heightField.shape[1], heightField.shape[0])
            h = subPix(heightField, xy[0], xy[1])
            self.points[i + offSet] = self.points[i+offSet] + h * normal


        # back
        heightField = heightFields[1]
        normal = np.asarray((0, 0, -1))
        offSet = self.pointNumPerFace*1
        for i in range(self.pointNumPerFace):
            uv = self.uvs[i]
            xy = uv * (heightField.shape[1], heightField.shape[0])
            h = subPix(heightField, xy[0], xy[1])
            self.points[i + offSet] = self.points[i + offSet] + h * normal

        # left
        heightField = heightFields[2]
        normal = np.asarray((-1, 0, 0))
        offSet = self.pointNumPerFace * 2
        for i in range(self.pointNumPerFace):
            uv = self.uvs[i]
            xy = uv * (heightField.shape[1], heightField.shape[0])
            h = subPix(heightField, xy[0], xy[1])
            self.points[i + offSet] = self.points[i + offSet] + h * normal

        # right
        heightField = heightFields[3]
        normal = np.asarray((1, 0, 0))
        offSet = self.pointNumPerFace * 3
        for i in range(self.pointNumPerFace):
            uv = self.uvs[i]
            xy = uv * (heightField.shape[1], heightField.shape[0])
            h = subPix(heightField, xy[0], xy[1])
            self.points[i + offSet] = self.points[i + offSet] + h * normal

        # up
        heightField = heightFields[4]
        normal = np.asarray((0, 1, 0))
        offSet = self.pointNumPerFace * 4
        for i in range(self.pointNumPerFace):
            uv = self.uvs[i]
            xy = uv * (heightField.shape[1], heightField.shape[0])
            h = subPix(heightField, xy[0], xy[1])
            self.points[i + offSet] = self.points[i + offSet] + h * normal

        # down
        heightField = heightFields[5]
        normal = np.asarray((0, -1, 0))
        offSet = self.pointNumPerFace * 5
        for i in range(self.pointNumPerFace):
            uv = self.uvs[i]
            xy = uv * (heightField.shape[1], heightField.shape[0])
            h = subPix(heightField, xy[0], xy[1])
            self.points[i + offSet] = self.points[i + offSet] + h * normal

class Cylinder(Shape):
    def __init__(self, a=1.0, b=1.0, c=1.0, meshRes=(50, 150), radiusRes = 20):
        super(Cylinder,self).__init__()

        self.axisA = a
        self.axisB = b
        self.axisC = c
        self.meshRes = meshRes



    def genShape(self, matName = "mat"):
        super(Cylinder, self).__init__()


        # create points
        for iy in range(self.meshRes[0]):
            for ix in range(self.meshRes[1]):
                v = float(iy) / (self.meshRes[0] - 1)
                u = float(ix) / (self.meshRes[1] / 2)

                phi = u * np.pi

                x = self.axisA * np.cos(phi)
                y = self.axisB * np.sin(phi)
                z = self.axisC - self.axisC * v * 2.0

                self.points.append((x, y, z))



        # create uvs
        for iy in range(self.meshRes[0]):
            for ix in range(self.meshRes[1]):
                v = float(iy) / (self.meshRes[0] - 1)
                u = float(ix) / (self.meshRes[1] / 2)
                if u > 1.0:
                    u = u - 1.0
                self.uvs.append((u, v))


        # create faces
        for iy in range(self.meshRes[0]-1):
            for ix in range(self.meshRes[1]):

                curId = iy * self.meshRes[1] + ix + 1
                bottomId = (iy + 1) * self.meshRes[1] + ix + 1
                if ix == self.meshRes[1] - 1:
                    rightId = iy * self.meshRes[1] + 1
                    rightBottomId = (iy + 1) * self.meshRes[1] + 1
                else:
                    rightId = (iy) * self.meshRes[1] + ix + 1 + 1
                    rightBottomId = (iy + 1) * self.meshRes[1] + ix + 1 + 1

                self.faces.append((curId, rightBottomId, rightId))
                self.faces.append((curId, bottomId, rightBottomId))

        self.points = np.reshape(self.points, (-1, 3)).astype(float)
        self.uvs = np.reshape(self.uvs, (-1, 2)).astype(float)
        self.faces = np.reshape(self.faces, (-1, 3)).astype(int)
        self.facesUV = np.copy(self.faces)
        self.matNames = ["%s_0"%matName]
        self.matStartId = self.matStartId = np.asarray([0],int)

        self._addMorphCircle((0, 0, self.axisC), self.axisA, self.axisB, X=(1,0,0), Z=(0,0,1),
                             circelRes=[self.meshRes[0]/2, self.meshRes[1]], matName="%s_1"%matName)

        self._addMorphCircle((0, 0, -self.axisC), self.axisA, self.axisB, X=(1, 0, 0), Z=(0, 0, -1),
                             circelRes=[self.meshRes[0]/2, self.meshRes[1]], matName="%s_2" % matName)



    def applyHeightField(self, heightFields, smoothCircleBoundRate = 0.25):
        if len(self.points) == 0:
            print("no points")
            return False
        if len(heightFields.shape) != 2 and len(heightFields.shape) != 3:

            print("wrong shape of heightfiels")
            return False

        if len(heightFields.shape) == 2:
            newH = []
            for i in range(3):
                newH.append(heightFields)
            heightFields = newH
        else:
            if heightFields.shape[0] < 3:
                newH = []
                for i in range(3):
                    if i < heightFields.shape[0]:
                        newH.append(heightFields[i])
                    else:
                        newH.append(heightFields[-1])
                heightFields = newH

        heightField = heightFields[0]
        i = 0
        for iy in range(self.meshRes[0]):
            for ix in range(self.meshRes[1]):
                u = float(ix) / (self.meshRes[1] / 2)

                phi = u * np.pi

                x = self.axisA * np.cos(phi)
                y = self.axisB * np.sin(phi)

                normal = np.reshape((x,y,0),-1)/ (self.axisA, self.axisB, self.axisC)
                normal = normal / np.linalg.norm(normal)
                xy = self.uvs[i] * (heightField.shape[1], heightField.shape[0])
                h = subPix(heightField, xy[0], xy[1])  # cv2.getRectSubPix(heightField, (1,1), (xy[0], xy[1]))

                self.points[i] += normal * h
                i+=1

        heightField = heightFields[1]
        circelRes = [int(self.meshRes[0] / 2), int(self.meshRes[1])]
        for iy in range(circelRes[0] - 1):
            for ix in range(circelRes[1]):
                normal = np.reshape((0, 0, 1),-1)
                xy = self.uvs[i] * (heightField.shape[1], heightField.shape[0])
                h = subPix(heightField, xy[0], xy[1])  # cv2.getRectSubPix(heightField, (1,1), (xy[0], xy[1]))

                l = np.linalg.norm(self.uvs[i] * 2 - 1.0)
                if l > smoothCircleBoundRate:
                    r = (1.0-l)/(1.0-smoothCircleBoundRate)
                    h = (1.0-(r-1.0)**2.0)* h

                self.points[i] += normal * h
                i+=1

        heightField = heightFields[2]
        circelRes = [int(self.meshRes[0] / 2), int(self.meshRes[1])]
        for iy in range(circelRes[0] - 1):
            for ix in range(circelRes[1]):
                normal = np.reshape((0, 0, -1), -1)
                xy = self.uvs[i] * (heightField.shape[1], heightField.shape[0])
                h = subPix(heightField, xy[0], xy[1])  # cv2.getRectSubPix(heightField, (1,1), (xy[0], xy[1]))

                l = np.linalg.norm(self.uvs[i] * 2 - 1.0)
                if l > smoothCircleBoundRate:
                    r = (1.0 - l) / (1.0 - smoothCircleBoundRate)
                    h = (1.0 - (r - 1.0) ** 2.0) * h

                self.points[i] += normal * h
                i += 1


import numpy as np

class MultiShape(Shape):
    """
    Shape types:
        0 – Ellipsoid
        1 – Cube
        2 – Cylinder

    Placement strategy:
        • for each primitive pick an area‑weighted random surface point
        • translate primitives so all sampled points coincide at (0,0,0)
        • if a new primitive’s AABB ends up entirely inside an existing one,
          nudge it 2 % of its own size outward along centre→anchor
        • after merging, centre the cluster, then apply one random spin
          and a small random translation (jitter) to boost visual variety
    """


    def __init__(self,
                 numShape=None,
                 candShapes=(0, 1, 2),
                 shape_counts=None,
                 smoothPossibility=0.1,
                 axisRange=(0.35, 1.55),
                 heightRangeRate=(0, 0.2),
                 rotateRange=(0, 180),
                 translation_control=None,
                 rotation_rng=None,
                 translation_rng=None,
                 size_rng=None):
        super().__init__()

        self.shape_counts      = shape_counts
        self.candShapes        = list(candShapes)
        self.smoothPossibility = smoothPossibility
        self.axisRange         = (0.7, 1.2) if axisRange == (0.35, 1.55) else axisRange
        self.heightRangeRate   = heightRangeRate
        self.rotateRange       = rotateRange
        self.translation_control = translation_control if translation_control is not None else 1.0

        self.rotation_rng    = rotation_rng    if rotation_rng    is not None else np.random
        self.translation_rng = translation_rng if translation_rng is not None else np.random
        self.size_rng        = size_rng        if size_rng        is not None else np.random

        # number of primitives
        if self.shape_counts is not None:
            self.numShape = sum(self.shape_counts.values())
        elif numShape is not None:
            self.numShape = int(numShape)
        else:
            self.numShape = 6

        self.materialSubShapeIdx = []

    def _create_shape(self, shape_type, axis_vals):
        if shape_type == 0:
            return Ellipsoid(*axis_vals)
        if shape_type == 1:
            return Cube(*axis_vals)
        if shape_type == 2:
            return Cylinder(*axis_vals)
        raise ValueError(f"Unknown shape type {shape_type}")

    def _get_shape_info(self, shape, position=None):
        if position is None:
            position = np.zeros(3)
        pts   = shape.points + position
        return dict(points      = pts,
                    center      = pts.mean(0),
                    min         = pts.min(0),
                    max         = pts.max(0),
                    dimensions  = pts.max(0) - pts.min(0))

    def _spin_about_point(self, shape, pivot, rng):
        """Rotate a shape in‑place around a pivot, keeping that pivot fixed."""
        ang = rng.uniform(0.0, 360.0, 3)          # XYZ Euler in degrees
        shape.translate(-pivot)
        shape.rotate((1, 0, 0), ang[0])
        shape.rotate((0, 1, 0), ang[1])
        shape.rotate((0, 0, 1), ang[2])
        shape.translate(pivot)


    def _sample_random_point(self, shape, rng):
        """Uniform surface sampler."""
        faces = shape.faces            # (N,3) 1‑based indices
        verts = shape.points           # (M,3)

        v0 = verts[faces[:, 0] - 1]
        v1 = verts[faces[:, 1] - 1]
        v2 = verts[faces[:, 2] - 1]
        areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

        f_idx = np.searchsorted(np.cumsum(areas), rng.random() * areas.sum())

        u = rng.random()
        v = rng.random()
        if u + v > 1.0:                            # fold back into simplex
            u, v = 1.0 - u, 1.0 - v
        w = 1.0 - u - v
        return u * v0[f_idx] + v * v1[f_idx] + w * v2[f_idx]

    def _aabb_inside(self, min_a, max_a, min_b, max_b, eps=1e-4):
        return np.all(min_a >= min_b - eps) and np.all(max_a <= max_b + eps)


    def genShape(self, no_hf=False):
        super().genShape()               # clear parent buffers
        rng_r, rng_t, rng_s = self.rotation_rng, self.translation_rng, self.size_rng


        def _random_axes():
            axis = rng_s.uniform(self.axisRange[0], self.axisRange[1], 3)
            
            if rng_s.rand() < 0.9:  # 90% chance of stretching
                stretch_type = rng_s.rand()
                
                if stretch_type < 0.5:  # 50% of 90% = 45% chance - stretch one axis
                    sa = rng_s.randint(0, 3)
                    sf = rng_s.uniform(1.2, 2.2)
                    tf = rng_s.uniform(0.2, 0.6)
                    
                    # Stretch one dimension
                    axis[sa] *= sf
                    # Make it thinner in other dimensions
                    axis[[i for i in range(3) if i != sa]] *= tf
                
                else:  # 50% of 90% = 45% chance - stretch two axes

                    thin_axis = rng_s.randint(0, 3)
                    stretch_axes = [i for i in range(3) if i != thin_axis]
                    

                    sf1 = rng_s.uniform(1.2, 2.0)
                    sf2 = rng_s.uniform(1.2, 2.0)
                    tf = rng_s.uniform(0.3, 0.6)  # Thinning factor for the remaining axis
                    
                    # Apply stretching to the selected axes
                    axis[stretch_axes[0]] *= sf1
                    axis[stretch_axes[1]] *= sf2
                    
                    # Apply thinning to the remaining axis
                    axis[thin_axis] *= tf

                
            return axis

        specs = []
        if self.shape_counts:
            for stype, cnt in self.shape_counts.items():
                specs.extend([dict(type=stype)] * cnt)
        else:
            for _ in range(self.numShape):
                specs.append(dict(type=rng_r.choice(self.candShapes)))

        for spec in specs:
            spec['axis'] = _random_axes()
            # h‑fields
            hfs = []
            minA = spec['axis'].min() * 2.0
            maxH = rng_s.uniform(self.heightRangeRate[0] * minA,
                                self.heightRangeRate[1] * minA, 6)
            for m in maxH:
                if no_hf or m == 0 or rng_s.rand() < self.smoothPossibility:
                    hfs.append(np.zeros((36, 36)))
                else:
                    hfg = HeightFieldCreator(maxHeight=(-m, m), rng=rng_s)
                    hfs.append(hfg.genHeightField())
            spec['hfs'] = np.asarray(hfs)
            spec['rot'] = rng_r.uniform(self.rotateRange[0],
                                        self.rotateRange[1], 3)


        shapes = []
        for i, spec in enumerate(specs):
            shp = self._create_shape(spec['type'], spec['axis'])
            shp.genShape(f"mat_{i}")
            shp.applyHeightField(spec['hfs'])
            shp.rotate((1, 0, 0), spec['rot'][0])
            shp.rotate((0, 1, 0), spec['rot'][1])
            shp.rotate((0, 0, 1), spec['rot'][2])
            shapes.append(shp)


        positions = []
        

        anchor0 = self._sample_random_point(shapes[0], rng_t)
        shapes[0].translate(-anchor0)
        positions.append(-anchor0)
        self.addShape(shapes[0], 0)
        

        assembled = Shape()
        assembled.points = np.copy(shapes[0].points)
        assembled.faces = np.copy(shapes[0].faces)
        

        for i in range(1, len(shapes)):

            connection_point = self._sample_random_point(assembled, rng_t)
            

            shape_point = self._sample_random_point(shapes[i], rng_t)
            

            trans = connection_point - shape_point
            shapes[i].translate(trans)
            positions.append(trans)
            

            self._spin_about_point(shapes[i], connection_point, rng_r)
            
            # Add to self for final output
            self.addShape(shapes[i], i)
            

            points_offset = len(assembled.points)
            faces_offset = len(assembled.faces)
            
            # Add points from the new shape
            new_points = shapes[i].points.copy()  # Already translated
            assembled.points = np.vstack([assembled.points, new_points])
            
            # Add faces from the new shape, updating indices
            new_faces = shapes[i].faces.copy()
            new_faces = new_faces + points_offset 
            
            if len(assembled.faces) > 0 and len(new_faces) > 0:
                assembled.faces = np.vstack([assembled.faces, new_faces])
            elif len(new_faces) > 0:
                assembled.faces = new_faces


        primitive_ids, axis_vals_s, translations = [], [], []
        translation1s, rotations = [], []
        rotation1s, height_fields_s = [], []

        for i, spec in enumerate(specs):
            primitive_ids  .append(spec['type'])
            axis_vals_s    .append(spec['axis'])
            translations   .append(positions[i])
            translation1s  .append(np.zeros(3))
            rotations      .append(spec['rot'])
            rotation1s     .append(np.zeros(3))
            height_fields_s.append(spec['hfs'])


        self.reCenter()

        return primitive_ids, axis_vals_s, translations, translation1s, rotations, rotation1s, height_fields_s


def createShapes(outFolder, shapeNum, subObjNum = 6):
    if not os.path.isdir(outFolder):
        os.makedirs(outFolder)

    for i in range(shapeNum):
        ms = MultiShape(
            shape_counts=shape_counts,
            smoothPossibility=smooth_probability,
            axisRange=(0.7, 1.2),  # Use consistent size range
            heightRangeRate=(0, 0.2),
            translateRangeRate=(0.3, 0.5),
            rotateRange=(0, 180)
        )
        subFolder = outFolder + "/Shape__%d"%i
        if not os.path.isdir(subFolder):
            os.makedirs(subFolder)
        ms.genShape()
        ms.genObj(subFolder + "/object.obj", bMat=True)
        ms.genMatList(subFolder + "/object.txt")
        ms.genInfo(subFolder + "/object.info")


def createVarObjShapes(outFolder, shapeIds, uuid_str='', shape_counts=None,
                       sub_obj_nums=[1,2,3,4,5,6,7,8,9], sub_obj_num_poss=[1,2,3,7,10,7,3,2,1],
                       bMultiObj=False, bPermuteMat=True, candShapes=[0,1,2],
                       bScaleMesh=False, bMaxDimRange=[0.3,0.5], smooth_probability=1.0,
                       no_hf=False, texture_images=None, texture_rng=None,
                       translation_control=None, rotation_rng=None, translation_rng=None, size_rng=None):
    """
    Create shapes with specified counts and types, or randomly if shape_counts is None, and save the .blend files.
    """
    if not os.path.isdir(outFolder):
        os.makedirs(outFolder)

    output_paths = []
    shapes_parameters = []

    if shape_counts is not None:
        fixed_sub_obj_num = sum(shape_counts.values())
        fixed_shape_counts = shape_counts
    else:
        # Randomly select sub_obj_num
        sub_obj_bound = np.reshape(sub_obj_num_poss, -1).astype(float)
        sub_obj_bound = sub_obj_bound / np.sum(sub_obj_bound)
        sub_obj_bound = np.cumsum(sub_obj_bound)  # normalized, cumulative sum of subObjPoss (possibility)

        if sub_obj_bound[-1] != 1.0:
            print("Incorrect bound")
            sub_obj_bound[-1] = 1.0  # setting 0.999... to 1.0

        counts = np.zeros(len(sub_obj_nums))

        chooses = np.random.uniform(0, 1.0, len(shapeIds))

    for ii, i in enumerate(shapeIds):  # for each MultiShape
        shape_parameters = {'uuid_str': uuid_str}

        if shape_counts is not None:
            sub_obj_num = fixed_sub_obj_num
            shape_parameters['sub_obj_num'] = sub_obj_num
            shape_parameters['sub_objs'] = [{} for _ in range(sub_obj_num)]
            print(f'i: {i}, sub_obj_num: {sub_obj_num}')
            ms = MultiShape(shape_counts=shape_counts,
                            smoothPossibility=smooth_probability,
                            axisRange=(0.7, 1.2),  # Use consistent size range
                            translation_control=translation_control,
                            rotation_rng=rotation_rng,
                            translation_rng=translation_rng,
                            size_rng=size_rng)
        else:
            # Randomly select sub_obj_num
            choose = chooses[ii]
            sub_obj_num = sub_obj_nums[-1]
            for iO in range(len(sub_obj_bound)):
                if choose < sub_obj_bound[iO]:
                    sub_obj_num = sub_obj_nums[iO]
                    counts[iO] += 1
                    break
            shape_parameters['sub_obj_num'] = sub_obj_num
            shape_parameters['sub_objs'] = [{} for _ in range(sub_obj_num)]
            print(f'i: {i}, sub_obj_num: {sub_obj_num}')
            ms = MultiShape(numShape=sub_obj_num,
                            candShapes=candShapes,
                            smoothPossibility=smooth_probability,
                            axisRange=(0.7, 1.2),  # Use consistent size range
                            translation_control=translation_control,
                            rotation_rng=rotation_rng,
                            translation_rng=translation_rng,
                            size_rng=size_rng)

        sub_objs_vals = list(ms.genShape(no_hf=no_hf))
        if bPermuteMat:
            ms.permuteMatIds()

        # Define the output file path
        filename = f'object_{ii:03d}.blend'

        output_path = Path(outFolder) / filename
        output_paths.append(str(output_path.resolve()))


        max_dim, material_ids = ms.genObj(
            str(output_path),
            bMat=True,
            bComputeNormal=True,
            bScaleMesh=bScaleMesh,
            bMaxDimRange=bMaxDimRange,
            texture_images=texture_images,
            texture_rng=texture_rng
        )
        shape_parameters['max_dim'] = max_dim

        # Adjust material_ids to match the number of sub-objects
        material_ids = [0] * sub_obj_num
        sub_objs_vals.append(material_ids)

        for i_key, key in enumerate(['primitive_id', 'axis_vals', 'translation', 'translation1', 'rotation', 'rotation1', 'height_fields', 'material_id']):
            for iS in range(sub_obj_num):
                sub_obj_val = sub_objs_vals[i_key]
                if isinstance(sub_obj_val, list) and len(sub_obj_val) > iS:
                    val = sub_obj_val[iS]
                else:
                    val = 0  # Default value if not enough data
                shape_parameters['sub_objs'][iS][key] = val.tolist() if isinstance(val, np.ndarray) else val

        shapes_parameters.append(shape_parameters)

    return output_paths, shapes_parameters



mat_keys = ["name", "basecolor", "metallic", "normal", "roughness"]


def get_matsynth_material(base_output_dir):
    ds = load_dataset(
        "gvecchio/MatSynth",
        streaming=True,
    )


    ds = ds.select_columns(mat_keys)

    ds = ds.shuffle(buffer_size=1)



    # save files for a single material
    for i, x in enumerate(ds['train']):
        save_dir = Path(f"{base_output_dir}{x['name']}")
        save_dir.mkdir(parents=True, exist_ok=True)
        for k in mat_keys:
            if k == "name":
                continue
            x[k].resize((512,512)).save(save_dir / f'{k}.png')  
        break
    return str(save_dir.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="create shapes")
    parser.add_argument('--output_dir', default='outputs', help='output directory')
    parser.add_argument('--num_shapes', default=1, type=int, help='number of shapes to create')
    parser.add_argument('--uuid_str', default='', type=str, help='uuid to use for the shape')
    parser.add_argument('--seed', default=0, type=int, help='seed for random number generation')
    parser.add_argument('--smooth_probability', default=1.0, type=float, help='probability of smoothing the height field')
    parser.add_argument('--shape_counts', type=str,
                        help='Comma-separated list of shape counts, e.g., cube:2,ellipsoid:3')
    parser.add_argument('--no_hf', default=False, action='store_true', help='do not use height field')
    parser.add_argument('--texture_folder', type=str, help='Folder containing texture images')
    parser.add_argument('--texture_seed', type=int, default=0, help='Seed for texture selection')
    parser.add_argument('--translation_control', type=float, default=0.4,
                        help='Control the translation of shapes (0.0 to 1.0)')
    parser.add_argument('--rotation_seed', type=int, default=None,
                        help='Seed for controlling the rotation of shapes')
    parser.add_argument('--sub_obj_nums', type=str, default='1,2,3,4,5,6,7,8,9', help='Possible numbers of sub-objects')
    parser.add_argument('--sub_obj_num_poss', type=str, default='1,2,3,7,10,7,3,2,1', help='Corresponding probabilities for numbers of sub-objects')
    parser.add_argument('--cand_shapes', type=str, default='0,1,2', help='Candidate shapes to select from')
    parser.add_argument('--translation_seed', type=int, default=None,
                        help='Seed for controlling the rotation of shapes')
    parser.add_argument('--size_seed', type=int, default=None,
                        help='Seed for controlling the size of shapes')                    


    args = parser.parse_args()


    seed_everything(args.seed)


    args.sub_obj_nums = [int(x) for x in args.sub_obj_nums.split(',')]
    args.sub_obj_num_poss = [int(x) for x in args.sub_obj_num_poss.split(',')]
    args.cand_shapes = [int(x) for x in args.cand_shapes.split(',')]


    if args.shape_counts:
        shape_counts = {}
        shape_counts_list = args.shape_counts.split(',')
        for item in shape_counts_list:
            shape_name, count = item.split(':')
            count = int(count)

            if shape_name.lower() == 'ellipsoid':
                shape_type = 0
            elif shape_name.lower() == 'cube':
                shape_type = 1
            elif shape_name.lower() == 'cylinder':
                shape_type = 2
            else:
                raise ValueError(f"Unknown shape name: {shape_name}")
            shape_counts[shape_type] = count
    else:
        shape_counts = None  # No shape_counts specified; use random selection


    if args.texture_folder:
        texture_images = find_texture_images(args.texture_folder)
        if texture_images:
            texture_rng = np.random.RandomState(args.texture_seed)
        else:
            texture_images = None
            texture_rng = None
    else:
        texture_images = None
        texture_rng = None


    if args.rotation_seed is not None:
        rotation_rng = np.random.RandomState(args.rotation_seed)
    else:
        rotation_rng = np.random
        
    if args.translation_seed is not None:
        translation_rng = np.random.RandomState(args.translation_seed)
    else:
        translation_rng = np.random

    if args.size_seed is not None:
        size_rng = np.random.RandomState(args.size_seed)
    else:
        size_rng = np.random  

    start_time = time.time()
    out_dir = args.output_dir
    num_shapes = args.num_shapes

    output_paths, shapes_parameters = createVarObjShapes(
        out_dir,
        range(num_shapes),
        uuid_str=args.uuid_str,
        shape_counts=shape_counts,
        sub_obj_nums=args.sub_obj_nums,
        sub_obj_num_poss=args.sub_obj_num_poss,
        candShapes=args.cand_shapes,
        bMultiObj=False,
        bPermuteMat=False,
        bScaleMesh=True,
        bMaxDimRange=[0.3, 0.45],
        smooth_probability=args.smooth_probability,
        no_hf=args.no_hf,
        texture_images=texture_images,
        texture_rng=texture_rng,
        translation_control=args.translation_control,  
        rotation_rng=rotation_rng,
        translation_rng=translation_rng,
        size_rng=size_rng
    )
    shape_generation_time = time.time()
    print('Saved shapes to', out_dir)


    print(f'TIME - create_shapes.py: shape_generation_time: {shape_generation_time - start_time:.2f}s')
