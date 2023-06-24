'''
Object File Format (.off)
Object File Format (.off) files are used to represent the geometry of a model by specifying the polygons of the model's surface. The polygons can have any number of vertices.
The .off files in the Princeton Shape Benchmark conform to the following standard. OFF files are all ASCII files beginning with the keyword OFF. The next line states the number of vertices, the number of faces, and the number of edges. The number of edges can be safely ignored.

The vertices are listed with x, y, z coordinates, written one per line. After the list of vertices, the faces are listed, with one face per line. For each face, the number of vertices is specified, followed by indices into the list of vertices. See the examples below.

Note that earlier versions of the model files had faces with -1 indices into the vertex list. That was due to an error in the conversion program and should be corrected now.

OFF numVertices numFaces numEdges
x y z
x y z
... numVertices like above
NVertices v1 v2 v3 ... vN
MVertices v1 v2 v3 ... vM
... numFaces like above

Note that vertices are numbered starting at 0 (not starting at 1), and that numEdges will always be zero.

A simple example for a cube:

OFF
8 6 0
-0.500000 -0.500000 0.500000
0.500000 -0.500000 0.500000
-0.500000 0.500000 0.500000
0.500000 0.500000 0.500000
-0.500000 0.500000 -0.500000
0.500000 0.500000 -0.500000
-0.500000 -0.500000 -0.500000
0.500000 -0.500000 -0.500000
4 0 1 3 2
4 2 3 5 4
4 4 5 7 6
4 6 7 1 0
4 1 7 5 3
4 6 0 2 4
'''

import os
import open3d as o3d
import numpy as np
from utils import *


# OBJECT = "tiny-can"             # debris
# OBJECT = "small-can"            # debris
# OBJECT = "medium-can"           # --> ok
# OBJECT = "food-can"             # --> ok
# OBJECT = "tall-can"             # debris

# OBJECT = "small-cup"            # --> ok         
# OBJECT = "medium-cup"           # --> ok
# OBJECT = "large-cup"            # --> ok

# OBJECT = "tiny-ball"            # --> ok
# OBJECT = "small-ball"           # --> ok
OBJECT = "medium-ball"          # --> ok
# OBJECT = "large-ball"           # debris

DRAW_AXIS = True
VIEW = False

ply_filenames = os.listdir(f'dataset/{OBJECT}')

count = 1
for i in range(len(ply_filenames)):

    ply_filename = ply_filenames[i].split('.')[0]

    save_folder = f"resample_dataset/{OBJECT}"
    save_txt_filename = f"{ply_filename}.txt"
    save_path = f"{save_folder}/{save_txt_filename}"

    file = open(save_path, "w")
    to_be_drawn = []

    # Read point cloud and mesh
    pcd_path = f"dataset/{OBJECT}/{ply_filename}.ply"
    pcd = o3d.io.read_point_cloud(filename=pcd_path)
    mesh = o3d.io.read_triangle_mesh(filename=pcd_path)
    mesh.compute_vertex_normals()
    original_mesh = copy.deepcopy(mesh)

    # Extract information of the point cloud
    vert = np.asarray(pcd.points)
    xs, ys, zs = extract_xyz_coordinates(pcd)
    cx, cy, cz, w, l, h = box_points(xs, ys, zs)

    x_max, x_min = cx + w, cx - w
    y_max, y_min = cy + l, cy - l
    z_max, z_min = cz + h, cz - h

    if VIEW == True:
        print("width original_pcd = ", 2*w)
        print("height original_pcd = ", 2*h)
        print("length original_pcd = ", 2*l)
        print(f"x_min = {x_min}, x_max = {x_max}")
        print(f"y_min = {y_min}, y_max = {y_max}")
        print(f"z_min = {z_min}, z_max = {z_max}")
        print("==========================================")

    # Set the centroid for normalization as object's centroid
    centroid = np.array([cx, cy, cz])

    # Normalize the the point cloud to [-1, +1] on Oxyz coordinate system
    scale_factor, normal_point_set = normalize_pcd(vert, centroid)

    # Make the normalized point cloud and estimate normal vectors
    normalize_pcd1 = o3d.geometry.PointCloud()
    normalize_pcd1.points = o3d.utility.Vector3dVector(normal_point_set)
    normalize_pcd1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Write to file
    for i in range(len(normal_point_set)):
        query_point = normal_point_set[i]
        query_normal = normalize_pcd1.normals[i]
        file.write(f"{query_point[0]},{query_point[1]},{query_point[2]},{query_normal[0]},{query_normal[1]},{query_normal[2]}\n")

    # Extract information of normalized point cloud
    xs, ys, zs = extract_xyz_coordinates(normalize_pcd1)
    cx, cy, cz, w, l, h = box_points(xs, ys, zs)

    x_max, x_min = cx + w, cx - w
    y_max, y_min = cy + l, cy - l
    z_max, z_min = cz + h, cz - h

    if VIEW == True:
        print("width normalize_pcd1 = ", 2*w)
        print("height normalize_pcd1 = ", 2*h)
        print("length normalize_pcd1 = ", 2*l)
        print(f"x_min = {x_min}, x_max = {x_max}")
        print(f"y_min = {y_min}, y_max = {y_max}")
        print(f"z_min = {z_min}, z_max = {z_max}")

    # ENABLE XYZ AXES DRAWING
    if DRAW_AXIS == True:
        x_axis = draw_line([[0, 0, 0], [500, 0, 0]], [1, 0, 0]) # x_axis
        y_axis = draw_line([[0, 0, 0], [0, 500, 0]], [0, 1, 0]) # y_axis
        z_axis = draw_line([[0, 0, 0], [0, 0, 500]], [0, 0, 1]) # z_axis
        to_be_drawn.append(x_axis)
        to_be_drawn.append(y_axis)
        to_be_drawn.append(z_axis)

    # Align the original mesh with the origin coordinates
    shift_x, shift_y, shift_z = move_obj_to_coord_for_mesh(original_mesh, [0, 0, 0])
    original_mesh = translate_pcd(original_mesh, [shift_x, shift_y, shift_z])

    # Append to be drawn objects
    rescale_normalized_pcd1 = scale_pcd(normalize_pcd1, scale_factor)
    to_be_drawn.append(original_mesh)
    to_be_drawn.append(normalize_pcd1)
    to_be_drawn.append(rescale_normalized_pcd1)

    # Print information
    if VIEW == True:
        print("==========================================")
        print("scale_factor = ", scale_factor)
        print("no_points orginal pcd = ", len(np.asarray(pcd.points)))
        print("no_points normalization = ", len(np.asarray(normalize_pcd1.points)))
        print("no_points rescale_normalized_pcd = ", len(np.asarray(rescale_normalized_pcd1.points)))

    file.close()

    print(f">> [{count:4d}/{len(ply_filenames)}] Text file {save_txt_filename} is saved to {save_folder}.")
    count += 1

    # View results
    if VIEW == True:
        o3d.visualization.draw_geometries(to_be_drawn)

    