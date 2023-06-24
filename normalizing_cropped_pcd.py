import open3d as o3d
import numpy as np
from utils import *

DRAW_AXIS = True
VIEW = True
EXP_NO = 1

ply_filename = "out_cropped"

save_folder = f"captures/exp{EXP_NO}/registered_pcd"
save_txt_filename = f"{ply_filename}.txt"
save_path = f"{save_folder}/{save_txt_filename}"

file = open(save_path, "w")
to_be_drawn = []

# Read point cloud
pcd_path = f"{save_folder}/{ply_filename}.ply"
pcd = o3d.io.read_point_cloud(filename=pcd_path)
# mesh = o3d.io.read_triangle_mesh(filename=pcd_path)
# mesh.compute_vertex_normals()
# original_mesh = copy.deepcopy(mesh)

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
# shift_x, shift_y, shift_z = move_obj_to_coord_for_mesh(original_mesh, [0, 0, 0])
# original_mesh = translate_pcd(original_mesh, [shift_x, shift_y, shift_z])

# Append to be drawn objects
rescale_normalized_pcd1 = scale_pcd(normalize_pcd1, scale_factor)
# to_be_drawn.append(original_mesh)
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

# View results
if VIEW == True:
    o3d.visualization.draw_geometries(to_be_drawn)
