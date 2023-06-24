import os
import time
import shutil
import datetime
import numpy as np
import open3d as o3d

from utils import *
from pytorch3d.io import load_ply
from pytorch3d.structures import Meshes
from pytorch3d.loss import mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency

import warnings
warnings.filterwarnings("ignore")


MODE = "run"
# MODE = "test"

mode_view = {
	"run": False,
	"test": False
}

VIEW = mode_view[MODE]

# DATASET_FOLDER = "dataset"
# DATASET_FOLDER = "demo_dataset"
DATASET_FOLDER = "smooth_dataset"

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

SLICE_THICKNESS = 1
SAMPLING_RATIO = 0.035
GRASP_REGION_RATIO = 0.9

# ITERATIONS = [50, 100, 150]
RADII = [4, 6, 8, 10, 12, 14]
ITERATIONS = [100]
ROOT_PATH = "/mnt/data/workspace/open3d/code"

to_be_drawn = []
start_time = time.time()
log_filename = "log_loss.txt"

# Create folder to save deformed meshes
deform_folder_by_class = DATASET_FOLDER + "/" + OBJECT      
if os.path.exists(deform_folder_by_class) and os.path.isdir(deform_folder_by_class):
    shutil.rmtree(deform_folder_by_class)
os.mkdir(deform_folder_by_class, 0o666)


# Open log file for deformation loss
log_file = open(f"{ROOT_PATH}/{DATASET_FOLDER}/{OBJECT}/{log_filename}", "w")

pcd_path = f"scans/downsample_objects/{OBJECT}.ply"
pcd = o3d.io.read_point_cloud(filename=pcd_path)
mesh = o3d.io.read_triangle_mesh(filename=pcd_path)
mesh.compute_vertex_normals()
original_mesh = copy.deepcopy(mesh)
original_mesh.paint_uniform_color([0.8, 0.8, 0.8])
if VIEW == True:
    to_be_drawn.append(original_mesh)

# # Calculate smoothess of the original mesh
# vertices_original_mesh, faces_original_mesh = load_ply(pcd_path)    
# original_mesh_torch = Meshes(verts=[vertices_original_mesh], faces=[faces_original_mesh])
# smoothness = mesh_laplacian_smoothing(original_mesh_torch, method="uniform")
# print(f">> Smoothness of {OBJECT} \t\t= {smoothness:.4f}")

# Get indexes of points that lie in the graspable region of point cloud
# Refer: identifying_graspable.py (for visualization)
graspable_idx = get_handle_indexes(pcd, GRASP_REGION_RATIO)

# Uniformly sample the indexes to generalize graspable region 
# Refer: sampling_handles.py (for visualization)
handle_samples_idx = sampling_handle_indexes(graspable_idx, SAMPLING_RATIO)

# Extract information to calculate parameters for slicing
xs, ys, zs = extract_xyz_coordinates(pcd)
cx, cy, cz, w, l, h = box_points(xs, ys, zs)
z_max, z_min = cz + GRASP_REGION_RATIO * h, cz - GRASP_REGION_RATIO * h

# Compute slice the object along z-coordinates (along the object)
# Refer: slicing_pcd.py (for visualization)
zs_slices = np.arange(z_min, z_max, SLICE_THICKNESS)
zs_slices[-1] = z_max

# Some parameters for monitoring
count = 0
no_detected_handles, no_detected_inliers = 0, 0

# Iterate through every slice along the object
for i in range(len(zs_slices) - 1):
# for i in range(50, 52):
# for i in range(45, 55):

    to_be_deformed_pcd = copy.deepcopy(pcd)
    to_be_deformed_mesh = copy.deepcopy(original_mesh)

    # Get handles points that lie in the current slice
    z_lower, z_upper = zs_slices[i], zs_slices[i+1]
    slice_handles_idx, slice_inliers_idx = get_handle_indexes_in_slice(to_be_deformed_pcd, handle_samples_idx, z_lower, z_upper)
    no_detected_handles += len(slice_handles_idx)
    no_detected_inliers += len(slice_inliers_idx)
    print(f"Slice {i+1:4d}/{len(zs_slices) - 1} of [{z_lower:8.4f}, {z_upper:8.4f}] has {len(slice_handles_idx):3d}/{len(slice_inliers_idx):3d} points.")

    # Check if there is any handle (to deform) in the current slice
    # If number of handles in slice is 0, skip it
    if len(slice_handles_idx) <= 0:
        print(">> No handle points to perform deforming.")
        print("===============================================================================")
        continue
        
    for handle_in_slice_idx in slice_handles_idx:
        for iter in ITERATIONS:                
            opposite_idx = find_opposite_point(to_be_deformed_mesh, slice_inliers_idx, handle_in_slice_idx)                
            vertices = np.asarray(to_be_deformed_mesh.vertices)
            mid_point = np.array([ \
                    (vertices[handle_in_slice_idx][0] + vertices[opposite_idx][0])/2, \
                    (vertices[handle_in_slice_idx][1] + vertices[opposite_idx][1])/2, \
                    (vertices[handle_in_slice_idx][2] + vertices[opposite_idx][2])/2])

            # Will deform 2 times on the same object 
            # 1st time at the handle point
            # 2nd time at the opposite point
            handle_list = [handle_in_slice_idx, opposite_idx]
            max_radius = np.linalg.norm(vertices[handle_in_slice_idx] - vertices[opposite_idx])/2.2

            for radius in RADII:

                to_be_deformed_mesh_by_radius = copy.deepcopy(to_be_deformed_mesh)
                
                # Check for deformation self-intersection
                if radius > max_radius:
                    radius = int(max_radius)

                # Check if the radius is still greater than 0
                if radius <= 0:
                    print(f">> Radius equals 0. Will not deform with this configuration.")
                    print("===============================================================================")
                    continue
                    
                # Check if the deformation configuration is exist. If so, skip this configuration
                if os.path.isfile(f"{ROOT_PATH}/{DATASET_FOLDER}/{OBJECT}/{OBJECT}_{handle_in_slice_idx}_{iter}_{radius}.ply") == True:
                    print(f">> {OBJECT}_{handle_in_slice_idx}_{iter}_{radius}.ply is already exist.")
                    print("===============================================================================")
                    continue     
                    
                print(f">> [{i+1:4d}/{len(zs_slices) - 1}] Start deforming mesh with handle of {handle_in_slice_idx}, iterations of {iter}, and radius of {radius}.")

                # Treat of handle point (in two-element list of handle and opposite points)
                # 1st time at the handle point
                # 2nd time at the opposite point
                for handle in handle_list:
                    static_ids = [idx for idx in np.where(vertices[:, 1] < -30)[0]]
                    static_pos = []
                    for id in static_ids:
                        static_pos.append(vertices[id])
                    
                    handle_ids = [handle]
                    handle_pos = [find_deform_point(vertices[handle], mid_point, radius)]

                    constraint_ids = o3d.utility.IntVector(static_ids + handle_ids)
                    constraint_pos = o3d.utility.Vector3dVector(static_pos + handle_pos)

                    to_be_deformed_mesh_by_radius = to_be_deformed_mesh_by_radius.deform_as_rigid_as_possible( \
                        constraint_ids, constraint_pos, max_iter=iter, smoothed_alpha=0.01)
                    # <DeformAsRigidAsPossibleEnergy.Smoothed: 1>
                    # <DeformAsRigidAsPossibleEnergy.Spokes: 0>

                print(f">> [{i+1:4d}/{len(zs_slices) - 1}] Done deforming mesh with handle of {handle_in_slice_idx}, iterations of {iter}, and radius of {radius}.")

                if VIEW == True:
                    count += 1
                    deform_mesh = translate_pcd(to_be_deformed_mesh_by_radius, [0, 100*count, 0])
                    deform_mesh.compute_vertex_normals()
                    to_be_drawn.append(deform_mesh)
                else:
                    deform_mesh = translate_pcd(to_be_deformed_mesh_by_radius, [0, 0, 0])
                    deform_mesh.compute_vertex_normals()

                # Save the deformed mesh into folder
                save_folder = f"{DATASET_FOLDER}/{OBJECT}/"
                save_filename = f"{OBJECT}_{handle_in_slice_idx}_{iter}_{radius}.ply"
                deform_path = f"{save_folder}/{save_filename}"
                full_deform_path = f"{ROOT_PATH}/{deform_path}"

                o3d.io.write_triangle_mesh(filename=f"{deform_path}", mesh=deform_mesh, write_ascii=True)
                print(f">> [{i+1:4d}/{len(zs_slices) - 1}] Mesh file {save_filename} is saved to folder {save_folder}.")

                # Load original point cloud and deformed point cloud
                pcd_original = o3d.io.read_point_cloud(filename=pcd_path)
                pcd_deform = o3d.io.read_point_cloud(filename=deform_path)
                
                # Calculate Chamfer distance
                chamfer_dist = pcd_original.compute_point_cloud_distance(pcd_deform)
                chamfer_dist = np.asarray(chamfer_dist)
                chamfer_dist = sum(chamfer_dist)/len(chamfer_dist)

                # Calculate mesh edge loss, mesh normal loss, and laplacian smoothing loss
                vertices, faces = load_ply(full_deform_path)    
                deform_mesh = Meshes(verts=[vertices], faces=[faces])
                loss_edge = mesh_edge_loss(deform_mesh)
                loss_normal = mesh_normal_consistency(deform_mesh)
                loss_laplacian = mesh_laplacian_smoothing(deform_mesh, method="uniform")
                
                # Calculate weighted sum loss
                weight_vec = np.array([1.0, 1.0, 0.01, 0.1])        # weight_vec = [w_chamfer, w_edge, w_normal, w_laplacian]
                loss_vec = np.array([chamfer_dist, loss_edge, loss_normal, loss_laplacian])
                deform_loss = np.dot(weight_vec, loss_vec)

                # Print deform losses
                print(f">> chamfer distance \t\t = {chamfer_dist:.8f}")
                print(f">> mesh edge loss \t\t = {loss_edge.item():.8f}")
                print(f">> normal consistency loss \t = {loss_normal.item():.8f}")
                print(f">> laplacian smooth loss \t = {loss_laplacian.item():.8f}")
                print(f">> total deform loss \t\t = {deform_loss:.8f}")

                log_file.write(f"{radius},{chamfer_dist:.8f},{loss_edge.item():.8f},{loss_normal.item():.8f},{loss_laplacian.item():.8f},{deform_loss:.8f}\n")

                print(f">> {save_filename} is saved to {save_folder}.")
                print("===============================================================================")


print(f">> Number of detected handles points = {no_detected_handles}/{len(handle_samples_idx)}")
print(f">> Number of detected inliers points = {no_detected_inliers}/{len(graspable_idx)}")
print(f">> Number of generated deformations = {len(os.listdir(f'{DATASET_FOLDER}/{OBJECT}'))-1}")
print(f">> Log file {log_filename} is saved to {DATASET_FOLDER}/{OBJECT}/.")

log_file.close()
end_time = time.time()
process_time = round(end_time - start_time)
process_time_HMS_format = datetime.timedelta(seconds=process_time)
print(f">> Process time: {process_time_HMS_format}.")

if VIEW == True:
    o3d.visualization.draw_geometries(to_be_drawn)
