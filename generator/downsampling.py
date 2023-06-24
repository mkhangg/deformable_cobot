import numpy as np
import open3d as o3d
from utils import *


# OBJECT = "tiny-can"
# OBJECT = "small-can"
# OBJECT = "medium-can"
# OBJECT = "food-can"
# OBJECT = "tall-can"

# OBJECT = "small-cup"
OBJECT = "medium-cup"
# OBJECT = "large-cup"

# OBJECT = "tiny-ball"
# OBJECT = "small-ball"
# OBJECT = "medium-ball"
# OBJECT = "large-ball"


VIEW = False
# VIEW = True
# SAVE_PCD = False
SAVE_PCD = True

object_voxelsize = {
    "tiny-can": 1.9310,
    "small-can": 1.7515,
    "medium-can": 2.0002,
    "food-can": 2.95123,
    "tall-can": 2.15298,

    "small-cup": 3.06682,
    "medium-cup": 3.0726,
    "large-cup": 2.6150,

    "tiny-ball": 1.99892,	
    "small-ball": 2.09832,	
	"medium-ball": 3.2673,	
    "large-ball": 4.3438
}

VOXEL_SIZE = object_voxelsize[OBJECT]
to_be_drawn = []

# Read point cloud
pcd_path = f"scans/scanned_objects/{OBJECT}.ply"
pcd = o3d.io.read_point_cloud(filename=pcd_path)

# Downsample the point cloud
downpcd = downsample_pcd(pcd, VOXEL_SIZE)

# Print information
print(">> Number of points in original pcd = ", len(np.asarray(pcd.points)))
print(">> Number of points in downsampled pcd = ", len(np.asarray(downpcd.points)))

# View point cloud and downsampled point cloud
if VIEW == True:
    to_be_drawn.append(pcd)
    to_be_drawn.append(translate_pcd(downpcd, [0, 150, 0]))
    o3d.visualization.draw_geometries(to_be_drawn)

# Save the downsampled point cloud
if SAVE_PCD == True:

    # Save path, folder, and filename
    save_folder = "scans/downsample_objects"
    save_filename = f"{OBJECT}.ply"
    save_path = f"{save_folder}/{save_filename}"
    
    # Convert the downsampled point cloud the mesh using Poisson
    reconstructed_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(downpcd, depth=9)
    
    o3d.io.write_triangle_mesh(filename=save_path, mesh=reconstructed_mesh, write_ascii=True)
    print(f">> Point cloud file {save_filename} is saved to {save_folder}.")


# ================================ #
FOLDER = "downsample_objects"

pcd_path = f"scans/{FOLDER}/{OBJECT}.ply"
pcd = o3d.io.read_point_cloud(filename=pcd_path)

print(">> Number of points in pcd = ", len(np.asarray(pcd.points)))

# o3d.visualization.draw_geometries([pcd])