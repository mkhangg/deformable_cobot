#! usr/bin/python3

# Import neccessary libraries
import os
import sys
import open3d as o3d


# Check input arguments
try:
    pcd_path = sys.argv[1]
except:
    sys.exit(">> ERROR USAGE: python3 view_pcd_folder.py <folder>")

# Count number of *.ply in folder
count = 0
for path in os.listdir(pcd_path):
    if os.path.isfile(os.path.join(pcd_path, path)):
        count += 1
print('>> Number of clouds:', count)

if count == 0:
    print(">> No point clouds to view!")

# View each *.ply file
for i in range(1, count + 1):
    pcd = o3d.io.read_point_cloud(pcd_path + "out_" + str(i) + ".ply")
    print(">> out_" + str(i) + ".ply: ", pcd)
    o3d.visualization.draw_geometries([pcd])

    # # Voxel downsampling
    # downpcd = pcd.voxel_down_sample(voxel_size=0.05)
    # o3d.visualization.draw_geometries([downpcd])

    # print("Recompute the normal of the downsampled point cloud")
    # downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
    #     radius=0.1, max_nn=30))
    # o3d.visualization.draw_geometries([downpcd])
