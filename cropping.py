import open3d as o3d
import numpy as np
import math
from utils import *

to_be_drawn = []

EXP_NO = 1
PCD = "out"
Y_THRESHOLD = 0.008
registered_pcd_path = f"/mnt/data/workspace/open3d/code/captures/exp{EXP_NO}/registered_pcd/{PCD}.ply"
cropped_pcd_path = f"/mnt/data/workspace/open3d/code/captures/exp{EXP_NO}/registered_pcd/{PCD}_cropped.ply"

print(registered_pcd_path)
registered_pcd = o3d.io.read_point_cloud(filename=registered_pcd_path)
shift_x, shift_y, shift_z = move_obj_to_coord(registered_pcd, [0, 0, 0])
registered_pcd = translate_pcd(registered_pcd, [shift_x, shift_y, shift_z])

removal_indexes = []
registered_pcd_pts = np.asarray(registered_pcd.points)
for i in range(len(registered_pcd_pts)):
    if registered_pcd_pts[i][1] <= Y_THRESHOLD:
        removal_indexes.append(i)

inlier_cloud, removed_cloud = paint_cloud_region(registered_pcd, removal_indexes)
inlier_cloud = rotate_pcd(inlier_cloud, [math.pi/2, 0, 0])
to_be_drawn.append(inlier_cloud)

x_axis = draw_line([[0, 0, 0], [500, 0, 0]], [1, 0, 0]) # x_axis
y_axis = draw_line([[0, 0, 0], [0, 500, 0]], [0, 1, 0]) # y_axis
z_axis = draw_line([[0, 0, 0], [0, 0, 500]], [0, 0, 1]) # z_axis
to_be_drawn.append(x_axis)
to_be_drawn.append(y_axis)
to_be_drawn.append(z_axis)

o3d.io.write_point_cloud(filename=f"{cropped_pcd_path}", pointcloud=inlier_cloud, write_ascii=True)
o3d.visualization.draw_geometries(to_be_drawn)
