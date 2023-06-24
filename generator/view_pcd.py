import open3d as o3d
from utils import *

to_be_drawn = []

EXP_NO = 1
# PCD = "out"
PCD = "out_cropped"

file_path = f"/mnt/data/workspace/open3d/code/captures/exp{EXP_NO}/registered_pcd/{PCD}.ply"
original = o3d.io.read_point_cloud(filename=file_path)
shift_x, shift_y, shift_z = move_obj_to_coord(original, [0, 0, 0])
original = translate_pcd(original, [shift_x, shift_y, shift_z])
to_be_drawn.append(original)

x_axis = draw_line([[0, 0, 0], [500, 0, 0]], [1, 0, 0]) # x_axis
y_axis = draw_line([[0, 0, 0], [0, 500, 0]], [0, 1, 0]) # y_axis
z_axis = draw_line([[0, 0, 0], [0, 0, 500]], [0, 0, 1]) # z_axis
to_be_drawn.append(x_axis)
to_be_drawn.append(y_axis)
to_be_drawn.append(z_axis)

o3d.visualization.draw_geometries(to_be_drawn)
