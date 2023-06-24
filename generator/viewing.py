import open3d as o3d
from utils import *


# OBJECT = "tiny-can"
# OBJECT = "small-can"
# OBJECT = "medium-can"
OBJECT = "food-can"
# OBJECT = "tall-can"

# OBJECT = "small-cup"
# OBJECT = "medium-cup"
# OBJECT = "large-cup"

# OBJECT = "tiny-ball"
# OBJECT = "small-ball"
# OBJECT = "medium-ball"
# OBJECT = "large-ball"

FOLDER = "scanned_objects"
# FOLDER = "downsample_objects"

pcd_path = f"scans/{FOLDER}/{OBJECT}.ply"
# pcd_path = "/mnt/data/workspace/open3d/code/dataset/food-can/food-can_4806_150_25.ply"
pcd = o3d.io.read_point_cloud(filename=pcd_path)

print(">> Number of points in pcd = ", len(np.asarray(pcd.points)))

o3d.visualization.draw_geometries([pcd])
