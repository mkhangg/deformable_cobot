import torch
import numpy as np
import open3d as o3d

from utils import *

path = "/mnt/data/workspace/open3d/code/resample_dataset/medium-ball/medium-ball_9546_50_20.txt"

no_points = 50
        
#View
points = np.loadtxt(path, delimiter=",")
points = farthest_point_sample(points, no_points)
points = np.expand_dims(points, axis=0)
points = torch.from_numpy(points)

p = points.numpy()
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(p[0, :,0:3])
o3d.visualization.draw_geometries([pcd])
