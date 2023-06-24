import copy
import numpy as np
import open3d as o3d


NO_VIEWS = 3

# Registration process
to_be_drawn = []
threshold = 0.02
target = o3d.io.read_point_cloud(filename="/mnt/data/workspace/open3d/code/pcd_data/out_1.ply")
to_be_drawn.append(target)

for i in range(2, NO_VIEWS):
    source = o3d.io.read_point_cloud(f"/mnt/data/workspace/open3d/code/pcd_data/out_{i}.ply")
    trans_init = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], 
                           [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))

    transform_mat = reg_p2p.transformation
    source_temp = copy.deepcopy(source)
    source_temp.transform(transform_mat)
    to_be_drawn.append(source_temp)

o3d.visualization.draw_geometries(to_be_drawn)
