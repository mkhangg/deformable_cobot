import os
import copy
import random
import math as m
import numpy as np
import open3d as o3d

from skspatial.objects import Line, Sphere

# ADD NOISES TO POINT CLOUDS            (Done)
# ADD DEFORMATION TO POINT CLOUDS       (Done)
# HIGHTLIGHT HIGHEST COORDINATES        (--)
# NORMAL VECTOR that goes down          (Done)
# BBOX ALIGNMENT                        (Done)
# IDENTIFY ORIENTATION OF HANDLE        (Done)


def create_pcd_path(filename):
    data_folder = "scans"
    obj_file_path = data_folder + "/" + filename + "_test.ply"

    return obj_file_path


def extract_xyz_coordinates(object_pcd):
    
    pts = np.asarray(object_pcd.points)
    x_coords, y_coords, z_coords = pts[:, 0], pts[:, 1], pts[:, 2]

    return x_coords, y_coords, z_coords


def box_points(coords_x, coords_y, coords_z):

    x_max, x_min = max(coords_x), min(coords_x)
    y_max, y_min = max(coords_y), min(coords_y)
    z_max, z_min = max(coords_z), min(coords_z)

    x_center = (x_max + x_min)/2
    y_center = (y_max + y_min)/2
    z_center = (z_max + z_min)/2

    width = x_max - x_center
    length = y_max - y_center
    height = z_max - z_center

    return x_center, y_center, z_center, width, length, height


def eight_points_bbox(center_x, center_y, center_z, width, length, height):

    # 4 points in the lower plane
    pt0 = [center_x-width, center_y+length, center_z-height]    
    pt1 = [center_x+width, center_y+length, center_z-height]
    pt2 = [center_x-width, center_y-length, center_z-height]
    pt3 = [center_x+width, center_y-length, center_z-height]

    # 4 points in the upper plane
    pt4 = [center_x-width, center_y+length, center_z+height]
    pt5 = [center_x+width, center_y+length, center_z+height]
    pt6 = [center_x-width, center_y-length, center_z+height]
    pt7 = [center_x+width, center_y-length, center_z+height]

    bbox_pts = [pt0, pt1, pt2, pt3, pt4, pt5, pt6, pt7]

    return bbox_pts


def eight_points_graspible_box(center_x, center_y, center_z, width, length, height, region):

    # 4 points in the lower plane
    pt0 = [center_x-width, center_y+length, center_z-region*height]    
    pt1 = [center_x+width, center_y+length, center_z-region*height]
    pt2 = [center_x-width, center_y-length, center_z-region*height]
    pt3 = [center_x+width, center_y-length, center_z-region*height]

    # 4 points in the upper plane
    pt4 = [center_x-width, center_y+length, center_z+region*height]
    pt5 = [center_x+width, center_y+length, center_z+region*height]
    pt6 = [center_x-width, center_y-length, center_z+region*height]
    pt7 = [center_x+width, center_y-length, center_z+region*height]

    bbox_pts = [pt0, pt1, pt2, pt3, pt4, pt5, pt6, pt7]

    return bbox_pts


def eight_points_slicing_box(center_x, center_y, width, length, lower_z, upper_z):

    # 4 points in the lower plane
    pt0 = [center_x-width, center_y+length, lower_z]    
    pt1 = [center_x+width, center_y+length, lower_z]
    pt2 = [center_x-width, center_y-length, lower_z]
    pt3 = [center_x+width, center_y-length, lower_z]

    # 4 points in the upper plane
    pt4 = [center_x-width, center_y+length, upper_z]
    pt5 = [center_x+width, center_y+length, upper_z]
    pt6 = [center_x-width, center_y-length, upper_z]
    pt7 = [center_x+width, center_y-length, upper_z]

    bbox_pts = [pt0, pt1, pt2, pt3, pt4, pt5, pt6, pt7]

    return bbox_pts


def eight_points_oriented_bbox(pcd):
    oriented_bbox = pcd.get_oriented_bounding_box()
    bbox_pts = np.asarray(oriented_bbox.get_box_points())
    bbox_pts = bbox_pts.tolist()

    return oriented_bbox, bbox_pts


def bbox_edges(eight_points, color):

    # Define which point connect to which point
    lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],
            [0, 4], [1, 5], [2, 6], [3, 7]]
    
    # Ref: http://www.open3d.org/docs/0.7.0/tutorial/Basic/visualization.html
    colors = [color for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(eight_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def oriented_bbox_edges(eight_points, oriented_bbox, color):
    pair_list = []
    lineset = o3d.geometry.LineSet.create_from_oriented_bounding_box(oriented_bbox)

    for i in range(12):
        
        pair = []
        line = lineset.get_line_coordinate(i)
        
        for j in range(len(eight_points)):
            if (line[0] == eight_points[j]).all() or (line[1] == eight_points[j]).all():
                pair.append(j)

        pair_list.append(pair)
    
    colors = [color for i in range(len(pair_list))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(eight_points)
    line_set.lines = o3d.utility.Vector2iVector(pair_list)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set, pair_list


def oriented_bbox_edges1(eight_points, pair_list, color):
    colors = [color for i in range(len(pair_list))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(eight_points)
    line_set.lines = o3d.utility.Vector2iVector(pair_list)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def move_obj_to_coord(obj, coords):
    x_obj, y_obj, z_obj = extract_xyz_coordinates(obj)
    xc_obj, yc_obj, zc_obj, _, _, _ = box_points(x_obj, y_obj, z_obj)

    x_shift, y_shift, z_shift = coords[0]-xc_obj, coords[1]-yc_obj, coords[2]-zc_obj

    return x_shift, y_shift, z_shift


def move_obj_to_coord_for_mesh(obj, coords):
    vert = np.asarray(obj.vertices)
    x_obj, y_obj, z_obj = vert[:, 0], vert[:, 1], vert[:, 2]
    xc_obj, yc_obj, zc_obj, _, _, _ = box_points(x_obj, y_obj, z_obj)

    x_shift, y_shift, z_shift = coords[0]-xc_obj, coords[1]-yc_obj, coords[2]-zc_obj

    return x_shift, y_shift, z_shift


def draw_line(arrow_points, color):

    # color = [1, 0, 0]
    lines = [[0, 1]]

    colors = [color for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(arrow_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def show_handle_point(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    inlier_cloud.paint_uniform_color([1, 0, 0])
    outlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])

    inlier_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    x_handle, y_handle, z_handle = extract_xyz_coordinates(inlier_cloud)
    shift = 10
    arrow_pts = [[x_handle, y_handle, z_handle], [x_handle+shift, y_handle+shift, z_handle+shift]]
    grav_line = draw_line(arrow_pts, [1, 0, 0])

    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud, grav_line])


def get_handle_indexes(pcd, g):
    no_points = len(np.asarray(pcd.points))
    xs, ys, zs = extract_xyz_coordinates(pcd)
    cx, cy, cz, w, l, h = box_points(xs, ys, zs)

    x_max, x_min = cx + w, cx - w
    y_max, y_min = cy + l, cy - l
    z_max, z_min = cz + g * h, cz - g * h

    inliers_idx = []
    for i in range(no_points):
        if (xs[i] <= x_max and xs[i] >= x_min) and \
            (ys[i] <= y_max and ys[i] >= y_min) and \
            (zs[i] <= z_max and zs[i] >= z_min):
            inliers_idx.append(i)
            
    return inliers_idx


def get_handle_indexes_in_slice(pcd, handles_idx, lower_z, upper_z):
    no_points = len(np.asarray(pcd.points))
    xs, ys, zs = extract_xyz_coordinates(pcd)
    cx, cy, _, w, l, _ = box_points(xs, ys, zs)

    x_max, x_min = cx + w, cx - w
    y_max, y_min = cy + l, cy - l
    z_max, z_min = upper_z, lower_z

    slice_handles_idx = []
    for i in handles_idx:
        if (xs[i] < x_max and xs[i] >= x_min) and \
            (ys[i] < y_max and ys[i] >= y_min) and \
            (zs[i] < z_max and zs[i] >= z_min):
            slice_handles_idx.append(i)

    slice_inliers_idx = []
    for j in range(no_points):
        if (xs[j] < x_max and xs[j] >= x_min) and \
            (ys[j] < y_max and ys[j] >= y_min) and \
            (zs[j] < z_max and zs[j] >= z_min):
            slice_inliers_idx.append(j)
    
    return slice_handles_idx, slice_inliers_idx


def sampling_handle_indexes(inliers_idx, sampling_ratio):
    no_handle_points = int(sampling_ratio * len(inliers_idx))
    handle_samples_idx = np.random.choice(inliers_idx, no_handle_points, replace=False)

    return handle_samples_idx


def create_exp_dir(exp_dir):

    no_dirs = len(os.listdir(f'{exp_dir}'))
    exp_path = f"{exp_dir}/exp{no_dirs + 1}/"
    pcd_data_dir = exp_path + 'pcd_data/'
    registered_pcd_dir = exp_path + 'registered_pcd/'
    directories = [pcd_data_dir, registered_pcd_dir]

    try:
        for dir in directories:
            if not os.path.exists(dir):
                os.makedirs(dir)
        return exp_path, pcd_data_dir, registered_pcd_dir
    
    except Exception as err:
        print("Fail to create directory!")
        exit(-1) 


def farthest_point_sample(point, npoint):
    N, _ = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]

    return point


def resampling_pcd(pcd_path, no_points):
    pcd = o3d.io.read_point_cloud(filename=pcd_path)
    pcd_pts = np.asarray(pcd.points)
    pcd_pts = farthest_point_sample(pcd_pts, no_points)
    resampled_pcd = o3d.geometry.PointCloud()
    resampled_pcd.points = o3d.utility.Vector3dVector(pcd_pts)
    
    return pcd_pts, resampled_pcd


def paint_cloud_region(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    inlier_cloud.paint_uniform_color([1, 0, 0])
    outlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])

    return outlier_cloud, inlier_cloud


def colorize_inlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    inlier_cloud.paint_uniform_color([1, 0, 0])
    outlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])

    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


def find_opposite_point(original_mesh, inliers_idx, handle_idx):
    dist_list = []
    vertices = np.asarray(original_mesh.vertices)
    principle_pt = np.array(vertices[handle_idx])
    for id in inliers_idx:
        query_pt = np.array(vertices[id])
        dist = np.linalg.norm(principle_pt - query_pt)
        dist_list.append(dist)
    
    query_id = dist_list.index(max(dist_list))
    opposite_idx = inliers_idx[query_id]
    
    return opposite_idx


def find_deform_point(handle, midpoint, radius):
    # Ref: https://scikit-spatial.readthedocs.io/en/stable/objects/sphere.html
    # Ref: https://scikit-spatial.readthedocs.io/en/stable/objects/line.html
    
    sphere = Sphere(handle, radius)
    line = Line.from_points(handle, midpoint)

    point1, point2 = sphere.intersect_line(line)
    point1, point2 = np.array(point1), np.array(point2)

    dist1 = np.linalg.norm(midpoint - point1)
    dist2 = np.linalg.norm(midpoint - point2)

    if dist1 < dist2:
        deform_point = point1
    else:
        deform_point = point2

    return deform_point


def draw_bbox(scanned_objects, pointcloud):
    bboxes = []
    to_be_drawn = [pointcloud]
    
    for i in range(len(scanned_objects)):
        xs, ys, zs = extract_xyz_coordinates(scanned_objects[i])
        cx, cy, cz, w, l, h = box_points(xs, ys, zs)
        points = eight_points_bbox(cx, cy, cz, w, l, h)

        color = [round(random.uniform(0, 1), 1), round(random.uniform(0, 1), 1), round(random.uniform(0, 1), 1)]
        box_edges = bbox_edges(points, color)

        to_be_drawn.append(box_edges)
        bboxes.append([cx, cy, cz, 2*w, 2*l, 2*h])

    o3d.visualization.draw_geometries(to_be_drawn)

    return bboxes


def translate_pcd(pcd, direcs):
    trans_pcd = copy.deepcopy(pcd).translate((direcs[0], \
                                            direcs[1], \
                                            direcs[2]))
    return trans_pcd


def scale_pcd(pcd, projection):
    scaled_pcd = copy.deepcopy(pcd)
    scaled_pcd = scaled_pcd.scale(projection, center=(0, 0, 0))

    return scaled_pcd


def rotate_pcd(pcd, rotations):
    rot_matrix = pcd.get_rotation_matrix_from_xyz((rotations[0], rotations[1], rotations[2]))
    rotated_pcd = pcd.rotate(rot_matrix, center=(0, 0, 0))
    
    return rotated_pcd


def downsample_pcd(pcd, size_voxel):
    downpcd = copy.deepcopy(pcd)
    downsample_pcd = downpcd.voxel_down_sample(voxel_size=size_voxel)
    
    return downsample_pcd


def normalize_pcd(points, centroid):
	# centroid = np.array([0, 0, 0])
	points -= centroid
	furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
	points /= furthest_distance

	return furthest_distance, points


def paint_color(pcd):
    colored_pcd = pcd.paint_uniform_color([round(random.uniform(0, 1), 1), \
                                            round(random.uniform(0, 1), 1), \
                                            round(random.uniform(0, 1), 1)])
    return colored_pcd


def add_gaussian_noises(pcd, mu, sigma):
    noised_pcd = copy.deepcopy(pcd)

    pts = np.asarray(noised_pcd.points)
    pts += np.random.normal(mu, sigma, size=pts.shape)
    noised_pcd.points = o3d.utility.Vector3dVector(pts)

    return noised_pcd


def Rx(theta, useDegree=True):
    if useDegree == True:
        theta = (m.pi/180)*theta

    return np.matrix([[ 1, 0           , 0           ],
                    [ 0, m.cos(theta),-m.sin(theta)],
                    [ 0, m.sin(theta), m.cos(theta)]])
  

def Ry(theta, useDegree=True):
    if useDegree == True:
        theta = (m.pi/180)*theta

    return np.matrix([[ m.cos(theta), 0, m.sin(theta)],
                    [ 0           , 1, 0           ],
                    [-m.sin(theta), 0, m.cos(theta)]])


def Rz(theta, useDegree=True):
    if useDegree == True:
        theta = (m.pi/180)*theta

    return np.matrix([[ m.cos(theta), -m.sin(theta), 0 ],
                    [ m.sin(theta), m.cos(theta) , 0 ],
                    [ 0           , 0            , 1 ]])


def rotate_bbox(org_box_pts, rot_angles):
    
    # rotation = np.dot(np.dot(Rx(rot_angles[0]), Ry(rot_angles[1])), Rz(rot_angles[2]))
    # rotation = np.dot(Rx(rot_angles[0]), Ry(rot_angles[1]))
    # rotated_box_pts = np.dot(np.array(org_box_pts), Rx(rot_angles[0]))

    rotated_box_pts = np.dot(np.dot(np.dot(np.array(org_box_pts), Rx(rot_angles[0])), Ry(rot_angles[1])), Rz(rot_angles[2]))

    rotated_box_pts = rotated_box_pts.tolist()

    return rotated_box_pts


def shift_bbox(bbox_pts, shift):
    xs = np.array(bbox_pts)[:, 0] + shift[0]
    ys = np.array(bbox_pts)[:, 1] + shift[1]
    zs = np.array(bbox_pts)[:, 2] + shift[2]

    shift_points = np.transpose(np.concatenate([xs, ys, zs], 0).reshape(3, 8))
    shift_points = shift_points.tolist()

    return shift_points


def take_y_min(temp_points):

    y_min = []

    for i in range(len(temp_points)):
        bbox_pts = temp_points[i]
        ys = np.array(bbox_pts)[:, 1]
        y_min.append(min(ys))

    return y_min


def draw_best_fit_bbox(scanned_objects, draw_axis=False):

    bboxes = []
    to_be_drawn = []
    rot_angles = [np.pi/3, np.pi/4, np.pi/5]

    for i in range(len(scanned_objects)-1):

        rotated_scanned_object = rotate_pcd(scanned_objects[i], rot_angles)
        rotated_scanned_object = paint_color(rotated_scanned_object)
        to_be_drawn.append(rotated_scanned_object)

        oriented_bbox = rotated_scanned_object.get_oriented_bounding_box()
        
        color = [round(random.uniform(0, 1), 1), round(random.uniform(0, 1), 1), round(random.uniform(0, 1), 1)]
        oriented_bbox.color = color

        oriented_pts = oriented_bbox.get_box_points()
        to_be_drawn.append(oriented_bbox)
        bboxes.append([oriented_pts])

    scene_pcd = scanned_objects[len(scanned_objects)-1]
    to_be_drawn.append(scene_pcd)

    xs, ys, zs = extract_xyz_coordinates(scene_pcd)
    cx, cy, cz, w, l, h = box_points(xs, ys, zs)
    scene_points = eight_points_bbox(cx, cy, cz, w, l, h)
    color = [round(random.uniform(0, 1), 1), round(random.uniform(0, 1), 1), round(random.uniform(0, 1), 1)]
    box_edges = bbox_edges(scene_points, color)
    to_be_drawn.append(box_edges)
    # bboxes.append([cx, cy, cz, 2*w, 2*l, 2*h])

    # ENABLE XYZ AXES DRAWING
    if draw_axis == True:
        x_axis = draw_line([[0, 0, 0], [500, 0, 0]], [1, 0, 0]) # x_axis
        y_axis = draw_line([[0, 0, 0], [0, 500, 0]], [0, 1, 0]) # y_axis
        z_axis = draw_line([[0, 0, 0], [0, 0, 500]], [0, 0, 1]) # z_axis
        to_be_drawn.append(x_axis)
        to_be_drawn.append(y_axis)
        to_be_drawn.append(z_axis)

    o3d.visualization.draw_geometries(to_be_drawn)

    return bboxes


def check_all_above_floor(pos_objs, floor):
    check_all = 0

    for i in range(len(pos_objs)):
        if pos_objs[i] > floor:
            check_all = 1
            break

    return check_all


def dropping_with_bboxes(scanned_objects, draw_axis=False):

    bboxes = []
    to_be_drawn = []
    pts_aa_bboxes = []
    pts_oriented_bboxes = []
    rot_angles = [np.pi/3, np.pi/4, np.pi/5]

    for i in range(len(scanned_objects)-1):

        # Rotate object "i"
        rotated_scanned_object = rotate_pcd(scanned_objects[i], rot_angles)
        rotated_scanned_object = paint_color(rotated_scanned_object)
        to_be_drawn.append(rotated_scanned_object)

        # Get oriented bounding box of object "i"
        oriented_bbox, oriented_pts = eight_points_oriented_bbox(rotated_scanned_object)
        color = [0, 0, 1]
        oriented_box_edges, pair_list = oriented_bbox_edges(oriented_pts, oriented_bbox, color)
        pts_oriented_bboxes.append(oriented_pts)
        to_be_drawn.append(oriented_box_edges)

        # Get axis-aligned bounding box of object "i"
        xs, ys, zs = extract_xyz_coordinates(rotated_scanned_object)
        cx, cy, cz, w, l, h = box_points(xs, ys, zs)
        points = eight_points_bbox(cx, cy, cz, w, l, h)
        color = [1, 0, 0]
        box_edges = bbox_edges(points, color)
        pts_aa_bboxes.append(points)
        to_be_drawn.append(box_edges)

    # Get axis-aligned bounding box of the scene 
    scene_pcd = scanned_objects[len(scanned_objects)-1]
    xs, ys, zs = extract_xyz_coordinates(scene_pcd)
    cx, cy, cz, w, l, h = box_points(xs, ys, zs)
    scene_points = eight_points_bbox(cx, cy, cz, w, l, h)
    color = [0, 1, 0]
    box_edges = bbox_edges(scene_points, color)
    to_be_drawn.append(scene_pcd)
    to_be_drawn.append(box_edges)
    # bboxes.append([cx, cy, cz, 2*w, 2*l, 2*h])

    # ENABLE XYZ AXES DRAWING
    if draw_axis == True:
        x_axis = draw_line([[0, 0, 0], [500, 0, 0]], [1, 0, 0]) # x_axis
        y_axis = draw_line([[0, 0, 0], [0, 500, 0]], [0, 1, 0]) # y_axis
        z_axis = draw_line([[0, 0, 0], [0, 0, 500]], [0, 0, 1]) # z_axis
        to_be_drawn.append(x_axis)
        to_be_drawn.append(y_axis)
        to_be_drawn.append(z_axis)
    ##########################

    temp_points = pts_aa_bboxes
    temp_oriented_pts = pts_oriented_bboxes
    y_floor = take_y_min([scene_points])
    y_min = take_y_min(temp_points)


    while check_all_above_floor(y_min, y_floor):
        o3d.visualization.draw_geometries(to_be_drawn)

        color_blue = [0, 0, 1]
        color_red = [1, 0, 0]
        for i in range(len(scanned_objects)-1):
            if y_min[i] > y_floor:
                to_be_drawn[(i+1)*3-3] = translate_pcd(to_be_drawn[(i+1)*3-3], [0, -10, 0])
                temp_points[i] = shift_bbox(temp_points[i], [0, -10, 0])
                temp_oriented_pts[i] = shift_bbox(temp_oriented_pts[i], [0, -10, 0])

                # Oriented bounding box edges are unordered --> SOLVED
                to_be_drawn[(i+1)*3-2] = oriented_bbox_edges1(temp_oriented_pts[i], pair_list, color_blue)
                to_be_drawn[(i+1)*3-1] = bbox_edges(temp_points[i], color_red)
                y_min = take_y_min(temp_points)

        print("y_min = ", y_min)
        print("all_above_floor = ", check_all_above_floor(y_min, y_floor))

    # final_pcd = []
    # for i in range(len(scanned_objects)):
    #     final_pcd.append(to_be_drawn[i*3])
        
    # print("final_pcd = ", final_pcd)
    # o3d.io.write_point_cloud("final_pcd.pcd", final_pcd)

    return bboxes
