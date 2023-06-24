#! usr/bin/python3

import os
import cv2
import copy
import math
import time
import datetime
import numpy as np
import open3d as o3d
import pyrealsense2 as rs

from generator.gen_utils import *

import socket
import sys

HOST = "192.168.1.249"  # Standard loopback interface address (localhost)
PORT = int(sys.argv[1])

conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
conn.setblocking(1)
conn.connect((HOST, PORT))
print('>> Connected to ', HOST)
       

NO_VIEWS = 3
MIN_DISTANCE = 0.25
MAX_DISTANCE = 0.70
# ROOT_PATH = "/mnt/data/workspace/deformable/generator"
ROOT_PATH = "generator"

# _, pcd_data_dir, registered_pcd_dir = create_exp_dir(f"{ROOT_PATH}/captures")

EXP_NO = len(os.listdir(f"{ROOT_PATH}/captures"))
# print(">> EXP_NO = ", EXP_NO)

listpc = []

class AppState:

    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = True

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)


state = AppState()

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

config.enable_stream(rs.stream.depth, rs.format.z16, 30)
config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Get stream profile and camera intrinsics
profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
w, h = depth_intrinsics.width, depth_intrinsics.height

# Processing blocks
pc = rs.pointcloud()                                    # point cloud
decimate = rs.decimation_filter()
decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)
colorizer = rs.colorizer()
threshold_filter = rs.threshold_filter()
threshold_filter.set_option(rs.option.min_distance, MIN_DISTANCE)
threshold_filter.set_option(rs.option.max_distance, MAX_DISTANCE)


def mouse_cb(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:
        state.mouse_btns[0] = True

    if event == cv2.EVENT_LBUTTONUP:
        state.mouse_btns[0] = False

    if event == cv2.EVENT_RBUTTONDOWN:
        state.mouse_btns[1] = True

    if event == cv2.EVENT_RBUTTONUP:
        state.mouse_btns[1] = False

    if event == cv2.EVENT_MBUTTONDOWN:
        state.mouse_btns[2] = True

    if event == cv2.EVENT_MBUTTONUP:
        state.mouse_btns[2] = False

    if event == cv2.EVENT_MOUSEMOVE:

        h, w = out.shape[:2]
        dx, dy = x - state.prev_mouse[0], y - state.prev_mouse[1]

        if state.mouse_btns[0]:
            state.yaw += float(dx) / w * 2
            state.pitch -= float(dy) / h * 2

        elif state.mouse_btns[1]:
            dp = np.array((dx / w, dy / h, 0), dtype=np.float32)
            state.translation -= np.dot(state.rotation, dp)

        elif state.mouse_btns[2]:
            dz = math.sqrt(dx**2 + dy**2) * math.copysign(0.01, -dy)
            state.translation[2] += dz
            state.distance -= dz

    if event == cv2.EVENT_MOUSEWHEEL:
        dz = math.copysign(0.1, flags)
        state.translation[2] += dz
        state.distance -= dz

    state.prev_mouse = (x, y)


cv2.namedWindow(state.WIN_NAME, cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow(state.WIN_NAME, w, h)
cv2.setMouseCallback(state.WIN_NAME, mouse_cb)


def project(v):
    """project 3d vector array to 2d"""
    h, w = out.shape[:2]
    view_aspect = float(h)/w

    # ignore divide by zero for invalid depth
    with np.errstate(divide='ignore', invalid='ignore'):
        proj = v[:, :-1] / v[:, -1, np.newaxis] * \
            (w*view_aspect, h) + (w/2.0, h/2.0)

    # near clipping
    znear = 0.03
    proj[v[:, 2] < znear] = np.nan
    return proj


def view(v):
    """apply view transformation on vector array"""
    return np.dot(v - state.pivot, state.rotation) + state.pivot - state.translation


def line3d(out, pt1, pt2, color=(0x80, 0x80, 0x80), thickness=1):
    """draw a 3d line from pt1 to pt2"""
    p0 = project(pt1.reshape(-1, 3))[0]
    p1 = project(pt2.reshape(-1, 3))[0]
    if np.isnan(p0).any() or np.isnan(p1).any():
        return
    p0 = tuple(p0.astype(int))
    p1 = tuple(p1.astype(int))
    rect = (0, 0, out.shape[1], out.shape[0])
    inside, p0, p1 = cv2.clipLine(rect, p0, p1)
    if inside:
        cv2.line(out, p0, p1, color, thickness, cv2.LINE_AA)


def grid(out, pos, rotation=np.eye(3), size=1, n=10, color=(0x80, 0x80, 0x80)):
    """draw a grid on xz plane"""
    pos = np.array(pos)
    s = size / float(n)
    s2 = 0.5 * size
    for i in range(0, n+1):
        x = -s2 + i*s
        line3d(out, view(pos + np.dot((x, 0, -s2), rotation)),
               view(pos + np.dot((x, 0, s2), rotation)), color)
    for i in range(0, n+1):
        z = -s2 + i*s
        line3d(out, view(pos + np.dot((-s2, 0, z), rotation)),
               view(pos + np.dot((s2, 0, z), rotation)), color)


def axes(out, pos, rotation=np.eye(3), size=0.075, thickness=2):
    """draw 3d axes"""
    line3d(out, pos, pos +
           np.dot((0, 0, size), rotation), (0xff, 0, 0), thickness)
    line3d(out, pos, pos +
           np.dot((0, size, 0), rotation), (0, 0xff, 0), thickness)
    line3d(out, pos, pos +
           np.dot((size, 0, 0), rotation), (0, 0, 0xff), thickness)


def frustum(out, intrinsics, color=(0x40, 0x40, 0x40)):
    """draw camera's frustum"""
    orig = view([0, 0, 0])
    w, h = intrinsics.width, intrinsics.height

    for d in range(1, 6, 2):
        def get_point(x, y):
            p = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
            line3d(out, orig, view(p), color)
            return p

        top_left = get_point(0, 0)
        top_right = get_point(w, 0)
        bottom_right = get_point(w, h)
        bottom_left = get_point(0, h)

        line3d(out, view(top_left), view(top_right), color)
        line3d(out, view(top_right), view(bottom_right), color)
        line3d(out, view(bottom_right), view(bottom_left), color)
        line3d(out, view(bottom_left), view(top_left), color)


def pointcloud(out, verts, texcoords, color, painter=True):
    """draw point cloud with optional painter's algorithm"""
    if painter:
        # Painter's algo, sort points from back to front

        # get reverse sorted indices by z (in view-space)
        v = view(verts)
        s = v[:, 2].argsort()[::-1]
        proj = project(v[s])
    else:
        proj = project(view(verts))

    if state.scale:
        proj *= 0.5**state.decimate

    h, w = out.shape[:2]

    # proj now contains 2d image coordinates
    j, i = proj.astype(np.uint32).T

    # create a mask to ignore out-of-bound indices
    im = (i >= 0) & (i < h)
    jm = (j >= 0) & (j < w)
    m = im & jm

    cw, ch = color.shape[:2][::-1]
    if painter:
        # sort texcoord with same indices as above
        # texcoords are [0..1] and relative to top-left pixel corner,
        # multiply by size and add 0.5 to center
        v, u = (texcoords[s] * (cw, ch) + 0.5).astype(np.uint32).T
    else:
        v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T
    # clip texcoords to image
    np.clip(u, 0, ch-1, out=u)
    np.clip(v, 0, cw-1, out=v)

    # perform uv-mapping
    out[i[m], j[m]] = color[u[m], v[m]]

index = -1
out = np.empty((h, w, 3), dtype=np.uint8)


while True:
    if not state.paused:
        #cv2.waitKey(0)
        data = conn.recv(1)
        print('>> Command from Robot... ')

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_frame = decimate.process(depth_frame)
        depth_frame = threshold_filter.process(depth_frame)

        # Grab new intrinsics (may be changed by decimation)
        depth_intrinsics = rs.video_stream_profile(
            depth_frame.profile).get_intrinsics()
        w, h = depth_intrinsics.width, depth_intrinsics.height

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_colormap = np.asanyarray(
            colorizer.colorize(depth_frame).get_data())

        if state.color:
            mapped_frame, color_source = color_frame, color_image
        else:
            mapped_frame, color_source = depth_frame, depth_colormap

        points = pc.calculate(depth_frame)
        pc.map_to(mapped_frame)

        if index == NO_VIEWS:
            break
        
        index = index + 1
        # points.export_to_ply(f'{pcd_data_dir}out_' + str(index) + '.ply', mapped_frame)

        # Write to file
        # outfile = f'{pcd_data_dir}out_' + str(index) + '.ply'
        # print('outfile = ', outfile)
        # points.export_to_ply(outfile, mapped_frame)
        # listpc.append(mapped_frame)

        # Pointcloud data to arrays
        v, t = points.get_vertices(), points.get_texture_coordinates()
        verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
        texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv
        listpc.append(verts)

    # Render
    now = time.time()

    out.fill(0)

    grid(out, (0, 0.5, 1), size=1, n=10)
    frustum(out, depth_intrinsics)
    axes(out, view([0, 0, 0]), state.rotation, size=0.1, thickness=1)

    if not state.scale or out.shape[:2] == (h, w):
        pointcloud(out, verts, texcoords, color_source)
    else:
        tmp = np.zeros((h, w, 3), dtype=np.uint8)
        pointcloud(tmp, verts, texcoords, color_source)
        tmp = cv2.resize(
            tmp, out.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        np.putmask(out, tmp > 0, tmp)

    if any(state.mouse_btns):
        axes(out, view(state.pivot), state.rotation, thickness=4)

    dt = time.time() - now

    cv2.setWindowTitle(
        state.WIN_NAME, "RealSense (%dx%d) %dFPS (%.2fms) %s" %
        (w, h, 1.0/dt, dt*1000, "PAUSED" if state.paused else ""))

    cv2.imshow(state.WIN_NAME, out)
    key = cv2.waitKey(1)

# Stop streaming
pipeline.stop()

# Delete the file that has index 0
# should_be_deleted = f"{pcd_data_dir}out_0.ply"
# if os.path.exists(should_be_deleted):
#     os.remove(should_be_deleted)

# Registration process
# to_be_drawn = []
# target = o3d.io.read_point_cloud(filename=f"{pcd_data_dir}out_1.ply")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(listpc[1])
target = pcd
# to_be_drawn.append(target)

print(">> Start registration...")
# exit()
t1 = time.time()
for i in range(2, NO_VIEWS+1):
    # source = o3d.io.read_point_cloud(f"{pcd_data_dir}out_{i}.ply")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(listpc[i])    
    source = pcd

    threshold = 0.02
    trans_init = np.array([[1.0, 0.0, 0.0, 0.0], 
                            [0.0, 1.0, 0.0, 0.0], 
                            [0.0, 0.0, 1.0, 0.0], 
                            [0.0, 0.0, 0.0, 1.0]])

    print(f'>> Registrating point cloud {i} to the target point cloud...')
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000))

    transform_mat = reg_p2p.transformation
    source_temp = copy.deepcopy(source)
    source_temp.transform(transform_mat)

t2 = time.time()
process_time = round(t2 - t1)
process_time_HMS_format = datetime.timedelta(seconds=process_time)
print(f">> Registration time: {process_time_HMS_format}.")

registered_pcd = source_temp
shift_x, shift_y, shift_z = move_obj_to_coord(registered_pcd, [0, 0, 0])
registered_pcd = translate_pcd(registered_pcd, [shift_x, shift_y, shift_z])
registered_pcd = rotate_pcd(registered_pcd, [math.pi, 0, 0])

THRESHOLD = -0.175
Z_ROTATION = 0                 # degree

removal_indexes = []
registered_pcd_pts = np.asarray(registered_pcd.points)
for i in range(len(registered_pcd_pts)):
    if registered_pcd_pts[i][1] <= THRESHOLD:
        removal_indexes.append(i)

inlier_cloud, removed_cloud = paint_cloud_region(registered_pcd, removal_indexes)
inlier_cloud = rotate_pcd(inlier_cloud, [math.pi/2, 0, 0])
inlier_cloud = rotate_pcd(inlier_cloud, [(Z_ROTATION/180) * math.pi, 0, 0])


np_points = np.asarray(inlier_cloud.points)
# print('inlier_cloud = ', np_points.shape)

from utils import farthest_point_sample, load_yaml
from openvino.runtime import Core
import numpy as np        
import time

conf = load_yaml('config/deform.yaml')
onnx_path = f'{conf["checkpoint"]}/model_nfeatures{conf["num_features"]}_ckpnt_{conf["epochs"]}.onnx'    
npoints = conf['npoints']            
points = farthest_point_sample(np_points, npoint=npoints)
# print('inlier_cloud = ', points.shape)

catergory_file = os.path.join(f'config/{conf["dataset"]}{conf["num_categories"]}_shape_names.txt')        
categories = [line.rstrip() for line in open(catergory_file)]  
print(categories)

ie = Core()
onnx_model = ie.read_model(model=onnx_path)
compiled_model = ie.compile_model(model=onnx_model, device_name="CPU")    
output_layer = compiled_model.output(0)

print(f'>> Load ONNX model: {onnx_path}')
points = points[None,...] 
points = points.transpose(0, 2, 1)
t1 = time.time()
onnx_pred = compiled_model([points])[output_layer]        
t2 = time.time()
onxx_cls = np.argmax(onnx_pred, axis=1)        
print(f'>> The grasping object is {categories[onxx_cls[0]]}')
print(f'>> Inference time: %2.2f ms' %((t2-t1)*1000))        
