from utils import farthest_point_sample, load_yaml
from openvino.runtime import Core
import numpy as np        
import time
import os

conf = load_yaml('config/deform.yaml')
onnx_path = f'{conf["checkpoint"]}/model_nfeatures{conf["num_features"]}_ckpnt_{conf["epochs"]}.onnx'    
npoints = conf['npoints']            
# points = farthest_point_sample(np_points, npoint=npoints)
# print('inlier_cloud = ', points.shape)

catergory_file = os.path.join(f'config/{conf["dataset"]}{conf["num_categories"]}_shape_names.txt')         
categories = [line.rstrip() for line in open(catergory_file)]  
print(categories)

exit()

ie = Core()
onnx_model = ie.read_model(model=onnx_path)
compiled_model = ie.compile_model(model=onnx_model, device_name="CPU")    
output_layer = compiled_model.output(0)

print(f'>> Load ONNX model: {onnx_path}')
# points = points[None,...] 
points = points.transpose(0, 2, 1)
t1 = time.time()
onnx_pred = compiled_model([points])[output_layer]        
t2 = time.time()
onxx_cls = np.argmax(onnx_pred, axis=1)        
print(f'>> The grasping object is {categories[onxx_cls[0]]}')
print(f'>> Inference time: %2.2f ms' %((t2-t1)*1000))        
