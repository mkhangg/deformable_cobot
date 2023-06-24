# https://docs.openvino.ai/latest/notebooks/001-hello-world-with-output.html

import cv2
import matplotlib.pyplot as plt
import numpy as np
from openvino.runtime import Core
from utils import load_yaml
from dataset import ModelNetDataSet
from torch.utils.data import DataLoader

# path = "/home/tuandang/intel/public/mobilenet-v3-small-1.0-224-tf/FP32/mobilenet-v3-small-1.0-224-tf.xml"
conf = load_yaml('config/deform.yaml')
path = f'{conf["checkpoint"]}/best_model_nfeatures{conf["num_features"]}.onnx'
print('path = ', path)
ie = Core()
model = ie.read_model(model=path)
compiled_model = ie.compile_model(model=model, device_name="GPU")
print('compiled_model = ', compiled_model)
output_layer = compiled_model.output(0)
print('output_layer = ', output_layer)


eval_loader =  DataLoader(dataset=ModelNetDataSet(conf, mode="test"),  batch_size= 1) 

i = 0
for points, target in eval_loader:
    # print('points 1 : ', points[0,0,:])
    points = points.numpy() #tensor to numpy
    # print('points 2: ', points[0,0,:])
    # print('points 1 =', points.shape)
    points = points.transpose(0, 2, 1)
    # print('points 2 =', points.shape)
    pred = compiled_model([points])[output_layer]
    print('pred =', pred.shape, pred)
    cls = np.argmax(pred, axis=1)
    print('ret = ', cls, ', target = ', target[0].item())
    i += 1
    if i > 3:
        break
    # exit()