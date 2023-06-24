from DNet import DNet
from utils import load_yaml
from torchsummary import summary
from dataset import ModelNetDataSet
from torch.utils.data import DataLoader
from openvino.runtime import Core
import numpy as np        
import time

if __name__ == '__main__':   
    conf = load_yaml('config/deform.yaml')
    # print(conf) 
    onnx_path = f'{conf["checkpoint"]}/model_nfeatures{conf["num_features"]}_ckpnt_{conf["epochs"]}.onnx'    
    eval_loader =  DataLoader(dataset=ModelNetDataSet(conf, mode="test"),  batch_size= 1)         
    nfeatures = 6 if conf['use_normal_vector'] else 3
    npoints = conf['npoints']            

    ie = Core()
    onnx_model = ie.read_model(model=onnx_path)
    compiled_model = ie.compile_model(model=onnx_model, device_name="CPU")    
    output_layer = compiled_model.output(0)
    
    print(f'Load ONNX: {onnx_path}')
    i = 0
    for points, target in eval_loader:                
        points = points.transpose(2, 1)                
        points = points.detach().numpy() 
        t1 = time.time()
        onnx_pred = compiled_model([points])[output_layer]        
        t2 = time.time()        
        onxx_cls = np.argmax(onnx_pred, axis=1)        
        print("%03d" %(i+1), ': onnx cls = ', onxx_cls[0], ', target = ', target[0].item(), '. Inference time: %2.2f ms' %((t2-t1)*1000))        
        i += 1
        if i >= 20:
            break