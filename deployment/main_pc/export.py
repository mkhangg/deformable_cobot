from DNet import DNet
from utils import load_yaml
from torchsummary import summary
from dataset import ModelNetDataSet
from torch.utils.data import DataLoader
import torch

if __name__ == '__main__':   
    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("================================")
    print('Torch version: ', torch.__version__)
    print('Using device:', device)
    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('\tAllocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('\tCached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
        print("================================")
    # exit()

    conf = load_yaml('config/deform.yaml')
    # print(conf)
    weights_path = f'{conf["checkpoint"]}/model_nfeatures{conf["num_features"]}_ckpnt_{conf["epochs"]}.pth'
    weights = torch.load(weights_path)    
    eval_loader =  DataLoader(dataset=ModelNetDataSet(conf, mode="test"),  batch_size= 1) 
    model = DNet(conf['num_categories'], conf['use_normal_vector'], conf['num_features'])            
    # model.to(device='cpu')
    model.load_state_dict(weights)    
    model.eval()
    nfeatures = 6 if conf['use_normal_vector'] else 3
    npoints = conf['npoints']
    if conf['verbose']:
        summary(model, input_size=(nfeatures, npoints), device='cpu')
        print(model)  
    print("================================")
    print('Exporting ONNX...')
    print('weights_path = ', weights_path)
    
    # Call the export function
    import subprocess
    import time
    onnx_path = f'{conf["checkpoint"]}/model_nfeatures{conf["num_features"]}_ckpnt_{conf["epochs"]}.onnx'
    subprocess.run(f'rm {onnx_path}', shell=True)
    time.sleep(3)

    input = torch.zeros(1, nfeatures, npoints, requires_grad=False).to(device='cpu')    
    torch.onnx.export(model, 
                      input, 
                      onnx_path, 
                      opset_version=10)

    #Test ONNX Model
    from openvino.runtime import Core
    import numpy as np        
    ie = Core()
    onnx_model = ie.read_model(model=onnx_path)
    compiled_model = ie.compile_model(model=onnx_model, device_name="GPU")    
    output_layer = compiled_model.output(0)
    
    print(f'Load ONNX: {onnx_path}')
     #Test original models    
    i = 0
    for points, target in eval_loader:        
        points = torch.Tensor(points)        
        points = points.transpose(2, 1)        
        py_pred = model(points).detach().numpy()        
        py_cls = np.argmax(py_pred, axis=1)

        points = points.detach().numpy() 
        t1 = time.time()
        onnx_pred = compiled_model([points])[output_layer]        
        t2 = time.time()
        print('Inference time: %2.2f ms' %((t2-t1)*1000))
        onxx_cls = np.argmax(onnx_pred, axis=1)
        # print('pred =', py_pred.shape, py_pred)
        print('py_cls = ', py_cls[0], 'vs onnx cls = ', onxx_cls[0], ', target = ', target[0].item())
        i += 1
        if i > 20:
            break

    '''
    #Convert to OPENVINO format
    bin_path = f'{conf["checkpoint"]}/best_model_nfeatures{conf["num_features"]}.bin'
    mapping_path = f'{conf["checkpoint"]}/best_model_nfeatures{conf["num_features"]}.mapping'
    xml_path = f'{conf["checkpoint"]}/best_model_nfeatures{conf["num_features"]}.xml'
    subprocess.run(f'rm {bin_path}', shell=True)
    subprocess.run(f'rm {mapping_path}', shell=True)
    subprocess.run(f'rm {xml_path}', shell=True)

    time.sleep(3)
    # for simple commands
    cmd = f'mo --input_model {onnx_path} --input_shape "[1, {nfeatures}, {npoints}]" --output_dir checkpoint'
    print('Executing: ', cmd)
    subprocess.run(cmd, shell=True)
    '''