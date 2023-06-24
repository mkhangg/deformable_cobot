from DNet import DNet
from utils import load_yaml
from torchsummary import summary
import torch
from dataset import ModelNetDataSet
from torch.utils.data import DataLoader
import numpy as np

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
    print(conf)
    weights_path = f'{conf["checkpoint"]}/best_model_nfeatures{conf["num_features"]}.pth'
    weights = torch.load(weights_path)
    model = DNet(conf['num_categories'], conf['use_normal_vector'], conf['num_features'])    
    model.load_state_dict(weights)
    model.to(device='cuda')
    nfeatures = 6 if conf['use_normal_vector'] else 3
    npoints = conf['npoints']
    if conf['verbose']:
        summary(model, input_size=(nfeatures, npoints), device='cuda')
        # print(model)  
    eval_loader =  DataLoader(dataset=ModelNetDataSet(conf, mode="test"),  batch_size= 1) 

    i = 0
    for points, target in eval_loader:
        # print('points: ', points[0,0,:])
        points = torch.Tensor(points)
        # print('points =', points.shape)
        points = points.transpose(2, 1)
        points = points.cuda()
        # print('points =', points.shape)
        pred = model(points)
        cls = pred.data.max(1)[1]
        print('pred =', pred.shape, pred)
        print('cls = ', cls[0].item(), ', target = ', target[0].item())
        i += 1
        if i > 3:
            break