from DNet import DNet, DLoss

from utils import load_yaml
import copy
from dataset import ModelNetDataSet
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import logging
import numpy as np
import pandas as pd
import os

def eval_model(model, conf, loader):    
    model.eval()   
    num_correct = 0
    total = 0    
    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)): 
        points = torch.Tensor(points)   #(bs, npoints, 3 or 6)
        points = points.transpose(2, 1) #(bs, 3 or 6, npoints)            
        if conf['use_gpu']:
            points, target = points.cuda(), target.cuda()           
    
        vote_pool = torch.zeros(target.size()[0], conf['num_categories']).cuda()                           
        for _ in range(conf['num_votes']):            
            pred = model(points) # (bs, num_cls)
            vote_pool += pred   
        pred = vote_pool / conf['num_votes']
        pred_choice = pred.data.max(1)[1]   # (bs)        
        num_correct += pred_choice.eq(target.long().data).cpu().sum()
        total += points.shape[0]

    return num_correct, total, float(num_correct/total)
    

def train(model, conf, train_loader, criterion):    
    model.train()
    ncorrect = 0
    nsamples = 0      
    for batch_id, (points, target) in tqdm(enumerate(train_loader, 0), total=len(train_loader), smoothing=0.9):            
        # points = points.data.numpy()
        # points = provider.random_point_dropout(points)
        # points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
        # points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
        points = torch.Tensor(points)   #(bs, npoints, 3 or 6)
        points = points.transpose(2, 1) #(bs, 3 or 6, npoints)            
        if conf['use_gpu']:
            points, target = points.cuda(), target.cuda()                            
        pred = model(points) #pred = (bs, num_class)            

        # Back-probagation
        loss = criterion(pred, target.long())            
        optimizer.zero_grad() # Clear gradients
        loss.backward()
        optimizer.step()

        # Calculate training accuracy
        pred_choice = pred.max(1)[1] #(bs), find max -> get index or class id for each pointcloud
        ncorrect += pred_choice.eq(target.long().data).cpu().sum() #Number of correctness in a batch
        nsamples += points.shape[0]                        
    train_acc = float(ncorrect/nsamples)        
    train_loss = loss.detach().cpu()
    return train_acc, train_loss.item()
        
    

from torchsummary import summary
if __name__ == '__main__':   
    conf = load_yaml('config/deform.yaml')
    print(conf)
    train_loader = DataLoader(dataset=ModelNetDataSet(conf, mode="train"), batch_size=conf["batch_size"]) 
    eval_loader =  DataLoader(dataset=ModelNetDataSet(conf, mode="test"),  batch_size= conf["batch_size"]) 
    #Log
    if os.path.exists(f'{conf["log_dir"]}/train.log'):
        os.remove(f'{conf["log_dir"]}/train.log')
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f'{conf["log_dir"]}/train.log'),
            logging.StreamHandler()
        ],
        
    )
    logging.info(str(conf))
    model = DNet(conf['num_categories'], conf['use_normal_vector'], conf['num_features'])        
    if conf['verbose']:
        summary(model, input_size=(6, 50), device='cpu')
        print(model)        
        exit()
    
    criterion = DLoss()    
    if conf['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=conf['learning_rate'],
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=conf['decay_rate']
        )
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    if conf['use_gpu']:
        model, criterion = model.cuda(), criterion.cuda()        
    
    best_eval_acc = 0.0
    logging.info('Start training...')
    results = []

    for epoch in range(0, conf['epochs']):                                
        train_acc, train_loss = train(model, conf, train_loader, criterion)                
        with torch.no_grad():
            num_eval_correct, eval_nsamples, eval_acc = eval_model(model, conf, eval_loader)
            logging.info(f" Epoch = {epoch + 1}/{conf['epochs']}, loss = {train_loss:.4f}, train_acc : {train_acc:.4f}, eval_acc : {eval_acc:.4f} = {num_eval_correct}/{eval_nsamples}. Best: {best_eval_acc:.4f}")    
            results.append([epoch + 1, train_loss, train_acc, eval_acc])
            if (eval_acc >= best_eval_acc):
                best_eval_acc = eval_acc                                              
                savepath = f'{conf["checkpoint"]}/best_model_nfeatures{conf["num_features"]}.pth'
                logging.info('Saving at %s' % savepath)                
                torch.save(model.state_dict(), savepath)
            if epoch + 1 == conf['epochs']:
                savepath = f'{conf["checkpoint"]}/model_nfeatures{conf["num_features"]}_ckpnt_{epoch + 1}.pth'
                logging.info('Saving at %s' % savepath)
                torch.save(model.state_dict(), savepath)
        scheduler.step()
    #end of for epoch

    # print(results)
    df = pd.DataFrame(results)
    df.to_csv(f'{conf["log_dir"]}/results.csv',mode='w', index=False, header=False)
    #Plot results
    results = np.array(results)
    import matplotlib.pyplot as plt
    plt.plot(results[:, 0], results[:, 2])
    plt.plot(results[:, 0], results[:, 3])
    plt.xlabel('Epoches')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Eval'])
    plt.show()