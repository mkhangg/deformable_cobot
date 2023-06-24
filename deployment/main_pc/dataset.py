import os
import numpy as np
import warnings
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset
from utils import pc_normalize, farthest_point_sample
warnings.filterwarnings('ignore')

class ModelNetDataSet(Dataset):
    def __init__(self, conf, mode='train'):
        self.conf = conf #save for other methods
        self.mode = mode

        catergory_file = os.path.join(conf['root'], f'{conf["dataset"]}{conf["num_categories"]}_shape_names.txt')        
        categories = [line.rstrip() for line in open(catergory_file)]        
        self.classes = dict(zip(categories, range(len(categories))))         
        
        if mode == 'train':
            filenames= [line.rstrip() for line in open(f'{conf["root"]}/{conf["dataset"]}{conf["num_categories"]}_train.txt')]
        else: #Test
            filenames  = [line.rstrip() for line in open(f'{conf["root"]}/{conf["dataset"]}{conf["num_categories"]}_test.txt')]        
            # print(filenames)

        assert (mode == 'train' or mode == 'test')
        if conf["dataset"] == 'modelnet':
            shape_names = [x[0:-5] for x in filenames] # filenames: shape_name_xxxx 
        else:
            shape_names = [x.split('_')[0] for x in filenames] # filenames: shape-name_x_x_x_x     
        self.datapath = [(shape_names[i], 
                        os.path.join(conf['root'], shape_names[i], filenames[i]) + '.txt') 
                        for i in range(len(filenames))] # Each element is a tuple (name. path)
        print('The number of %s pointclouds is %d. First file: %s' % (mode, len(self.datapath), self.datapath[0]))

        # if mode == 'test':
        #     print(self.datapath)
        
        if conf['is_sampled']:        
            cache_path = os.path.join(conf['cache'], '%s%d_%s_%dpts_sampling.dat' % (conf["dataset"], conf['num_categories'], mode, conf['npoints']))
        else:
            cache_path = os.path.join(conf['cache'], '%s%d_%s_%dpts.dat' % (conf["dataset"], conf['num_categories'], mode, conf['npoints']))
        
        if conf['is_cached']:
            if not os.path.exists(cache_path):
                print('Caching data %s...' % cache_path)
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index] # fn(names, path)                
                    cls = self.classes[fn[0]]                    
                    cls = np.array([cls]).astype(np.int32)
                    point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32) #(npoints, 6)

                    if conf['is_sampled']: 
                        point_set = farthest_point_sample(point_set, conf['npoints'])
                    else:
                        point_set = point_set[0:conf['npoints'], :]

                    self.list_of_points[index] = point_set #(npoints, 6)
                    self.list_of_labels[index] = cls

                with open(cache_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print('Load cached data from %s' % cache_path)
                with open(cache_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.conf['is_cached']:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

            if self.conf['is_sampled']:
                point_set = farthest_point_sample(point_set, self.conf['npoints'])
            else:
                point_set = point_set[0:self.conf['npoints'], :]
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        if not self.conf['use_normal_vector']:
            point_set = point_set[:, 0:3]

        return point_set, label[0]

    def __getitem__(self, index):
        return self._get_item(index)


import torch
from torch.utils.data import DataLoader
from utils import load_yaml
if __name__ == '__main__':
    conf = load_yaml('conf.yaml')
    print(conf)
    dataset = ModelNetDataSet(conf)    
    loader = DataLoader(dataset=dataset, batch_size=conf["batch_size"])
    for point, label in loader:
        print(point.shape)
        print(label.shape)
        break