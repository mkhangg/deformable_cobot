import torch
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.nn.functional as F
from torch.autograd import Variable


class STN3d(nn.Module):
    def __init__(self, num_channels=None, num_features=None):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(num_channels, num_features, 1)
        self.fc1 = nn.Linear(num_features, 9)
        self.gelu = nn.GELU('tanh')        
        # self.bn1 = nn.BatchNorm1d(num_features)

    def forward(self, x):                       # x: (bs, 6, npoints)
        # x = F.relu(self.bn1(self.conv1(x)))     # x: (bs, nfeatures, npoints): extend 3 or 6 features to nfeatures
        x = self.gelu(self.conv1(x))     # x: (bs, nfeatures, npoints): extend 3 or 6 features to nfeatures
        x = torch.max(x, 2, keepdim=True)[0]    # x: (bs, nfeatures, 1): Choose best features between npoints
        x = x.view(-1, x.size()[1])             # x: (bs, nfeatures)
        x = self.fc1(x)                         # x: (bs, 9)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 
                                                   0, 1, 0, 
                                                   0, 0, 1]).astype(np.float32))).view(1, 9).repeat(x.size()[0], 1)
        iden = iden.cuda() if x.is_cuda else iden
        x = x + iden
        x = x.view(-1, 3, 3)    # x: (bs, 3, 3)

        return x
    

class PointNetEncoder(nn.Module):
    def __init__(self, num_channels=None, num_features=None):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(num_channels=num_channels, num_features=num_features)
        self.conv1 = torch.nn.Conv1d(num_channels, int(num_features/2), 1)
        self.conv2 = torch.nn.Conv1d(int(num_features/2), num_features, 1)
        # self.bn1 = nn.BatchNorm1d(int(num_features/2))
        # self.bn2 = nn.BatchNorm1d(num_features)        

    def forward(self, x):                               # x = (bs, 6, npoints=2000)
        _, D, N = x.size()
        spatial_point_feature = self.stn(x)             # (bs, 6, n) => (bs, 3, 3)
        x = x.transpose(2, 1)                           # (bs, 6, n) => (bs, npoints, 6)
        
        if D > 3:
            feature = x[:, :, 3:]                       # (bs, npoints, 3)
            x = x[:, :, :3]                             # (bs, npoints, 3): spatial information XYZ
        x = torch.bmm(x, spatial_point_feature)         # batch matrix multiplication: (bs, npoints, 3) x (bs, 3, 3) => (bs, npoints, 3): (bs, npoints, 3)
        if D > 3:
            x = torch.cat([x, feature], dim=2)          # (bs, n, 6)
        x = x.transpose(2, 1)                           # (bs, 6, n)
        # x = F.relu(self.bn1(self.conv1(x)))             # (bs, num_features/2,  n)        
        x = F.relu(self.conv1(x))             # (bs, num_features/2,  n)        

        # x = F.relu(self.bn2(self.conv2(x)))             # (bs, num_features, n)
        x = F.relu(self.conv2(x))             # (bs, num_features, n)
        x = torch.max(x, 2, keepdim=True)[0]            # (bs, num_features, 1)
        x = x.view(-1, x.size()[1])                     # (bs, num_features)

        return x
        