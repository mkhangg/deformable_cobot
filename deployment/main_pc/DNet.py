import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from net_utils import PointNetEncoder


class DNet(nn.Module):
    def __init__(self, num_classes=None, normal_channel=True, num_features=128):
        super(DNet, self).__init__()
        num_channels = 6 if normal_channel else 3
        self.feat = PointNetEncoder(num_channels, num_features)
        self.fc1 = nn.Linear(num_features, num_classes)        

    def forward(self, x):                       #(bs, 6, npoints)
        x = self.feat(x)         #x = (bs, nfeatures), _, trans_feat = (bs, 64, 64)
        x = self.fc1(x)                         #(bs, num_class)
        x = F.log_softmax(x, dim=1)             #(bs, num_class)
        return x


class DLoss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(DLoss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = 0        
        return loss + mat_diff_loss * self.mat_diff_loss_scale
