import torch
import torchvision
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch import optim
from torch import nn
from torch.utils.data import DataLoader

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 12, kernel_size = 5)

        self.fc1 = nn.Linear(in_features = 12*4*4, out_features = 120)
        self.fc2 = nn.Linear(in_features = 120, out_features = 60)
        self.out = nn.Linear(in_features = 60, out_features = 10)
    
    def forward(self, t):
        #the input layer should take a tensor and retain the same shape 
        #like the identity transformation f(x) = x
        # (1)
        t = t
        # (2)
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size = 2, stride = 2)
        # (3)
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size = 2, stride = 2)
        # whenever you pass from conv to fc layer, always flatten
        # (4)
        t = t.reshape(-1, 12*4*4)
        t = self.fc1(t)
        t = F.relu(t)
        # (5)
        t = self.fc2(t)
        t = F.relu(t)
        # (6)
        t = self.out(t)
        '''why no softmax-- we wont need softmax for crossentropy since it implicity
        performs softmax'''
        return t  


