import torch
import torchvision
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch import optim
from torch import nn
from torch.utils.data import DataLoader

class Network(nn.Module): #the module class can keep track of the weights
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 12, kernel_size = 5)

        self.fc1 = nn.Linear(in_features = 12*4*4, out_features = 120)
        self.fc2 = nn.Linear(in_features = 120, out_features = 60)
        self.out = nn.Linear(in_features = 60, out_features = 10)
    
    def forward(self, t):
        pass

'''
kernel_size -- simply the filter size
out_channels -- tells the number of filters (the depth as well)
in_channels -- the input channels
out_features -- the size of the output tensor

'''

network = Network()
print(network) #the bias by default set to True
print(network.conv1) #access this layer
print(network.conv1.weight) #the weights

#one easy way to access layers
for name, param in network.named_parameters():
    print(name, param.shape)