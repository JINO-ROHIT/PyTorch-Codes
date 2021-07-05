import torch
import torchvision
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch import optim
from torch import nn
from torch.utils.data import DataLoader

import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt

torch.set_printoptions(linewidth = 120)

train_set = torchvision.datasets.FashionMNIST(
    root = './data',
    train = True,
    download = True,
    transform = transforms.Compose([transforms.ToTensor()
    ])
)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 12, kernel_size = 5)

        self.fc1 = nn.Linear(in_features = 12*4*4, out_features = 120)
        self.fc2 = nn.Linear(in_features = 120, out_features = 60)
        self.out = nn.Linear(in_features = 60, out_features = 10)
    
    def forward(self, t):
        t = t
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size = 2, stride = 2)
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size = 2, stride = 2)
        t = t.reshape(-1, 12*4*4)
        t = self.fc1(t)
        t = F.relu(t)
        t = self.fc2(t)
        t = F.relu(t)
        t = self.out(t)
        return t  
    
# we wont be training yet, so turn of the gradient graph to conserve memory
torch.set_grad_enabled(False)

network = Network()
sample = next(iter(train_set))
image, label = sample
print(image.shape) # [1,28,28]

#but our network has 4 ranks-- expects a batch also (batch_size,in_channels,height,width)
print(image.unsqueeze(0).shape)

#now pass to network
pred = network(image.unsqueeze(0))
print(pred)
print('the original label',label)
print('the predicted label',pred.argmax(dim = 1))
print('as probabilties', F.softmax(pred, dim =1))

#-------------------------------
#now we pass as batches using the dataloader
data_loader = torch.utils.data.DataLoader(
    train_set, batch_size = 10
)

batch = next(iter(data_loader))
images, labels = batch

preds = network(images)
print('the predicted labels', preds.argmax(dim = 1))
print('true labels', labels)

def get_num_correct(preds, labels):
    return preds.argmax(dim = 1).eq(labels).sum().item()

ans = get_num_correct(preds, labels)
print('{} predictions were correct out of 10'.format(ans))
