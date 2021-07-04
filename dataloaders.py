#some basics of dataloaders

import torch
import torchvision
import torchvision.transforms as transforms

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
'''
print('the length is ',len(train_set))

print(train_set.targets) #this lists the target labels
print(train_set.targets.bincount()) #the frequency in each class

sample = next(iter(train_set))
print(len(sample)) #gives 2 ----one for image one for label

image, label = sample
print(image.shape)

plt.imshow(image.squeeze(), cmap='gray')
torch.tensor(label)
print('label :', label)
plt.show()
'''
#------------the dataloaders now---------------

display_loader = torch.utils.data.DataLoader(
    train_set, batch_size = 10
)

batch = next(iter(display_loader))
print('len :', len(batch)) #again we get 2

images , labels = batch #Instead of single image like last time, we get [10,1,28,28]

grid = torchvision.utils.make_grid(images, nrow = 10)
plt.figure(figsize=(15,15))
plt.imshow(grid.permute(1,2,0))
plt.show()
print('labels:',labels)

#the labels for reference

'''
Index	Label
    0	T-shirt/top
    1	Trouser
    2	Pullover
    3	Dress
    4	Coat
    5	Sandal
    6	Shirt
    7	Sneaker
    8	Bag
    9	Ankle boot
'''


