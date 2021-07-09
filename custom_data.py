"""This notebook shows how to load if you have your own custom data in pytorch
"""

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import os
import pandas as pd
from skimage import io
from torch.utils.data import Dataset, DataLoader

class CatsandDogs(Dataset):
    def __init__(self, csv_file, root_dir, transform = None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0]) #0 cause the first column has image path
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)
        
        return (image, y_label)

dataset = CatsandDogs(
    csv_file = 'cat_dog.csv',
    root_dir= 'cat_dog_resized',
    transform = transforms.ToTensor()
)

train_set, test_set = torch.utils.data.random_split(dataset, [5, 5])
train_loader = DataLoader(dataset=train_set, batch_size= 128, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size= 128, shuffle=True)