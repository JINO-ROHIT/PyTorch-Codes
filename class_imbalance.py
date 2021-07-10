import torch
import torchvision.datasets as datasets
import os
from torch.utils.data import WeightedRandomSampler, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn

def get_loader(root_dir, batch_size):
    my_transforms = transforms.Compose(
        [
           transforms.Resize((224,224)),
           transforms.ToTensor(),
        ]    
    )

    dataset = datasets.ImageFolder(root = root_dir, transform = my_transforms)
    #class_weights = [1,50] #since retriver has 50 imgs and elkhound has only one, put more emphasis on elkhounds
    #instead of manually assigning class weights
    class_weights = []
    for root, subdir, files in os.walk(root_dir):
        if len(files) > 0:
            class_weights.append(1/len(files))
    sample_weights = [0] * len(dataset) #initialize all to zero at start

    for idx, (data, label) in enumerate(dataset):
        class_weight = class_weights[label]
        sample_weights[idx] = class_weight
    
    sampler = WeightedRandomSampler(sample_weights, num_samples = len(dataset), replacement = True)
    loader = DataLoader(dataset, batch_size = batch_size, sampler = sampler)
    return loader

def main():
    loader = get_loader(root_dir='data_imbalance', batch_size= 8)

    for data, labels in loader:
        print(labels)

if __name__ == "__main__":
    main()