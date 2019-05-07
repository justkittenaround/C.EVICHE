from torchvision import datasets, models, transforms
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
import visdom
vis = visdom.Visdom()

PATH = '/home/whale/Desktop/Rachel/CeVICHE/modelsvgg.pt'
data_dir = '/home/whale/Desktop/Rachel/CeVICHE/conv_ceviche_data'

#load the saved model
feature_model = models.vgg11_bn()
feature_model.load_state_dict(torch.load(PATH))
feature_model.eval()

#data
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
print("Initializing Datasets and Dataloaders...")

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#run on GPU
feature_model = feature_model.to(device)

for inputs, labels in dataloaders[phase]:
    inputs = inputs.to(device)
    labels = labels.to(device)
    my_feat = feature_model.features(inputs)
    print(my_feat.shape)
