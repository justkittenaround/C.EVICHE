from torchvision import datasets, models, transforms
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
import visdom
vis = visdom.Visdom()

PATH = '/home/whale/Desktop/Rachel/CeVICHE/models/modelsvgg.pt'
data_dir = '/home/whale/Desktop/Rachel/CeVICHE/conv_ceviche_data'
batch_size = 1

#load the saved model
test_model = models.load(PATH)
test_model.eval()

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

dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False, num_workers=4) for x in ['train', 'val']}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#run on GPU
test_model = test_model.to(device)

#run data through and print stuff
worms_seizing_pred = []
worms_seizing_act = []
for inputs, labels in dataloaders[phase]:
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    worms_seizing_pred.append(preds)
    worms_seizing_act.append(labels.data)
    vis.line(worms_seizing_pred, win='worms_seizing', opts=dict(title= model_name + '-predictions'))
    vis.line(worms_seizing_act, win='worms_seizing', opts=dict(title= model_name + '-predictions'))
