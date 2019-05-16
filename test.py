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
num_classes = 7
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_size = 224
phase = 'train'

##DATASETS AND DATALOADERS################################################
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
print("Initializing Datasets and Dataloaders...")
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in [phase]}
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False, num_workers=4) for x in [phase]}

##TESTING PROCEDURE AND PRINTS################################################
def testing_model(model, dataloaders):
    since = time.time()

    worms_seizing_pred = [np.array([7])]
    worms_seizing_smoothed = [np.array([7])]
    worms_seizing_act = [np.array([7])]
    testing_acc = []
    confusion = np.zeros([8, 8])
    running_corrects = 0
    count = 2

    for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = test_model(inputs)
        _, predicts = torch.max(outputs, 1)
        running_corrects += torch.sum(predicts == labels.data)
        preds = predicts.cpu().numpy()
        worms_seizing_pred.append(preds)
        worms_seizing_act.append(labels.data.cpu().numpy())
        seizing_preds = np.asarray(worms_seizing_pred)
        print(count, len(seizing_preds))
        if len(seizing_preds) == count and :
            max = np.amax(seizing_preds[count:])
            worms_seizing_smoothed.append(max)
            count += 5
        vis.line(worms_seizing_smoothed, win='worms_seizing_smoothed', opts=dict(title= 'smoothed_predictions'))
        vis.line(worms_seizing_pred, win='worms_seizing_predict', opts=dict(title= 'trainedVGG-Predictions'))
        vis.line(worms_seizing_act, win='worms_seizing_target', opts=dict(title= 'trainedVGG-Target'))
        confusion[worms_seizing_pred[-1], worms_seizing_act[-1]] += 1
        vis.heatmap(confusion, win='confusion_matrix', opts=dict(ylabel= 'predicted', xlabel= 'target', colormap= 'Electric'))
    testing_acc = (running_corrects/len(labels.data))
    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Testing_acc:', testing_acc.cpu().numpy())
    return test_model, testing_acc

## INITIALIZE AND RESHAPE MODEL################################################
def initialize_model(PATH):
    test_model= torch.load(PATH)
    test_model.eval()
    return test_model

# Initialize the test_modelfor this run
test_model = initialize_model(PATH)
#Send the model to GPU
test_model= test_model.to(device)
#Run test model
test_model, testing_acc = testing_model(testing_model, dataloaders_dict)
