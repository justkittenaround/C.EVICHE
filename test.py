import torch
from torchvision import datasets, models, transforms
import torch.multiprocessing as mp
import torch.utils.data as data_utils
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
import h5py
from glob import glob
import skimage
from scipy.misc import imsave, imresize
import os
import sys
import copy
import cv2
from PIL import Image as pil
from scipy import signal
import csv
import visdom
vis = visdom.Visdom()

label_file = '/home/whale/Desktop/Rachel/CeVICHE/Time2Seize - Sheet1 (3) - Time2Seize - Sheet1 (3).csv'
vid_folder = '/home/whale/Desktop/Rachel/CeVICHE/conv_ceviche_data/Evaluate/'

# PATH = '/home/blu/C.EVICHE/saved_models/modelsvgg (1).pt'
PATH = '/home/whale/Desktop/Rachel/CeVICHE/models/modelsvgg.pt'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 1
num_classes = 7
input_size = 224
thresh = 61
fps = 29

##progress bar##################################################################
def progress(count, total, status=''):
    bar_len = 40
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

##TESTING PROCEDURE AND PRINTS################################################
def testing_model(model, dataloaders, vid, total_frames, names):
    since = time.time()
    seizing_preds = []
    t = 0
    worms_seizing_act = []
    testing_acc = []
    confusion = np.zeros([8, 8])
    running_corrects = 0
    for inputs, labels in dataloaders[phase]:
        s = time.time()
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = test_model(inputs)
        _, predicts = torch.max(outputs, 1)
        running_corrects += torch.sum(predicts == labels.data)
        preds = predicts.cpu().numpy()
        seizing_preds.append(preds)
        worms_seizing_act.append(labels.data.cpu().numpy())
        seizing_preds = np.asarray(seizing_preds)
        med = signal.medfilt(seizing_preds, 91) #kernal size should be odd
        confusion[seizing_preds[-1], worms_seizing_act[-1]] += 1
        vis.line(seizing_preds, win='seizing_preds', opts=dict(title= 'Raw_Predictions'))
        vis.line(med, win='medfilter_preds', opts=dict(title= 'Medfilter_Predictions'))
        vis.line(worms_seizing_act, win='worms_seizing_target', opts=dict(title= 'trainedVGG-Target'))
        vis.heatmap(confusion, win='confusion_matrix', opts=dict(ylabel= 'predicted', xlabel= 'target', colormap= 'Electric'))
        p = time.time() - s
        progress(t, total_frames, status=('predicting ' + str(t)))
        time.sleep(p)
    testing_acc = (running_corrects/len(labels.data))
    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Testing_acc:', testing_acc.cpu().numpy())
    total_worms_seizing = max(smoothed)
    total_worms_seizing = max(med)
    avg_med = ((sum(med)/(len(med)/total_frames))/total_worms_seizing)/fps
    print('REPORT: ')
    print('Video Analyzed: ', vid)
    print('Total Worms Seizing: ', total_worms_seizing)
    print('Average time Seizing: ', avg_med + ' seconds')
    with open('test.csv', 'wb') as csvfile:
        s = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        s.writerow('Ground Truth:', worms_seizing_act)
        s.writerow('Predicted:', med)
        s.writerow('Confusion Matrix:', confusion)
        s.writerow('Accuracy:', testing_acc.cpu().numpy())
        s.writerow('Predicted Average Time Seizing:', avg_med)
        s.writerow('Video Analyzed:', vid)
    if vid != names[-1]:
        print('Initializing Next Video...')
    else:
        print('Analysis Complete!')
    return test_model, testing_acc

##INITIALIZE AND RESHAPE MODEL##################################################
def initialize_model(PATH):
    test_model= torch.load(PATH)
    test_model.eval()
    # Initialize the test_modelfor this run
    test_model = initialize_model(PATH)
    #Send the model to GPU
    test_model= test_model.to(device)
    return test_model

#Load Data and RUN##############################################################
names = os.listdir(vid_folder)
for vid in names:
    split = vid.split('.')
    tf = split[-1]
    phase = str(vid)
    data_transforms = {
        phase: transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    root = vid_folder + phase + '/'
    image_datasets = {x: datasets.ImageFolder(os.path.join(root), data_transforms[x]) for x in [phase]}
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False, num_workers=4) for x in [phase]}
    print("Initializing Datasets and Dataloaders...")
    #Run test model
    test_model = initialize_model(PATH)
    test_model, testing_acc = testing_model(testing_model, dataloaders_dict, vid, tf, names)
