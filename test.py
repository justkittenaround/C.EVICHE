import torch
from torchvision import datasets, models, transforms
import torch.multiprocessing as mp
import torch.utils.data as data_utils
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torch.nn as nn
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
import seaborn as sn
import pandas as pd
import csv
# import visdom
# vis = visdom.Visdom()


# vid_folder = '/home/whale/Desktop/Rachel/CeVICHE/conv_ceviche_data/pseudoEval/'
vid_folder = '/home/whale/Desktop/Rachel/CeVICHE/conv_ceviche_data/Evaluate' + '/'
results_foler = '/home/whale/Desktop/Rachel/CeVICHE/Test_Results/'
PATH = '/home/whale/Desktop/Rachel/CeVICHE/models/models/vgg.pt'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 1
num_classes = 7
input_size = 224
thresh = 91
fps = 29
class DataParallelModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Linear(10, 20)
        # wrap block2 in DataParallel
        self.block2 = nn.Linear(20, 20)
        self.block2 = nn.DataParallel(self.block2)
        self.block3 = nn.Linear(20, 20)
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

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
    print(vid)
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
        preds = int(preds)
        seizing_preds.append(preds)
        worms_seizing_act.append(labels.data.cpu().numpy())
        # seizing_preds = np.asarray(seizing_preds)
        med = signal.medfilt(seizing_preds, 91) #kernal size should be odd
        confusion[seizing_preds[-1], worms_seizing_act[-1]] += 1
        # vis.line(seizing_preds, win='seizing_preds', opts=dict(title= 'Raw_Predictions'))
        # vis.line(med, win='medfilter_preds', opts=dict(title= 'Medfilter_Predictions'))
        # vis.line(worms_seizing_act, win='worms_seizing_target', opts=dict(title= 'trainedVGG-Target'))
        # vis.heatmap(confusion, win='confusion_matrix', opts=dict(ylabel= 'predicted', xlabel= 'target', colormap= 'Electric'))
        # vis.text(vid, win='Info')
        t += 1
        p = time.time() - s
        progress(t, total_frames, status=('predicting ' + str(t)))
        time.sleep(p)
    testing_acc = (running_corrects/len(labels.data))
    total_worms_seizing_pred = int(max(med))
    total_worms_seizing_act = int(max(worms_seizing_act))
    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Testing_acc:', testing_acc.cpu().numpy())
    # vis.text(vid, win='Info')
    # vis.text(error, win='Info', opts=dict(title= 'error'))
    pred_w = total_worms_seizing_pred
    act_w = total_worms_seizing_act
    if total_worms_seizing_pred == 0:
        pred_w = 1
    if total_worms_seizing_act == 0:
        act_w = 1
    avg_med = ((int(sum(med))/(int(len(med))/int(total_frames)))/int(pred_w))/fps
    avg_act = ((int(sum(worms_seizing_act))/(int(len(worms_seizing_act))/int(total_frames)))/int(act_w))/fps
    if avg_act == 0:
        error = str(((avg_act - avg_med)/1)*100) + '%'
    else:
        error = str(((avg_act - avg_med)/avg_act)*100) + '%'
    print('REPORT: ')
    print('Video Analyzed: ', vid)
    print('Total Worms Seizing: ', total_worms_seizing_act)
    print('Predicted Average time Seizing: ', avg_med, ' seconds')
    print('Actual Average time Seizing: ', avg_act, ' seconds')
    print('Percent Error:', error)
    folders = os.listdir(results_foler)
    if vid not in folders:
        os.mkdir(results_foler + vid)
    predplt = plt.figure()
    plt.plot(seizing_preds)
    predplt.savefig(results_foler + vid + '/Raw_Predictions')
    medplt = plt.figure()
    plt.plot(med)
    medplt.savefig(results_foler + vid + '/filtered_predictions')
    actplt = plt.figure()
    plt.plot(worms_seizing_act)
    actplt.savefig(results_foler + vid + '/actual_predictions')
    row_labels = ['0', '1', '2', '3', '4', '5', '6', 'seizing_predicted']
    col_labels = ['0', '1', '2', '3', '4', '5', '6', 'seizing_actual']
    fig, ax = plt.subplots()
    im = ax.imshow(confusion)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            text = ax.text(j, i, confusion[i, j],
                           ha="center", va="center", color="w")
    ax.set_title("Predicted Worms Seizing per Frame Accuracy)")
    fig.tight_layout()
    plt.savefig(results_foler + vid + '/confusion_matrix')
    with open((results_foler+vid+ '/'+vid+'.csv'), 'w') as csvfile:
        s = csv.writer(csvfile, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
        s.writerow(['error:', error, 'avg_time_seizing_predicted:', avg_med,'avg_time_seizing_actual:', avg_act, 'predicted_worms_seizing:', total_worms_seizing_pred, 'actual_worms_seizing:', total_worms_seizing_act])
    if vid != names[-1]:
        print('Initializing Next Video...')
    else:
        print('Analysis Complete!')
    return test_model, testing_acc

##INITIALIZE AND RESHAPE MODEL##################################################
def initialize_model(PATH):
    test_model= torch.load(PATH)
    test_model.eval()
    if torch.cuda.device_count() > 1:
          print("Let's use", torch.cuda.device_count(), "GPUs!")
          # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
          test_model = nn.DataParallel(test_model)
    test_model= test_model.to(device)
    return test_model

#Load Data and RUN##############################################################
names = os.listdir(vid_folder)
print(len(names))
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
    image_datasets = {x: datasets.ImageFolder(root, data_transforms[x]) for x in [phase]}
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False, num_workers=4) for x in [phase]}
    print("Initializing Datasets and Dataloaders...")
    #Run test model
    test_model = initialize_model(PATH)
    test_model, testing_acc = testing_model(testing_model, dataloaders_dict, vid, tf, names)
