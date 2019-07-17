import torch
import torch.multiprocessing as mp
import torch.utils.data as data_utils
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time
import os
import sys
import csv
import copy
import cv2
import statistics as stats
from PIL import Image as pil
import visdom
vis = visdom.Visdom()


PATH = '/home/whale/Desktop/Rachel/CeVICHE/models/oldmodels/modelsvgg.pt'

root = '/media/whale/biodata/nematodes/mutantavi'
label_file = '/home/whale/Desktop/Rachel/CeVICHE/Data/mutants/Time2Seize - Sheet1 (3) - Time2Seize - Sheet1 (3).csv'
results_folder = '/home/whale/Desktop/Rachel/CeVICHE/Data/mutants/test_results/'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# root = '/home/whale/Desktop/Rachel/CeVICHE/Data/psuedoval'
# label_file = '/home/whale/Desktop/Rachel/CeVICHE/Data/psuedoval/pseudolabel.csv'
# results_folder = '/home/whale/Desktop/Rachel/CeVICHE/Data/psuedoval/pseudoresults/'

input_size = 224
thresh = 91 #has to be odd

##initialize_model#############################################################
test_model= torch.load(PATH)
test_model.eval()
test_model= test_model.to(device)
##progress bar##################################################################
def progress(count, total, status=''):
    bar_len = 40
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()
##TESTING PROCEDURE AND PRINTS#################################################
labels = np.genfromtxt(label_file, delimiter=',', skip_header=2, usecols=range(0,20))
data_transforms = transforms.Compose([transforms.Resize(input_size),transforms.CenterCrop(input_size),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
#images = ['pil_image', 'frame_position', 'video_number']
root = os.path.expanduser(root)
files = [d for d in os.listdir(root) if os.path.isfile(os.path.join(root, d))]
files.sort()
folder = sorted(files)
for target in folder:
    d = os.path.join(root, target)
    if not os.path.isfile(d):
        continue
    cap = cv2.VideoCapture(d)
    total_frames = int(round(cap.get(7)))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print('Total frames in ', target, ' is: ', total_frames)
    confusion = np.zeros([8, 8])
    row = (folder.index(target))
    print('label: ', labels[row, :])
    seizing_worms = int(labels[row, 12])
    bubble = int(labels[row, 13])
    w1 = labels[row, 14]
    w2 = labels[row, 15]
    w3 = labels[row, 16]
    w4 = labels[row, 17]
    w5 = labels[row, 18]
    w6 = labels[row, 19]
    check = np.zeros((6, total_frames))
    worms = [int(w1), int(w2), int(w3), int(w4), int(w5), int(w6)]
    for idx, worm in enumerate(worms):
        if worm == 0:
            break
        elif worm >= total_frames:
            worm = (total_frames - 1)
        for frame_index in range(bubble, worm):
            # print(frame_index)
            check[idx, frame_index] = 1
    worms_count = np.sum(check,axis=0)
    seizing_preds = []
    frames_bin= []
    t = 0
    check_type = np.array([1])
    print("Running Predictions...")
    since = time.time()
    for t in range(0, (total_frames+1)):
        s = time.time()
        cap.set(cv2.CAP_PROP_POS_FRAMES,t)
        ret, frame = cap.read()
        if type(frame) is not type(check_type):
            print('skipped', t)
            break
        worms_count = worms_count.astype(int)
        worm_target = list(worms_count[:(t+1)])
        frame = pil.fromarray(frame)
        frame = data_transforms(frame)
        frame = frame.unsqueeze(0)
        input = frame.to(device)
        outputs = test_model(input)
        _, predicts = torch.max(outputs, 1)
        preds = predicts.cpu().numpy()
        preds = int(preds)
        vis.line(worm_target, win='target_preds', opts=dict(title= 'Target_Predictions'))
        seizing_preds.append(preds)
        vis.line(seizing_preds, win='seizing_preds', opts=dict(title= 'Raw_Predictions'))
        med = signal.medfilt(seizing_preds, 91) #kernal size should be odd
        vis.line(med, win='medfilter_preds', opts=dict(title= 'Medfilter_Predictions'))
        p = time.time() - s
        progress(t, total_frames, status=('predicting ' + str(t)))
        time.sleep(p)
    time_elapsed = time.time() - since
    print('Testing completed in {:.0f}m {:.0f}'.format(time_elapsed // 60, time_elapsed % 60), 'for ' + str(target))
    folders = os.listdir(results_folder)
    if target not in folders:
        os.mkdir(results_folder + target)
    for n in range(total_frames-1):
        confusion[int(med[n]), worm_target[n]] += 1
    predplt = plt.figure()
    plt.plot(seizing_preds)
    predplt.savefig(results_folder + target + '/Raw_Predictions')
    medplt = plt.figure()
    plt.plot(med)
    medplt.savefig(results_folder + target + '/filtered_predictions')
    actplt = plt.figure()
    plt.plot(worms_count)
    actplt.savefig(results_folder + target + '/actual_predictions')
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
    plt.savefig(results_folder + target + '/confusion_matrix')
    total_worms_seizing_pred = max(med)
    total_worms_seizing_act = max(worm_target)
    pred_w = total_worms_seizing_pred
    act_w = total_worms_seizing_act
    if total_worms_seizing_pred == 0:
        pred_w = 1
    if total_worms_seizing_act == 0:
        act_w = 1
    avg_med = ((int(sum(med))/(int(len(med))/int(total_frames)))/int(pred_w))/fps
    avg_act = ((int(sum(worm_target))/(int(len(worm_target))/int(total_frames)))/int(act_w))/fps
    if avg_act == 0:
        error = str(((avg_act - avg_med)/1)*100) + '%'
    else:
        error = str(((avg_act - avg_med)/avg_act)*100) + '%'
    pre = []
    re = []
    for n in range(0,(total_worms_seizing_act+1)):
        tp = confusion[n,n]
        fp = confusion[n,:8]
        fn = confusion[:8,n]
        prec = tp/(sum(fp)+tp)
        pre.append(prec)
        rec = tp/(sum(fn)+tp)
        re.append(rec)
    precision = stats.mean(pre)
    recall = stats.mean(re)
    fscore = (2*(precision*recall))/(precision+recall)
    precision = str(precision)
    recall = str(recall)
    fscore = str(fscore)
    # roc = plt.figure()
    # plt.plot(precision, recall)
    # roc.savefig(results_folder + target + '/ROC')
    # auc = 7
    prename = target.split('.')
    name = prename[0]
    with open((results_folder + target + '/' + name + '.csv'), 'w') as csvfile:
        s = csv.writer(csvfile, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
        s.writerow(['time elapsed:', time_elapsed, 'error:', error, 'avg_time_seizing_predicted:', avg_med,'avg_time_seizing_actual:', avg_act, 'predicted_worms_seizing:', total_worms_seizing_pred, 'actual_worms_seizing:', total_worms_seizing_act, 'precision: ', precision, 'recall: ', recall, 'fscore: ', fscore])

    print('REPORT: ')
    print('Video Analyzed: ', target)
    print('Total Worms Seizing: ', total_worms_seizing_pred)
    print('Average time Seizing: ', avg_med, '(seconds)')
    print('Precision: ', precision)
    print('Recall: ', recall)
    # print('AUC: ', auc)


    if target != folder[-1]:
        print('Initializing Next Video...')
    else:
        print('Analysis Complete!')
