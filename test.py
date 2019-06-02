import torch
from torchvision import datasets, models, transforms
import torch.multiprocessing as mp
import torch.utils.data as data_utils
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
import copy
import cv2
from PIL import Image as pil
import visdom
vis = visdom.Visdom()

label_file = '/home/whale/Desktop/Rachel/CeVICHE/Time2Seize - Sheet1 (3).csv'
vid_folder = '/home/whale/Desktop/Rachel/CeVICHE/Data/Evaluate/'
save_path = '/home/whale/Desktop/Rachel/CeVICHE/conv_ceviche_data/test/Evaluate'

# PATH = '/home/blu/C.EVICHE/saved_models/modelsvgg (1).pt'
PATH = '/home/whale/Desktop/Rachel/CeVICHE/models/modelsvgg.pt'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 1
num_classes = 7
input_size = 224
phase = 'Evaluate'

thresh = 15
surround = 6
bubble_check = 120
if thresh % 2 != 0:
    thresh += 1
num = thresh*surround
half = thresh/2
half_num = num/2

##progress bar##################################################################
def progress(count, total, status=''):
    bar_len = 40
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

##DATASETS AND DATALOADERS################################################
def get_frames():
    os.chdir(vid_folder)
    train_names = sorted(glob('**/*/*.avi', recursive=True))
    train_labels = np.genfromtxt(label_file, delimiter=',', skip_header=2, usecols=range(0,20))
    print('labels:', train_labels.shape, 'names', len(train_names))

    for vid in names:
        cap = cv2.VideoCapture(vid)
        total_frames = int(round(cap.get(7)))
        fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
        row = (names.index(vid))
        seizing_worms = int(labels[row, 12])
        bubble = int(labels[row, 13])
        w1 = labels[row, 14]
        w2 = labels[row, 15]
        w3 = labels[row, 16]
        w4 = labels[row, 17]
        w5 = labels[row, 18]
        w6 = labels[row, 19]
        worms = [int(w1), int(w2), int(w3), int(w4), int(w5), int(w6)]
        check = np.zeros((6, (total_frames+1)))
        for idx, worm in enumerate(worms):
            if worm == 0:
                break
            for frame_index in range(bubble, worm):
                # print(frame_index)
                check[idx, frame_index] = 1
        worms_count = np.sum(check,axis=0)
        for idx, num in enumerate(worms_count):
            if idx >= 0 & idx <= total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES,idx)
                ret, frame = cap.read()
                # print(idx, type(frame))
                if type(frame) is not type(worms_count):
                    print('skipped', idx)
                    break
                # print(idx, num, frame.shape[2])
                frame = imresize(frame, (224,224,3))
                r = str(idx)
                imsave(save_path + vid + r + '.jpg', frame)
                if int(num) == 0:
                    imsave(save_path + vid + '0w/'+ r + '.jpg', frame)
                if int(num) == 1:
                    imsave(save_path + vid + '1w/'+ r + '.jpg', frame)
                if int(num) == 2:
                    imsave(save_path + vid + '2w/'+ r + '.jpg', frame)
                if int(num) == 3:
                    imsave(save_path + vid + '3w/'+ r + '.jpg', frame)
                if int(num) == 4:
                    imsave(save_path + vid + '4w/'+ r + '.jpg', frame)
                if int(num) == 5:
                    imsave(save_path + vid + '5w/'+ r + '.jpg', frame)
                if int(num) == 6:
                    imsave(save_path + vid + '6w/'+ r + '.jpg', frame)
        print('finished with vid:', vid)
        return(train_names, train_labels, total_frames, fps)
names, labels, total_frames, fps, vid = get_frames()

data_transforms = {
    phase: transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
print("Initializing Datasets and Dataloaders...")
image_datasets = {x: datasets.ImageFolder(os.path.join(save_path, x), data_transforms[x]) for x in [phase]}
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False, num_workers=4) for x in [phase]}

##TESTING PROCEDURE AND PRINTS################################################
def testing_model(model, dataloaders):
    since = time.time()
    seizing_preds = []
    inter = []
    smoothed =[0]
    t = 0
    idx = 0
    id = 1
    frame_bubble = []
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
        vis.line(seizing_preds, win='seizing_preds', opts=dict(title= 'Raw_Predictions'))
        worms_seizing_act.append(labels.data.cpu().numpy())
        seizing_preds = np.asarray(seizing_preds)

        def correct_15(idx, mode):
            if len(mode[:idx]) > half:
                max_15 = np.argmax(np.bincount(np.squeeze(mode[int(idx-half):int(idx+half+1)])))
            else:
                length = len(mode[:idx])
                max_15 = np.argmax(np.bincount(np.squeeze(mode[int(idx-length):int(idx+half+1)])))
            return max_15
        def correct_90(idx, mode):
            if len(mode[:idx]) > half_num:
                max_90 = np.argmax(np.bincount(np.squeeze(mode[int(idx-half_num):int(idx+half_num+1)])))
            else:
                length = len(mode[:idx])
                max_90 = np.argmax(np.bincount(np.squeeze(mode[int(idx-length):int(idx+half_num+1)])))
            return max_90
        def bubble(idx):
            if len(inter[:idx]) > bubble_check:
                max_bub = np.argmax(np.bincount(np.squeeze(inter[int(idx-bubble_check):int(idx+bubble_check+1)])))
            else:
                length = len(inter[:idx])
                max_bub = np.argmax(np.bincount(np.squeeze(inter[int(idx-length):int(idx+bubble_check+1)])))
            return max_bub

        if len(seizing_preds) > (thresh+half):
            inter.append(correct_15(idx, seizing_preds))
            idx+=1
            vis.line(inter, win='inter_preds', opts=dict(title= 'Intermediate_Predictions'))
        if len(inter) > bubble_check:
            bub = bubble(id)
            if bub >= 1:
                frame_bubble.append(t)
                if inter[id] > inter[int(id-1)]:
                    smoothed.append(inter[int(id-1)])
                    # print('4')
                elif inter[id] == inter[int(id-1)]:
                    smoothed.append(inter[id])
                    # print('5')
                elif inter[id] < inter[int(id-1)]:
                    smoothed.append(correct_90(id, smoothed))
                    # print('6')
                elif inter[id] > smoothed[int(id-1)]:
                    smoothed.append(smoothed[int(id-1)])
                vis.line(smoothed, win='smoothed_preds', opts=dict(title= 'Smoothed_Predictions'))
            else:
                # print(bub)
                if inter[id] > inter[int(id-1)]:
                    smoothed.append(correct_90(id, inter))
                    # print(4, correct_90(id, inter))
                elif inter[id] != smoothed[int(id-1)]:
                    if inter[id] > smoothed[int(id-1)]:
                        smoothed.append(correct_90(id, inter))
                        # print(5.5, correct_90(idx, inter), inter[id])
                    elif inter[id] < smoothed[int(id-1)]:
                        smoothed.append(smoothed[int(id-1)])
                    # print(5, correct_90(id, inter), inter[id])
                elif inter[id] < inter[int(id-1)]:
                    smoothed.append(correct_90(id, inter))
                    # print(6, correct_90(id, inter))
                elif inter[id] == smoothed[int(id-1)]:
                    smoothed.append(inter[id])
                    # print(7, inter[id])
                vis.line(smoothed, win='smoothed_preds', opts=dict(title= 'Smoothed_Predictions'))
            id += 1

        p = time.time() - s
        progress(t, total_frames, status=('predicting ' + str(t)))
        time.sleep(p)
        vis.line(worms_seizing_act, win='worms_seizing_target', opts=dict(title= 'trainedVGG-Target'))
        confusion[seizing_preds[-1], worms_seizing_act[-1]] += 1
        vis.heatmap(confusion, win='confusion_matrix', opts=dict(ylabel= 'predicted', xlabel= 'target', colormap= 'Electric'))

    testing_acc = (running_corrects/len(labels.data))
    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Testing_acc:', testing_acc.cpu().numpy())
    total_worms_seizing = max(smoothed)
    total_seizing_preds = len(seizing_preds)
    total_intermediate = len(inter)
    total_smoothed = len(smoothed)
    avg_seizing_preds = (sum(seizing_preds)/(total_seizing_preds/total_frames))/total_worms_seizing
    avg_intermediate = (sum(inter)/(total_intermediate/total_frames))/total_worms_seizing
    avg_smoothed = (sum(smoothed)/(total_smoothed/total_frames))/total_worms_seizing
    avg = ((avg_seizing_preds+avg_intermediate+avg_smoothed)/3)/30
    print('REPORT: ')
    print('Video Analyzed: ', (PATH + phase))
    print('Total Worms Seizing: ', total_worms_seizing)
    print('Average time Seizing: ', avg + ' seconds')
    print('Analysis Complete!')
    return test_model, testing_acc

##INITIALIZE AND RESHAPE MODEL################################################
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
