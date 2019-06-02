import torch
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
vis = visdom.Visdom(port=8090)
# PATH = '/home/blu/C.EVICHE/saved_models/modelsvgg (1).pt'
# root = '/home/blu/C.EVICHE/data/test/'
PATH = '/home/whale/Desktop/Rachel/CeVICHE/models/modelsvgg.pt'
root = '/home/whale/Desktop/Rachel/CeVICHE/Data/test/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_size = 224
thresh = 15
surround = 6
bubble_check = 120
if thresh % 2 != 0:
    thresh += 1

num = thresh*surround
half = thresh/2
half_num = num/2
##initialize_model#############################################################
test_model= torch.load(PATH)
test_model.eval()
test_model= test_model.to(device)
##multiprocessing##############################################################
# if __name__ == '__main__':
#     nump_processes = 4
#     test_model = test_model()
#     test_model.share_memory()
#     idxor rank in range(num_processes):
#         p = mp.Process(target=testing_model, args=(model,))
#         p.start()
#         processes.append(p)
#     for p in processes:
#         p.join()
##progress bar##################################################################
def progress(count, total, status=''):
    bar_len = 40
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()
##TESTING PROCEDURE AND PRINTS#################################################
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
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
    print('Total frames in ', target, ' is: ', total_frames)
    seizing_preds = []
    inter = []
    smoothed =[0]
    frames_bin= []
    t = 0
    idx = 0
    id = 1
    frame_bubble = []
    check_type = np.array([1])
    print("Running Predictions...")
    since = time.time()
    for t in range(0, total_frames):
        s = time.time()
        cap.set(cv2.CAP_PROP_POS_FRAMES,t)
        ret, frame = cap.read()
        if type(frame) is not type(check_type):
            print('skipped', t)
            break
        frame = pil.fromarray(frame)
        frame = data_transforms(frame)
        frame = frame.unsqueeze(0)
        input = frame.to(device)
        outputs = test_model(input)
        _, predicts = torch.max(outputs, 1)
        preds = predicts.cpu().numpy()
        preds = int(preds)
        seizing_preds.append(preds)
        vis.line(seizing_preds, win='seizing_preds', opts=dict(title= 'Raw_Predictions'))
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
#------------------------------------insert if >/< but also == to below------------------------------------------------
        if len(seizing_preds) < (thresh+half):
            frames_bin.append(t)
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
            frames_bin.append(t)
        p = time.time() - s
        progress(t, total_frames, status=('predicting ' + str(t)))
        time.sleep(p)
        total_seizing_preds = len(seizing_preds)
        total_intermediate = len(inter)
        total_smoothed = len(smoothed)
    time_elapsed = time.time() - since
    print('Testing completed in {:.0f}m {:.0f}'.format(time_elapsed // 60, time_elapsed % 60), 'for ' + str(target))
    total_worms_seizing = max(smoothed)
    total_seizing_preds = len(seizing_preds)
    total_intermediate = len(inter)
    total_smoothed = len(smoothed)
    avg_seizing_preds = (sum(seizing_preds)/(total_seizing_preds/total_frames))/total_worms_seizing
    avg_intermediate = (sum(inter)/(total_intermediate/total_frames))/total_worms_seizing
    avg_smoothed = (sum(smoothed)/(total_smoothed/total_frames))/total_worms_seizing
    avg = ((avg_seizing_preds+avg_intermediate+avg_smoothed)/3)/30

    print('REPORT: ')
    print('Video Analyzed: ', target)
    print('Total Worms Seizing: ', total_worms_seizing)
    print('Average time Seizing: ', avg + ' seconds')

    if target != folder[-1]:
        print('Initializing Next Video...')
    else:
        print('Analysis Complete!')
