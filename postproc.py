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
vis = visdom.Visdom()
PATH = '/home/blu/C.EVICHE/saved_models/modelsvgg (1).pt'
root = '/home/blu/C.EVICHE/data/test/'
# PATH = '/home/whale/Desktop/Rachel/CeVICHE/models/modelsvgg.pt'
# root = '/home/whale/Desktop/Rachel/CeVICHE/Data/test/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_size = 224
thresh = 15
surround = 6
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
for target in sorted(files):
    d = os.path.join(root, target)
    if not os.path.isfile(d):
        continue
    cap = cv2.VideoCapture(d)
    total_frames = int(round(cap.get(7)))
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
    print('Total frames in ', target, ' is: ', total_frames)
    seizing_preds = []
    smoothed =[]
    t = 0
    idx = 1
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
        def correct_up(idx):
            if len(seizing_preds[:idx]) > half:
                max_15 = np.argmax(np.bincount(np.squeeze(seizing_preds[int(idx-half):int(idx+half+1)])))
            else:
                length = len(seizing[:idx])
                max_15 = np.argmax(np.bincount(np.squeeze(seizing_preds[int(idx-length):int(idx+half+1)])))
            return max_15
        def correct_down(idx):
            if len(seizing_preds[:idx]) > half_num:
                max_90 = np.argmax(np.bincount(np.squeeze(seizing_preds[int(idx-half_num):int(idx+half_num+1)])))
            else:
                length = len(seizing[:idx])
                max_90 = np.argmax(np.bincount(np.squeeze(seizing_preds[int(idx-length):int(idx+half_num+1)])))
            return max_90
------------------------------------insert if >/< but also == to below------------------------------------------------
        while len(seizing_preds[idx:]) > half_num:
            if seizing_preds[idx] == seizing_preds[int(idx-1)] and seizing_preds[idx] == seizing_preds[int(idx+1)]:
                smoothed.append(seizing_preds[idx])
            elif seizing_preds[idx] > seizing_preds[int(idx-1)] and seizing_preds[idx] > seizing_preds[int(idx+1)]:
                smoothed.append(correct_up(idx))
            elif seizing_preds[idx] > seizing_preds[int(idx-1)] and seizing_preds[idx] < seizing_preds[int(idx+1)]:
                smoothed.append(correct_up(idx))
            elif seizing_preds[idx] < seizing_preds[int(idx-1)] and seizing_preds[idx] < seizing_preds[int(idx+1)]:
                smoothed.append(correct_down(idx))
            elif seizing_preds[idx] < seizing_preds[int(idx-1)] and seizing_preds[idx] > seizing_preds[int(idx+1)]:
                smoothed.append(correct_down(idx))
            else:
                smoothed.append(seizing_preds[idx])
                print('missed scenario')
            idx +=1
            vis.line(smoothed, win='smoothed_preds', opts=dict(title= 'Smoothed_Predictions'))
        p = time.time() - s
        progress(t, total_frames, status=('predicting ' + str(t)))
        time.sleep(p)
    time_elapsed = time.time() - since
    print('Testing completed in {:.0idx}m {:.0idx}s'.idxormat(time_elapsed // 60, time_elapsed % 60), 'idxor ' + str(target))
