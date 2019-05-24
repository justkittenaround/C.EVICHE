from torchvision import datasets, models, transforms
import torch
import torch.multiprocessing as mp
import torch.utils.data as data_utils
from torch.utils.data import Dataset
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


PATH = '/home/whale/Desktop/Rachel/CeVICHE/models/modelsvgg.pt'
root = '/home/whale/Desktop/Rachel/CeVICHE/Data/test/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_size = 224
thresh = 15
surround = 6


##initialize_model#############################################################
test_model= torch.load(PATH)
test_model.eval()
test_model= test_model.to(device)
##multiprocessing##############################################################
# if __name__ == '__main__':
#     nump_processes = 4
#     test_model = test_model()
#     test_model.share_memory()
#     for rank in range(num_processes):
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
    print('Total frames in ', target, ' is: ', total_frames)
    bin = np.array([3, total_frames])
    worms_seizing_pred = []
    worms_seizing_smoothed = []
    worms_seizing_smoothed_1 = []
    worms_seizing_smoothed_2 = []
    worms_seizing_smoothed_3 = []
    bin = np.array([3,1])
    init = 2
    count_1 = thresh + init
    count_2 = thresh + init
    idx = 0
    t = 0
    print("Running Predictions...")
    since = time.time()
    for n in range(0, total_frames):
        s = time.time()
        t = n
        cap.set(cv2.CAP_PROP_POS_FRAMES,n)
        ret, frame = cap.read()
        if type(frame) is not type(bin):
            print('skipped', n)
            break
        frame = pil.fromarray(frame)
        frame = data_transforms(frame)
        frame = frame.unsqueeze(0)
        input = frame.to(device)
        outputs = test_model(input)
        _, predicts = torch.max(outputs, 1)
        preds = predicts.cpu().numpy()
        seizing_preds = np.asarray(worms_seizing_pred)
        worms_seizing_pred.append(preds)
        vis.line(worms_seizing_pred, win='worms_seizing_predict', opts=dict(title= 'trainedVGG-Predictions'))
        if len(seizing_preds) == count_1:
            tally_1 = np.bincount(np.squeeze(seizing_preds[(count_1-thresh):]))
            max_1 = np.argmax(tally_1)
            worms_seizing_smoothed_1.append(max_1)
            count_1 += thresh
            vis.line(worms_seizing_smoothed_1, win='worms_seizing_smoothed_1', opts=dict(title= 'smoothed_predictions_block'))
        if len(seizing_preds) == count_2:
            tally_2 = np.bincount(np.squeeze(seizing_preds[idx:]))
            max_2 = np.argmax(tally_2)
            worms_seizing_smoothed_2.append(max_2)
            vis.line(worms_seizing_smoothed_2, win='worms_seizing_smoothed_2', opts=dict(title= 'smoothed_predictions_after'))
            idx += 1
            count_2 += 1
        if idx >= (thresh*surround):
            position = idx - (thresh*surround)
            tally_3 = np.bincount(np.squeeze(seizing_preds[position:]))
            max_3 = np.argmax(tally_3)
            worms_seizing_smoothed_3.append(max_3)
            vis.line(worms_seizing_smoothed_3, win='worms_seizing_smoothed_3', opts=dict(title= 'smoothed_predictions_surround'))
            print(tally_1.shape, tally_2.shape, tally_3.shape)
            for n in range(0, total_frames):
                a = np.append([tally_1[n]], [tally_2[n]], axis=1)
                print(a)
                a = np.append([a], [tally_3[n]], axis=1)
                tally = np.bincount(np.squeeze(a[:, n]), axis=0)
                max = np.argmax(tally)
                worms_seizing_smoothed.append(max)
                vis.line(worms_seizing_smoothed, win='worms_seizing_smoothed', opts=dict(title= 'smoothed_predictions_combined'))

        p = time.time() - s
        progress(n, total_frames, status=('predicting ' + str(n)))
        time.sleep(p)
    time_elapsed = time.time() - since
    print('Testing completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60), 'for ' + str(target))
