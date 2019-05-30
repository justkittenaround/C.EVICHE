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

#PATH = '/home/blu/C.EVICHE/saved_models/modelsvgg (1).pt'
#root = '/home/blu/C.EVICHE/data/test/'
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
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
    print('Total frames in ', target, ' is: ', total_frames)
    worms_seizing_pred = []
    worms_seizing_smoothed = []
    worms_seizing_smoothed_1 = []
    worms_seizing_smoothed_2 = []
    worms_seizing_smoothed_3 = []
    init = 1
    count_1 = thresh
    count_2 = thresh
    idx = 0
    t = 0
    check_type = np.array([1])
    p = np.asarray([0])
    print("Running Predictions...")
    since = time.time()
    for t in range(0, total_frames):
        s = time.time()
        cap.set(cv2.CAP_PROP_POS_FRAMES,t)
        ret, frame = cap.read()
        if type(frame) is not type(check_type):
            print('skipped', t)
            continue
        frame = pil.fromarray(frame)
        frame = data_transforms(frame)
        frame = frame.unsqueeze(0)
        input = frame.to(device)
        outputs = test_model(input)
        _, predicts = torch.max(outputs, 1)
        preds = predicts.cpu().numpy()
        seizing_preds = np.asarray(worms_seizing_pred)
        worms_seizing_pred.append(preds)
        vis.line(worms_seizing_pred, win='worms_seizing_predict', opts=dict(title= 'Raw_Predictions_' + str(round(fps/fps)) + 'sec'))

        if len(seizing_preds) == count_1:
            leng_1 = seizing_preds[(count_1-thresh):]
            max_1 = np.argmax(np.bincount(np.squeeze(leng_1)))
            worms_seizing_smoothed_1.append(max_1)
            count_1 += thresh
            vis.line(worms_seizing_smoothed_1, win='worms_seizing_smoothed_1', opts=dict(title= 'smoothed_' + str(round(fps/len(leng_1))) + 'sec_block'))

        if len(seizing_preds) == count_2:
            leng_2 = seizing_preds[idx:]
            max_2 = np.argmax(np.bincount(np.squeeze(leng_2)))
            worms_seizing_smoothed_2.append(max_2)
            vis.line(worms_seizing_smoothed_2, win='worms_seizing_smoothed_2', opts=dict(title= 'smoothed_' + str(round(fps/len(leng_2))) + 'sec_slide'))
            idx += 1
            count_2 += 1

        if idx >= (thresh*surround):
            num = thresh*surround
            position = idx - (num)
            leng_3 = seizing_preds[position:]
            max_3 = np.argmax(np.bincount(np.squeeze(leng_3)))
            worms_seizing_smoothed_3.append(max_3)
            vis.line(worms_seizing_smoothed_3, win='worms_seizing_smoothed_3', opts=dict(title= 'smoothed_' + str(round(len(leng_3)/fps)) + 'sec_surround'))

        p = time.time() - s
        progress(t, total_frames, status=('predicting ' + str(t)))
        time.sleep(p)

    print('lengths of time', len(leng_1), len(leng_3))
    print('smoothed1', len(worms_seizing_smoothed_1), 'smoothed3', len(worms_seizing_smoothed_3))

    time_elapsed = time.time() - since
    print('Testing completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60), 'for ' + str(target))