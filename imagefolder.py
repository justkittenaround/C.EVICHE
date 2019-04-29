

import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import skimage
import imageio

#vid = '/home/blu/C.EVICHE/data/train/10-26-worms/avi/010/10-26-worms-010.avi'
# label_file = '/home/whale/Desktop/Rachel/CeVICHE/Time2Seize - Sheet1 (3).csv'
label_file = '/home/blu/C.EVICHE/Time2Seize - Sheet1 (3).csv'
def get_train():
    os.chdir('/home/blu/C.EVICHE/data/train')
    train_names = sorted(glob('**/*/*.avi', recursive=True))
    train_labels = np.genfromtxt(label_file, delimiter=',', skip_header=2, usecols=range(0,20))
    print('labels:', train_labels.shape, 'names', len(train_names))
    return(train_names, train_labels)
names, labels = get_train()
for vid in names:
    cap = cv2.VideoCapture(vid)
    total_frames = int(round(cap.get(7)))
    row = names.index(vid) - 1
    seizing_worms = int(labels[row, 12])
    bubble = int(labels[row, 13])
    w1 = labels[row, 14]
    w2 = labels[row, 15]
    w3 = labels[row, 16]
    w4 = labels[row, 17]
    w5 = labels[row, 18]
    w6 = labels[row, 19]
    worms = [int(w1), int(w2), int(w3), int(w4), int(w5), int(w6)]
    check = np.zeros((5, (total_frames+1)))
    for idx, worm in enumerate(worms):
        if worm == 0:
            break
        for frame_index in range(bubble, worm):
            # print(frame_index)
            check[idx, frame_index] = 1
            # if idx > check.shape[0]:
            #     print('idx bigger than check array')
            # if frame_index > check.shape[1]:
            #     print('frame idx bigger than check array')
            # if idx == check.shape[0]:
            #     print('idx same as check array')
            # if frame_index == check.shape[1]:
            #     print('frame idx same as check array')
    worms_count = np.sum(check,axis=0)
    for idx, num in enumerate(worms_count):
        frames = np.zeros([1,224,256,3])
        if idx >= 0 & idx <= total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES,idx)
        else:
            print('idx not in total_frames')
            break
        while(True):
             ret, frame = cap.read()
             frame = skimage.transform.resize(frame, (224,256,3))
             frames[0, ...] = frame
             if cv2.waitKey(1):
                 break
             cap.release()
             cv2.destroyAllWindows()
        frames = np.squeeze(frames)
        # print('frame_shape:', frames.shape)
        r = str(np.random.randint(99999999))
        # print(idx, num)
        if int(num) == 0:
            imageio.imwrite('/home/blu/C.EVICHE/data/conv_ceviche_data/train/0w/'+ r + '.jpg', frames.astype(np.uint8))
        if int(num) == 1:
            imageio.imwrite('/home/blu/C.EVICHE/data/conv_ceviche_data/train/1w/'+ r + '.jpg', frames.astype(np.uint8))
        if int(num) == 2:
            imageio.imwrite('/home/blu/C.EVICHE/data/conv_ceviche_data/train/2w/'+ r + '.jpg', frames.astype(np.uint8))
        if int(num) == 3:
            imageio.imwrite('/home/blu/C.EVICHE/data/conv_ceviche_data/train/3w/'+ r + '.jpg', frames.astype(np.uint8))
        if int(num) == 4:
            imageio.imwrite('/home/blu/C.EVICHE/data/conv_ceviche_data/train/4w/'+ r + '.jpg', frames.astype(np.uint8))
        if int(num) == 5:
            imageio.imwrite('/home/blu/C.EVICHE/data/conv_ceviche_data/train/5w/'+ r + '.jpg', frames.astype(np.uint8))
        if int(num) == 6:
            imageio.imwrite('/home/blu/C.EVICHE/data/conv_ceviche_data/train/6w/'+ r + '.jpg', frames.astype(np.uint8))
    print('finished with vid:', vid)

print('meowzers')
