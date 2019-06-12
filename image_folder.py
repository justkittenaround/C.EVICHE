import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import skimage
from scipy.misc import imsave, imresize

label_file = '/home/whale/Desktop/Rachel/CeVICHE/Time2Seize - Sheet1 (3) - Time2Seize - Sheet1 (3).csv'
vid_folder = '/home/whale/Desktop/Rachel/CeVICHE/Data/train/'
save_path = '/home/whale/Desktop/Rachel/CeVICHE/conv_ceviche_data/Evaluate/'

#label_file = '/home/blu/C.EVICHE/Time2Seize - Sheet1 (3).csv'
#vid_folder = '/home/blu/C.EVICHE/data/train'
#save_path = '/home/blu/C.EVICHE/data/conv_ceviche_data/train/'

def get_data():
    os.chdir(vid_folder)
    train_names = sorted(glob('**/*/*.avi', recursive=True))
    vid_names = []
    for name in train_names:
        a = name.split('/')
        name = a[-1]
        vid_names.append(name)
    train_labels = np.genfromtxt(label_file, delimiter=',', skip_header=2, usecols=range(0,20))
    print('labels:', train_labels.shape, 'names', len(train_names))
    return(train_names, vid_names, train_labels)

names, vid_names, labels = get_data()

##for training##################################################################
id = 0
for vid in names:
    cap = cv2.VideoCapture(vid)
    total_frames = int(round(cap.get(7)))
    video = vid_names[id]
    folders = os.listdir(save_path)
    if video not in folders:
        final = (save_path + str(video) + '.' + str(total_frames) + '/')
        os.mkdir(final)
        print('made new folders in ', save_path)
    else:
        print('good')
    folders2 = os.listdir(final)
    for bit in ['0w', '1w', '2w', '3w', '4w', '5w', '6w']:
        if bit not in folders2:
            os.mkdir(final + bit)
        else:
            print('good')
    id += 1
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
        elif worm >= total_frames:
            worm = (total_frames - 1)
        for frame_index in range(bubble, worm):
            # print(frame_index)
            check[idx, frame_index] = 1
    worms_count = np.sum(check,axis=0)
    # print(worms_count)
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
            if int(num) == 0:
                imsave(final + '0w/'+ r + '.jpg', frame)
            if int(num) == 1:
                imsave(final + '1w/'+ r + '.jpg', frame)
            if int(num) == 2:
                imsave(final + '2w/'+ r + '.jpg', frame)
            if int(num) == 3:
                imsave(final + '3w/'+ r + '.jpg', frame)
            if int(num) == 4:
                imsave(final + '4w/'+ r + '.jpg', frame)
            if int(num) == 5:
                imsave(final + '5w/'+ r + '.jpg', frame)
            if int(num) == 6:
                imsave(final + '6w/'+ r + '.jpg', frame)

    print('finished with vid:', video)


##for testing##################################################################
# vid_folder = '/home/whale/Desktop/Rachel/CeVICHE/Data/train/'
# save_path = '/home/whale/Desktop/Rachel/CeVICHE/conv_ceviche_data/Evaluate/'
# vid = names[38]
# for idx in range(0, total_frames):
#     cap.set(cv2.CAP_PROP_POS_FRAMES,idx)
#     ret, frame = cap.read()
#     if type(frame) is not type(worms_count):
#         print('skipped', idx)
#         break
#     frame = imresize(frame, (224,224,3))
#     r = str(idx)
#     imsave(save_path + r + '.jpg', frame)
# print('done')
