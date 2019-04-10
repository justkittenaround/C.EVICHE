#CeVICHE pre
#torch implementation of Resnet--> stacked bLSTM

#utils
import torch
from torchvision import datasets, transforms, models
import numpy as np
from glob import glob
import os
import cv2

#Hyperparameters
label_file = '/home/blu/C.EVICHE/data/Time2Seize - Sheet1 (1).csv'
vid_dir = '/home/blu/C.EVICHE/data/train'
timeDepth = 10000
channels = 3
wSize = 224
hSize = 224



#Data
class WormDataset():
    "videos of seizing and not seizing worms"
    def __init__(self, label_file, vid_dir, timeDepth, channels, wSize, hSize, transform=None):
        """
        Args:
            label_file (string): path to csv of labels
            vid_dir (string): directory of videos
            channels: Number of channels of frames
            timeDepth: Number of frames to be loaded in a sample
            wSize: width of frame
            hSize: height of frame
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.labels = np.genfromtxt(label_file, delimiter=',', skip_header=2, usecols=range(0,20))
        self.vid_dir = vid_dir
        self.timeDepth = timeDepth
        self.channels = channels
        self.wSize = wSize
        self.hSize = hSize
        self.transform = transform

    def __nvids__(self):
        return len(self.labels)
    def __len__(self):
        return len(self.labels)
    def readvids(self, vid):
        cap = cv2.VideoCapture(vid)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(5))
        nFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #property number 7
        frames = torch.FloatTensor(self.channels, self.timeDepth, self.wSize, self.hSize)
        print(vid, 'fps', fps, 'nframes', nFrames, 'width', w, 'height', h)
        failedClip = False
        for f in range(self.timeDepth):
             ret, frame = cap.read()
             if ret:
                 frame = torch.from_numpy(frame)
                 # HWC2CHW
                 frame = frame.permute(2, 0, 1)
                 frames[:, f, :, :] = frame

             else:
                print('skipped!')
                failedClip=True
                break
        for c in range(3):
            frames[c] -= self.mean[c]
        #normalize
        frames /= 255
        return frames, failedClip

    def __getitem__(self, idx):
        search_path = os.path.join(self.vid_dir, '**/*/*.avi')
        vid_names = sorted(glob(search_path, recursive=True))
        vid = vid_names[idx]
        clip, failedClip = self.readvids(vid)
        if self.transform:
            clip = self.transform(clip)
        sample = {'clip': clip, 'label': self.labels[idx][1], 'failedClip': failedClip}
        return sample

data = WormDataset(label_file, vid_dir, timeDepth, channels, wSize, hSize)
print('check:', len(data), data[0])
