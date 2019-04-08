
import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import tensorflow as tf
import keras
from keras.models import load_model
from keras.utils import multi_gpu_model
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Embedding
from keras.layers import ConvLSTM2D
from keras.layers import Dense
from keras.layers import Activation
from keras import losses
from keras import optimizers
from keras import backend as K
import skimage
from keras.callbacks import TensorBoard
from time import time
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="1";



batch_size = 32
label_file = '/home/whale/Desktop/Rachel/CeVICHE/Time2Seize - Sheet1 (2).csv'

def get_test():
    os.chdir('/home/whale/Desktop/Rachel/CeVICHE/test')
    test_names = sorted(glob('**/*/*.avi', recursive=True))
    test_labels = np.genfromtxt(label_file, delimiter=',', skip_header=2, usecols=range(0,20), missing_values = ' ', filling_values = 0, deletechars='nan')
    test_labels = test_labels[35:53, :]
    return(test_names, test_labels)

test_names, test_labels = get_test()

n = 300

def get_frames(vid):
    cap = cv2.VideoCapture(vid)
    fps = int(cap.get(5))
    total_frames = int(round(cap.get(7)))
    print(total_frames, vid)
    all_frames = np.zeros([n,224,256,3])
    count = 0
    ind = np.random.choice(total_frames, n, replace=False)
    while(True):
         ret, frame = cap.read()
         frame = skimage.transform.resize(frame, (224,256,3))
         print(frame.shape)
         if count in ind:
            all_frames[count, ...] = frame
         count += 1
         if cv2.waitKey(1):
            break
    cap.release()
    cv2.destroyAllWindows()
    return(all_frames, total_frames, ind)

def get_hot(vid, names, labels):
    get, total_frames, ind = get_frames(vid)
    row = names.index(vid)
    seizing_worms = int(labels[row, 13])
    bubble = int(labels[row, 14])
    w1 = labels[row, 15]
    w2 = labels[row, 16]
    w3 = labels[row, 17]
    w4 = labels[row, 18]
    w5 = labels[row, 19]
    w6 = labels[row, -1]
    worms = [int(w1), int(w2), int(w3), int(w4), int(w5), int(w6)]
    hot = np.zeros((7, total_frames))
    for idx, worm in enumerate(worms):
        for i in range(bubble, worm):
            if i  in ind:
                hot[idx, i] = 1
                if hot[idx, i] ==1:
                    hot[idx, 7] = 1
    hot = hot[:, ind]
    seize_time_frame = np.sum(hot,axis=0)
    return(get, seize_time_frame)

def trans(x):
    x_mean = np.mean(x, 0)
    x_mean = x_mean.astype('uint8', copy=False)
    x -= x_mean
    return(x)

def test_it(test_vid, current):
    x_test, y_test = get_hot(test_vid, test_names, test_labels)
    x_test = trans(x_test)
    res = load_model('/home/whale/Desktop/saved_res2.h5')
    predictions = res.predict(x_test)
    print('First Prediction:', predictions[0])
    score = res.evaluate(x_test, y_test, batch_size=70, verbose=0)
    print('Test accuracy:', score)
    return(x_test)

# def visualize(current, x_test):
#     print(x_test.shape)
#     show = x_test[100, :, :, :]
#     plt.imshow(show)
#     plt.show()
#     plt.savefig(current, show)

def test_loop():
    for current, test_vid in enumerate(test_names):
        go = test_it(test_vid, current)
        # a = visualize(current, go)
        print(go)
        return(go)

testing = test_loop()
