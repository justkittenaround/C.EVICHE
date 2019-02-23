

import cv2
import numpy as np
import os
from glob import glob
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0";

import tensorflow as tf
import keras
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

label_file = '/home/whale/Desktop/Rachel/CeVICHE/Time2Seize - Sheet1 (2).csv'

def get_train():
    os.chdir('/home/whale/Desktop/Rachel/CeVICHE/train')
    train_names = sorted(glob('**/*/*.avi', recursive=True))
    train_labels = np.genfromtxt(label_file, delimiter=',', skip_header=2, usecols=range(0,20))
    return(train_names, train_labels)

def get_test():
    os.chdir('/home/whale/Desktop/Rachel/CeVICHE/test')
    test_names = sorted(glob('**/*/*.avi', recursive=True))
    return(test_names)

train_names, train_labels = get_train()
# test_names = get_test()

def get_frames(vid):
    cap = cv2.VideoCapture(vid)
    fps = int(cap.get(5))
    total_frames = int(round(cap.get(7)))
    all_frames = np.asarray([])
    while(True):
         ret, frame = cap.read()
         idx = int(round(cap.get(1)))
         frame = np.resize(frame, (idx, 256, 256, 3))
         all_frames = np.concatenate([frame for idx in range(total_frames)], axis=0)
         if cv2.waitKey(1):
            break
    cap.release()
    cv2.destroyAllWindows()
    return(all_frames, total_frames)

def get_hot(vid, names, labels):
    get, cols = get_frames(vid)
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
    hot = np.zeros((6, cols))
    for idx, worm in enumerate(worms):
        for n in range(bubble, worm):
            hot[idx, n] = 1
    seize_time_frame = np.sum(hot,axis=0)
    if seizing_worms > 0:
        seize_time_frame = seize_time_frame/seizing_worms
    else:
        pass
    return(get, seize_time_frame)

def trans(x):
    x_mean = np.mean(x, 0)
    x_mean = x_mean.astype('uint8', copy=False)
    x -= x_mean
    return(x)

epochs = 1
batch_size = 3
opt = 'adam'
act = 'elu'
soft = 'softmax'
loss = 'mean_squared_error'
dialation_rate = (1, 1)
units = 100

def create_network():
    res = Sequential()
    res.add(ResNet50(weights='imagenet', include_top=False, input_shape=(256,256,3)))
    res.add(keras.layers.GlobalMaxPooling2D(data_format=None))
    res.add(keras.layers.Dense((1), activation=soft))
    res.compile(loss=loss,optimizer=opt)
    lstm = Sequential()
    lstm.add(keras.layers.LSTM(units, activation='tanh', recurrent_activation='hard_sigmoid', go_backwards=False))
    lstm.add(keras.layers.LSTM(units, activation='tanh', recurrent_activation='hard_sigmoid', go_backwards=True))
    lstm.add(keras.layers.Dense(1, activation=soft))
    lstm.compile(loss=loss,optimizer=opt)
    return(res, lstm)

def run_it(train_vid):
    x_train, y_train = get_hot(train_vid, train_names, train_labels)
    # x_test = get_frames(test_vid)
    x_train = trans(x_train)
    # x_test = trans(x_test)

    res, lstm = create_network()

    res.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=False)
    score = res.evaluate(x_train, y_train, batch_size=batch_size)
    classes = res.predict(x_train, batch_size=batch_size)
    classes = np.asarrray(classes)
    classes = np.resize(classes, (batch_size, 1, 1))

    model = lstm.fit(classes, y_train, epochs=epochs, batch_size=batch_size, shuffle=False)
    score = lstm.evaluate(classes, y_train, batch_size=batch_size)
    # final = lstm.predict(x_test, batch_size=batch_size)
    return(final, model, score)

def train_loop():
    win = np.asarray([])
    for train_vid in train_names:
        final = run_it(train_vid)
        win = np.concatenate([final], axis=0)
    return(win, model, score)

win, model, score = train_loop()
