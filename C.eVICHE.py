#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 12:35:08 2018
@author: Rachel St Clair
"""

###C.eVICHE###
##C.elegan Visual Implementation in Computer Heuristic Experiments##
##############

import cv2
import numpy as np
import os
from glob import glob
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0";

import keras
from keras.applications import resnet50
from keras.models import Sequential
from keras.layers import ConvLSTM2D

label_file = '/home/whale/Desktop/Rachel/CeVICHE/Time2Seize - Sheet1 (1).csv'
vid = '10-26-worms/avi/015/10-26-worms-015.avi'
# vid ='/home/default/Downloads/DJI_0009 Shark cut 10.avi'

os.chdir('/home/whale/Desktop/Rachel/CeVICHE/train')
names = glob('**/*/*.avi', recursive=True)
labels = np.genfromtxt(label_file, delimiter=',', skip_header=1, usecols=range(0,20))

def get_frames(vid):
    cap = cv2.VideoCapture(vid)
    fps = int(cap.get(5))
    total_frames = int(round(cap.get(7)))
    all_frames = np.asarray([])
    while(True):
         ret, frame = cap.read()
         idx = int(round(cap.get(1)))
         frame = frame.resize(256,256,3)
         frame = np.resize(frame, (idx, 256, 256, 3))
         all_frames = np.concatenate([frame for idx in range(total_frames)], axis=0)
         if cv2.waitKey(1):
            break
    cap.release()
    cv2.destroyAllWindows()
    return(all_frames, total_frames)

def get_hot(vid):
    get, cols = get_frames(vid)
    row = names.index(vid)
    bubble = int(labels[row, 13])
    w1 = labels[row, 14]
    w2 = labels[row, 15]
    w3 = labels[row, 16]
    w4 = labels[row, 17]
    w5 = labels[row, 18]
    w6 = labels[row, 19]
    worms = [int(w1), int(w2), int(w3), int(w4), int(w5), int(w6)]
    seizing_worms = int(labels[row, 12])
    hot = np.zeros((6, cols))
    for idx, worm in enumerate(worms):
        duration = bubble - worm
        for n in range(duration):
            hot = np.put(hot, [idx, n], 3)
    seize_time_frame = np.sum(hot,axis=0)
    if seizing_worms == 0:
        pass
    if seizing_worms > 0:
        seize_time_frame = seize_time_frame/seizing_worms
    return(get, seize_time_frame)

x, y = get_hot(vid)

#transformations
x = x.astype('float32')
x_mean = np.mean(x, 0)
x -= x_mean

##model
#hyperparameters
epochs = 3
batch_size = 3
optimizer = 'adam'
act = 'tanh'
loss = 'categorical_crossentropy'
dialation_rate = (3, 3)

##model
model = Sequential()
model.add(resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(256,256,3)))
model.add(keras.layers.ConvLSTM2D(512,(3,3), strides=(1, 1), dilation_rate=dialation_rate, go_backwards=False)
model.add(keras.layers.ConvLSTM2D(512,(3,3), strides=(1, 1), dilation_rate=dialation_rate, go_backwards=True)
# model.add(keras.layers.LSTM(2, activation=activation, recurrent_activation=activation)
model.add(keras.layers.GlobalAveragePooling2D(data_format=None))
model.add(Dense(1, activation=(keras.activations.softmax(x, axis=-1))))

model.compile(loss= loss,
              optimizer=‘adam’,
              metrics=['accuracy'])

##train
model.fit(x, y, epochs=epochs, batch_size=batch_size, shuffle=False)

##evaluate
# score = model.evaluate(x_test, y_test, batch_size=batch_size)

##test
# classes = model.predict(x_test, batch_size=batch_size)
