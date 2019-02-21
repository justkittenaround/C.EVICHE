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

vid = '/home/whale/Desktop/Rachel/CeVICHE/10-26 xtra tracks/019/10-26-worms-019.avi'

##
os.chdir('/home/whale/Desktop/Rachel/CeVICHE/train/')
names = glob('**/*/*.avi', recursive=True)

def get_frames(vid):
    cap = cv2.VideoCapture(vid)
    fps = cap.get(5)
    stack = np.zeros([1, 224, 256, 3])
    while(True):
         ret, frame = cap.read()
         idx = int(cap.get(1))
         frame = np.resize(frame, (idx, 224, 256, 3))
         stack = np.concatenate([stack, frame], axis=0)
         if cv2.waitKey(1):
            break
    cap.release()
    cv2.destroyAllWindows()
    return stack

def scrape(names, label_file):
    data = np.zeros([len(names), 480, 224, 256, 3])
    for idx, vid in enumerate(names):
        frames = get_frames(vid)
        # frames = frames.mean(axis=-1, keepdims=1)
        frames = np.resize(frames, (224, 256, 3))
        data[idx, ...] = frames
        labels = np.genfromtxt(label_file, delimiter=',', skip_header=1, usecols=range(0,20))
        labels = labels[:, 12:]
    return data, labels


###run
x, y = scrape(names, '/home/whale/Desktop/Rachel/CeVICHE/Time2Seize - Sheet1 (1).csv')

#normalize
x_mean = np.mean(x, 0)
x -= x_mean


##model
#hyperparameters
epochs = 3
batch_size = 1
optimizer = 'adam'
activation = 'relu'
loss = 'categorical_crossentropy'

##model
model = Sequential()
model.add(resnet50.ResNet50(weights='imagenet', include_top=False, input_shape='256,256,1'))
model.add(keras.layers.ConvLSTM2D(512,(3,3), strides=(1, 1), dilation_rate=(1, 1), activation=activation, recurrent_activation=activation, go_backwards=False)
model.add(keras.layers.ConvLSTM2D(512,(3,3), strides=(1, 1), dilation_rate=(1, 1), activation=activation, recurrent_activation=activation, go_backwards=True)
# model.add(keras.layers.LSTM(2, activation=activation, recurrent_activation=activation)
model.add(keras.layers.GlobalAveragePooling2D(data_format=None))
model.add(Dense(2, activation=(keras.activations.softmax(x, axis=-1)))

model.compile(loss= loss,
              optimizer=‘adam’,
              metrics=['accuracy'])

model = model()

##train
model.fit(x, y, epochs=epochs, batch_size=batch_size)

##evaluate
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=batch_size)

##test
classes = model.predict(x_test, batch_size=batch_size)
