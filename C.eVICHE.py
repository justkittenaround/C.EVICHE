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
from keras import optimizers
from keras import backend as K

label_file = '/home/default/C.EVICHE/Time2Seize - Sheet1 (1).csv'
vid = '/home/default/C.EVICHE/train/10-26-worms/avi/001/10-26-worms-001.avi'

os.chdir('/home/default/C.EVICHE/train')
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
x = tf.convert_to_tensor(x)
y = tf.convert_to_tensor(y)

##model
#hyperparameters
epochs = 3
batch_size = 3
optimizer = 'adam'
act = 'tanh'
loss = 'categorical_crossentropy'
dialation_rate = (3, 3)

##model
def create_network():
    res = Sequential()
    res.add(ResNet50(weights='imagenet', include_top=False, input_shape=(256,256,3)))
    res.add(keras.layers.Embedding(4, 5))
    res.add(keras.layers.Flatten(data_format=None))
    # model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
    model = Sequential()
    model.add(keras.layers.TimeDistributed(res))
    model.add(keras.layers.ConvLSTM2D(5,(1,1), strides=(1, 1), dilation_rate=dialation_rate, go_backwards=False))
    model.add(keras.layers.ConvLSTM2D(5,(1,1), strides=(1, 1), dilation_rate=dialation_rate, go_backwards=True))
    model.add(keras.layers.GlobalAveragePooling2D(data_format=None))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss= loss,optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),metrics=['accuracy'])
    return model
network_model = create_network()
model.fit(x, y, epochs=epochs, batch_size=batch_size, shuffle=False)


x = np.zeros((618,256,256,3))
y = np.zeros((618))



##evaluate
# score = model.evaluate(x_test, y_test, batch_size=batch_size)

##test
# classes = model.predict(x_test, batch_size=batch_size)
