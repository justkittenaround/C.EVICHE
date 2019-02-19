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

#vid = '/home/whale/Desktop/Rachel/CeVICHE/10-26-worms/avi/001/10-26-worms-001.avi'

##
os.chdir('/home/whale/Desktop/Rachel/CeVICHE/train/')
names = glob('**/*/*.avi', recursive=True)

def get_frames(vid):
  cap = cv2.VideoCapture(vid)
  ret, frame = cap.read()
  return frame

def clean(ind, maxlen):
  if len(names) == maxlen:
    del(names[ind])
  else:
    pass

def scrape(names, label_file):
  data = np.zeros([len(names), 224, 256, 1])
  for idx, vid in enumerate(names):
    frames = get_frames(vid)
    frames = frames.mean(axis=-1, keepdims=1)
    frames = np.resize(frames, (224, 256, 1))
    data[idx, ...] = frames
    labels = np.genfromtxt(label_file, delimiter=',', missing_values='-', skip_header=1, filling_values=0, usecols=range(0,5))
    labels = labels[:, 1:]
  return data, labels

###run###
x, y = scrape(names, '/home/whale/Desktop/Rachel/CeVICHE/Time2Seize.csv')
###

x_mean = np.mean(x, 0)
x -= x_mean
