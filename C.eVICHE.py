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
os.chdir('/home/whale/Desktop/Rachel/CeVICHE/')
names = glob('**/*/*.avi', recursive=True)

def get_frames(vid):
  cap = cv2.VideoCapture(vid)
  ret, frame = cap.read()
  return frame

if len(names) == 166:
  del(names[113])
else:
  pass

def scrape(names):
  data = np.zeros([len(names), 224, 256, 1])
  for idx, vid in enumerate(names):
    frames = get_frames(vid)
    frames = frames.mean(axis=-1, keepdims=1)
    frames = np.resize(frames, (224, 256, 1))
    data[idx, ...] = frames
  return data

data = scrape(names)
