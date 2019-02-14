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

folders = '/home/whale/Desktop/Rachel/CeVICHE/10-26-worms/avi/', '/home/whale/Desktop/Rachel/CeVICHE/11-14 WORMS', '/home/whale/Desktop/Rachel/CeVICHE/11-17-WORMS/avi/', '/home/whale/Desktop/Rachel/CeVICHE/11-21-worms/avi/'

videos = []
for folder in folders:
  fin = len(os.listdir(folder))
  for n in range(1, fin):
    if n < 10:
     n = '00' + str(n)
    else:
     n = '0' + str(n)
    folder = folder + n + '/'
    for vid in os.listdir(folder):
      videos.append(vid)
      print(vid)




def get_frames(videos)
  #which video
  vid = '/home/whale/Desktop/Rachel/CeVICHE/10-26-worms/avi/001/10-26-worms-001.avi'

  # Create a VideoCapture object and read from input file
  # If the input is the camera, pass 0 instead of the video file name
  cap = cv2.VideoCapture(vid)

  # Read until video is completed
  while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
 
      # Display the resulting frame
      #cv2.imshow('Frame',frame)
 
      # Press Q on keyboard to  exit
      #if cv2.waitKey(25) & 0xFF == ord('q'):
        #break
 
    # Break the loop
    else: 
      break

  # When everything done, release the capture
  #cap.release()
  #cv2.destroyAllWindows()

  return frame
##frame transformations
frame = frame.mean(axis=-1,keepdims=1)



