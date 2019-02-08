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

vid = '/home/default/Downloads/DJI_0009 Shark cut 10.avi'
cap = cv2.VideoCapture(vid)
ret, frame = cap.read()
frame.shape
###

vid = '/home/default/Downloads/DJI_0009 Shark cut 10.avi'
cap = cv2.VideoCapture(vid)
cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
cv2.resizeWindow('frame', 800,800)
while(cap.isOpened()):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
##
while(cap.isOpened()):
    ret, frame = cap.read()
    print(frame.shape)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()



















