#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 12:35:08 2018

@author: mpcr
"""

###C.eVICHE###
##C.elegan Visual Implementation in Computer Heuristic Experiments##
#https://github.com/AlexEMG/DeepLabCut/blob/master/docs/functionDetails.md#g-train-the-network#

import deeplabcut
import os, yaml
from pathlib import Path

##once#create the project, set working directory for videos, and find videos
deeplabcut.create_new_project('CeVICHE','Rachel', ['/home/whale/Desktop/Rachel/CeVICHE/avi/001/ceviche-001.avi', '/home/whale/Desktop/Rachel/CeVICHE/avi/002/ceviche-002.avi', '/home/whale/Desktop/Rachel/CeVICHE/avi/003/ceviche-003.avi' ], working_directory='/home/whale/Desktop/Rachel/CeVICHE', copy_videos=False) 

#specify path to config.yaml
####change yaml for the project
config_path = '/home/whale/Desktop/Rachel/CeVICHE/CeVICHE-Rachel-2019-01-14/config.yaml'

##opt# add more videos
video_path = '/home/whale/Desktop/Rachel/CeVICHE/avi/004/ceviche-00'
video_path1 = '.avi'
for n in range(4,72):
	video_number = str(n)
	video_directory = video_path + video_number + video_path1
	deeplabcut.add_new_videos(config_path, [video_directory], copy_videos=False)

#data selection (auto)
deeplabcut.extract_frames(config_path,'automatic','uniform', crop=False, checkcropping=False)

##opt#extract data frames by hand
deeplabcut.extract_frames(config_path,'manual')

#label frames
deeplabcut.label_frames(config_path)

##opt#check annotated frames
deeplabcut.check_labels(config_path)

#create training dataset
deeplabcut.create_training_dataset(config_path,num_shuffles=1)

#train the network --> additional parameters
deeplabcut.train_network(config_path, shuffle=1, trainingsetindex=0, gputouse=390.87, max_snapshots_to_keep=5, autotune=False, displayiters=None, saveiters=None)

#evaluate the trained network
deeplabcut.evaluate_network(config_path,shuffle=[1], plotting=True)

#analyze new video
deeplabcut.analyze_videos(config_path,[‘/analysis/project/videos/reachingvideo1.avi’],shuffle=1, save_as_csv=True)

#create labeled video --> optional parameters
deeplabcut.create_labeled_video(config_path,[‘/analysis/project/videos/reachingvideo1.avi’,‘/analysis/project/videos/reachingvideo2.avi’])

#plot trajectory of the extracted poses across the analyzed video
deeplabcut.plot_trajectories(‘config_path’,[‘/analysis/project/videos/reachingvideo1.avi’])

#extract outlier frames
deeplabcut.extract_outlier_frames(‘config_path’,[‘videofile_path’])

#refine labels int raining set for outlier condition
deeplabcut.refine_labels(‘config_path’)

#merge corrected frames dataset to existing
deeplabcut.merge_datasets(‘config_path’)









































