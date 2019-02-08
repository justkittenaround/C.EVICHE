#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 12:35:08 2018

@author: Rachel St Clair
"""

###DLC###
##basic code to run deeplabcut##
#https://github.com/AlexEMG/DeepLabCut/blob/master/docs/functionDetails.md#g-train-the-network#

import deeplabcut
import os, yaml
from pathlib import Path

##once#create the project, set working directory for videos, and find videos
deeplabcut.create_new_project('name_of_project','author', ['path_to_video.avi' ], working_directory='path_to_folder', copy_videos=False) 

#specify path to config.yaml
####change yaml for the project
config_path = 'copy the path to the created yaml file here.yaml'

##opt# add more videos
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









































