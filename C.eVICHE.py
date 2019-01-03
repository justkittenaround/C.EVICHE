#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 12:35:08 2018

@author: mpcr
"""

###C.eVICHE###
##C.elegan Visual Implementation in Computer Heuristic Experiments##

import deeplabcut
import os, yaml
from pathlib import Path

##once#create the project, set working directory for videos, and find videos
deeplabcut.create_new_project('CeVICHE','Rachel', ['/home/mpcr/Desktop/Rachel/CeVICHE/avi/001/ceviche-001.avi', '/home/mpcr/Desktop/Rachel/CeVICHE/avi/002/ceviche-002.avi', '/home/mpcr/Desktop/Rachel/CeVICHE/avi/003/ceviche-003.avi' ], working_directory='/home/mpcr/Desktop/Rachel/CeVICHE', copy_videos=False) 

#specify path to config.yaml
####change yaml for the project
path_config_file = '/home/mpcr/Desktop/Rachel/CeVICHE/CeVICHE-Rachel-2018-12-05/config.yaml'

##opt# more videos
deeplabcut.add_new_videos(path_config_file, ['full path of video 4', 'full path of video 5'], copy_videos=False)

#data selection (auto)
deeplabcut.extract_frames(path_config_file,'automatic','uniform', crop=False, checkcropping=False)

##opt#extract data frames by hand
deeplabcut.extract_frames(config_path,'manual')

#label frames
deeplabcut.label_frames(path_config_file)

##opt#check annotated frames
deeplabcut.check_labels(path_config_file)

#create training dataset
deeplabcut.create_training_dataset(path_config_file,num_shuffles=1)

#train the network --> additional parameters
deeplabcut.train_network(path_config_file, shuffle=1, trainingsetindex=0, gputouse=390.87, max_snapshots_to_keep=5, autotune=False, displayiters=None, saveiters=None)

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









































