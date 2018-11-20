#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 12:35:08 2018

@author: mpcr
"""

###C.eVICHE###
##C.elegan Visual Implementation in Computer Heuristic Experiments##

import deeplabcut


#create the project, set working directory for videos, and find videos
deeplabcut.create_new_project('CeVICHE','Rachel_StClair', ['/home/mpcr/Desktop/Rachel/CeVICHE/VTS_01_1.VOB'], working_directory='/home/mpcr/Desktop/Rachel/CeVICHE',copy_videos=False)
#find more videos
deeplabcut.add_new_videos(`Full path of the project configuration file*',[`full path of video 4', `full path of video 5'],copy_videos=True/False)
#data selection
deeplabcut.extract_frames(config_path,‘automatic/manual’,‘uniform/kmeans’, crop=True, checkcropping=True)
#extract data frames by hand
deeplabcut.extract_frames(config_path,‘manual’)
#label frames
deeplabcut.label_frames(config_path, Screens=1)
#check annotated frames
deeplabcut.check_labels(config_path)
#create training dataset
deeplabcut.create_training_dataset(config_path,num_shuffles=1)
#train the network --> additional parameters
deeplabcut.train_network(config_path,shuffle=1)
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









































