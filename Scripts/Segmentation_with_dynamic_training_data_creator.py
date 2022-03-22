#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from tifffile import imread 
import sys
import os
import cv2
import glob
from tqdm import tqdm
import pandas as pd
from oneat.NEATUtils import MovieCreator
from oneat.NEATUtils.helpers import save_json, load_json
from oneat.NEATModels.TrainConfig import TrainConfig
from pathlib import Path


# In[2]:


#Specify the directory containing images
image_dir = '/home/sancere/VKepler/CurieTrainingDatasets/oneatimages/bin2/'
#Specify the directory contaiing csv files
csv_dir = '/home/sancere/VKepler/CurieTrainingDatasets/oneatcsv/apoptosisdiamondclassic/'
#Specify the directory containing the segmentations
seg_image_dir = '/home/sancere/VKepler/CurieTrainingDatasets/oneatimages/bin2/segmentation/'
#Specify the model directory where we store the json of categories, training model and parameters
model_dir = '/home/sancere/VKepler/CurieDeepLearningModels/WinnerOneatModels/'
#Directory for storing center ONEAT training data 
save_dir = '/home/sancere/VKepler/CurieTrainingDatasets/Apoptosism4p3data/'
Path(model_dir).mkdir(exist_ok = True)
Path(save_dir).mkdir(exist_ok = True)


# In[3]:


#Name of the  events
event_type_name = ["Normal", "Apoptosis"]
#Label corresponding to event
event_type_label = [0, 1]

#The name appended before the CSV files
csv_name_diff = 'ONEAT'
#with xythw and class terms only
yolo_v0 = False
#with confidence term
yolo_v1 = True
#with angle term
yolo_v2 = False
size_tminus = 4
size_tplus = 3
tshift = 1
trainshapex = 64
trainshapey = 64
npz_name = 'Celldeathpredictorm4p3'
npz_val_name = 'Celldeathpredictorm4p3val'
crop_size = [trainshapex,trainshapey,size_tminus,size_tplus]


# In[4]:


#X Y T dynamic events




event_position_name = ["x", "y", "t", "h", "w", "c"]
event_position_label = [0, 1, 2, 3, 4, 5]

dynamic_config = TrainConfig(event_type_name, event_type_label, event_position_name, event_position_label)

dynamic_json, dynamic_cord_json = dynamic_config.to_json()

save_json(dynamic_json, model_dir + "Celldeathcategories" + '.json')

save_json(dynamic_cord_json, model_dir + "Celldeathcord" + '.json')        


# In[5]:


MovieCreator.MovieLabelDataSet(image_dir, seg_image_dir, csv_dir, save_dir, event_type_name, event_type_label, csv_name_diff,crop_size, tshift = tshift, yolo_v0 = yolo_v0, yolo_v1 = yolo_v1, yolo_v2 = yolo_v2)


# In[6]:


MovieCreator.createNPZ(save_dir, axes = 'STXYC', save_name = npz_name, save_name_val = npz_val_name)

