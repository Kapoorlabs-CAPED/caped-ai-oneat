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


# In[7]:


#Specify the directory containing images
image_dir = '/gpfsstore/rech/jsy/uzj81mi/Mari_Data_Training/oneat_training/oneat_train_raw/'
#Specify the directory contaiing csv files
csv_dir = '/gpfsstore/rech/jsy/uzj81mi/Mari_Data_Training/oneat_training/oneat_train_diamond_csv/'
#Specify the directory containing the segmentations
seg_image_dir = '/gpfsstore/rech/jsy/uzj81mi/Mari_Data_Training/oneat_training/oneat_train_seg/'
#Specify the model directory where we store the json of categories, training model and parameters
model_dir = '/gpfsstore/rech/jsy/uzj81mi/Mari_Models/Oneat/'
#Directory for storing center ONEAT training data 
save_dir = '/gpfsstore/rech/jsy/uzj81mi/Mari_Data_Training/oneat_training/oneat_train_patches_m1p1/'
Path(model_dir).mkdir(exist_ok = True)
Path(save_dir).mkdir(exist_ok = True)


# In[8]:


#Name of the  events
event_type_name = ["Normal", "Division"]
#Label corresponding to event
event_type_label = [0, 1]
#The name appended before the CSV files
csv_name_diff = 'ONEAT'
size_tminus = 1
size_tplus = 1
trainshapex = 64
trainshapey = 64
trainshapez = 16
normalizeimage = True
npz_name = 'Xenopus_oneat_training_m1p1_noise'
npz_val_name = 'Xenopus_oneat_training_m1p1_noiseval'
crop_size = [trainshapex,trainshapey,trainshapez,size_tminus,size_tplus]


event_position_name = ["x", "y", "z", "t", "h", "w", "d", "c"]
event_position_label = [0, 1, 2, 3, 4, 5, 6, 7]

dynamic_config = TrainConfig(event_type_name, event_type_label, event_position_name, event_position_label)

dynamic_json, dynamic_cord_json = dynamic_config.to_json()

save_json(dynamic_json, model_dir + "Cellsplitdiamondcategoriesxenopus" + '.json')

save_json(dynamic_cord_json, model_dir + "Cellsplitdiamondcordxenopus" + '.json')        




MovieCreator.VolumeLabelDataSet(image_dir, 
                               seg_image_dir, 
                               csv_dir, 
                               save_dir, 
                               event_type_name, 
                               event_type_label, 
                               csv_name_diff,
                               crop_size,
                               normalizeimage = normalizeimage)




MovieCreator.createNPZ(save_dir, axes = 'STZXY', save_name = npz_name, save_name_val = npz_val_name)







# In[ ]:




