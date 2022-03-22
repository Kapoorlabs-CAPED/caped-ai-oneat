#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


#Specify the directory containing images
image_dir = 'Raw_images/'
#Specify the directory contaiing csv files
csv_dir = 'Csv_locations/'
#Specify the model directory where we store the json of categories, training model and parameters
model_dir = 'oneat_models/'
#Directory for storing center ONEAT training data 
save_dir = 'npz_directory/'
Path(model_dir).mkdir(exist_ok = True)
Path(save_dir).mkdir(exist_ok = True)


# In[ ]:


#Name of the focus events
foc_type_name = ["Totaloff","BestCad", "BestNuclei"]
#Label corresponding to static event
foc_type_label = [0, 1, 2]

#The name appended before the CSV files
csv_name_diff = 'ONEAT'
yolo_v0 = False
npz_name = 'Focalplanedetector'
npz_val_name = 'Focalplanedetectorval'
size_tminus = 1
size_tplus = 1
trainshapeX = 128
trainshapeY = 128
normPatch = True


# In[ ]:



crop_size = [trainshapeX,trainshapeY,size_tminus,size_tplus]

#Vectors attached to each static event
foc_position_name = ["x", "y", "z", "h", "w", "c"]
foc_position_label = [0, 1, 2, 3, 4, 5]


# In[ ]:


focus_config = TrainConfig(foc_type_name, foc_type_label, foc_position_name, foc_position_label)

focus_json, focus_cord_json = focus_config.to_json()

save_json(focus_json, model_dir + "Focalplanecategories" + '.json')

save_json(focus_cord_json, model_dir + "Focalplanecord" + '.json')


# # For center ONEAT, event is exactly in the center for all training examples

# In[ ]:


MovieCreator.SegFreeMovieLabelDataSet(image_dir, csv_dir, save_dir, foc_type_name, foc_type_label, csv_name_diff,crop_size, normPatch)


# In[ ]:


MovieCreator.createNPZ(save_dir, axes = 'SZYXC', save_name = npz_name, save_name_val = npz_val_name)

