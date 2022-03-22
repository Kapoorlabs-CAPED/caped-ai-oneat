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
image_dir = 'D:/TrainingData/Helacells_florescent/Raw_ch_hela_flou/'
#Specify the directory contaiing csv files
csv_dir = 'D:/TrainingData/Helacells_florescent/Csv_ch_hela_flou/'

#Specify the model directory where we store the json of categories, training model and parameters
model_dir = 'D:/TrainingData/Helacells_florescent/Oneat_ch_hela_flou/'
#Directory for storing center ONEAT training data 
save_dir = 'D:/TrainingData/Helacells_florescent/Patch_ch_hela_flou_long_big/'
Path(model_dir).mkdir(exist_ok = True)
Path(save_dir).mkdir(exist_ok = True)


# In[ ]:


#Name of the  events
event_type_name = ["Normal", "Division"]
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
size_tplus = 4
tshift = 0
trainshapex = 128
trainshapey = 128
npz_name = 'Cellsplithelafloulongbig'
npz_val_name = 'Cellsplithelafloulongbigval'
crop_size = [trainshapex,trainshapey,size_tminus,size_tplus]


# In[ ]:


#X Y T dynamic events




event_position_name = ["x", "y", "t", "h", "w", "c"]
event_position_label = [0, 1, 2, 3, 4, 5]

dynamic_config = TrainConfig(event_type_name, event_type_label, event_position_name, event_position_label)

dynamic_json, dynamic_cord_json = dynamic_config.to_json()

save_json(dynamic_json, model_dir + "Cellsplithelafloucategories" + '.json')

save_json(dynamic_cord_json, model_dir + "Cellsplithelafloucord" + '.json')        


# In[ ]:


MovieCreator.SegFreeMovieLabelDataSet(image_dir, csv_dir, save_dir, event_type_name, event_type_label, csv_name_diff,crop_size, yolo_v0 = yolo_v0, yolo_v1 = yolo_v1, yolo_v2 = yolo_v2, tshift = tshift)


# In[ ]:


MovieCreator.createNPZ(save_dir, axes = 'STXYC', save_name = npz_name, save_name_val = npz_val_name)


# In[ ]:





# In[ ]:




