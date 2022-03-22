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


#Name of the static events
cell_type_name = ["Normal","Division", "Apoptosis", "MacroCheate", "NonMatureP1", "MatureP1"]
#Label corresponding to static event
cell_type_label = [0, 1, 2, 3, 4, 5]

#The name appended before the CSV files
tshift = 1
csv_name_diff = 'ONEAT'
yolo_v0 = False
npz_name = 'Celltypedetector'
npz_val_name = 'Celltypedetectorval'
crop_size = [trainshapey,trainshapex]


# In[ ]:


cell_position_name = ["x", "y", "h", "w", "c"]
cell_position_label = [0, 1, 2, 3, 4]


# In[ ]:


static_config = TrainConfig(cell_type_name, cell_type_label, cell_position_name, cell_position_label)

static_json, static_cord_json = static_config.to_json()


save_json(static_json, model_dir + "Celltypecategories" + '.json')

save_json(static_cord_json, model_dir + "Celltypecord" + '.json')


# # For center ONEAT, event is exactly in the center for all training examples

# In[ ]:


MovieCreator.SegFreeImageLabelDataSet(image_dir, csv_dir, save_dir, cell_type_name, cell_type_label, csv_name_diff,crop_size)


# In[ ]:


MovieCreator.createNPZ(save_dir, axes = 'SYXC', save_name = npz_name, save_name_val = npz_val_name, static = True)

