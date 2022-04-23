#!/usr/bin/env python
# coding: utf-8

# In[1]:



import sys
import os
import glob
from oneat.NEATModels import NEATSynamic
from oneat.NEATModels.config import dynamic_config
from oneat.NEATUtils import helpers
from oneat.NEATUtils.helpers import load_json
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from pathlib import Path


# In[2]:


n_tiles = (4,4)
event_threshold = 0.9999
event_confidence = 0.99
iou_threshold = 0.6
fidelity = 16
nms_function = 'iou'
downsamplefactor = 1


# In[3]:


imagedir = 'D:/TestDatasets/Oneat/Hela_brightfield/'
model_dir = 'D:/TrainingModels/Oneat/'
savedir= 'D:/TestDatasets/Oneat/Hela_brightfield/results_d29f32s4k3/'
model_name = 'Cellsplitdetectorbrightfieldd29f32s4k3'

division_categories_json = model_dir + 'Cellsplithelafloucategories.json'
catconfig = load_json(division_categories_json)
division_cord_json = model_dir + 'Cellsplithelafloucord.json'
cordconfig = load_json(division_cord_json)
model = NEATSynamic(None, model_dir , model_name,catconfig, cordconfig)
Path(savedir).mkdir(exist_ok=True)

Raw_path = os.path.join(imagedir, '*tif')
X = glob.glob(Raw_path)

for imagename in X:

     model.predict_synamic(imagename,
                           savedir, 
                           n_tiles = n_tiles, 
                           event_threshold = event_threshold, 
                           event_confidence = event_confidence,
                           iou_threshold = iou_threshold,
                           fidelity = fidelity,
                           nms_function = nms_function,
                           downsamplefactor = downsamplefactor )
