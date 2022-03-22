#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import os
from glob import glob
from oneat.NEATModels import NEATPredict
from oneat.NEATUtils.helpers import load_json
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


# In[ ]:


Z_imagedir = 'Z_raw/'
imagedir = 'Projected_raw/'
model_dir =  '/Oneat_models/'
model_name = 'Cellsplitpredictor'

division_categories_json = model_dir + 'MicroscopeCategories.json'
catconfig = load_json(division_categories_json)
division_cord_json = model_dir + 'MicroscopeCord.json'
cordconfig = load_json(division_cord_json)
fileextension = '*TIF'

model = NEATPredict(None, model_dir , model_name,catconfig, cordconfig)
projection_model = ProjectionCARE(config = None, name = projection_model_name, basedir = model_dir)


# In[ ]:

#Number of tiles for projected image
n_tiles = (1,1)
#Number of tiles for Z stacks
Z_n_tiles = (1,2,2)
#Network probability threshold for detecting the event
event_threshold = 0.999
#Network confidence of having a cell
event_confidence = 0.99
#IOU of bounding boxes to indicate single or multiple detections
iou_threshold = 0.6
#Number of predictions to be reported
nb_predictions = 30
#Number of overlapping boxes (indicator of good detection)
fidelity = 8
#Should the image be downsampled to fit 10 cells in teh training patch size used
downsample = 1
#Region of interest to get the predcition in, value from 0 to 1
roi_start = 0.25
roi_end = 0.75

# In[ ]:


model.predict_microscope(imagedir, 
              [], 
              [], 
              Z_imagedir, 
              [], 
              [],  
              0, 
              0,
              downsample = downsample,
              fileextension = fileextension, 
              nb_prediction = nb_predictions, 
              n_tiles = n_tiles, 
              Z_n_tiles = Z_n_tiles, 
              event_threshold = event_threshold, 
              event_confidence = event_confidence,
              roi_start = roi_start,
              roi_end = roi_end,
              iou_threshold = iou_threshold,
              fidelity = fidelity)

