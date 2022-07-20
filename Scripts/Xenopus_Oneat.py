#!/usr/bin/env python
# coding: utf-8
import sys
import os
import glob
from csbdeep.models import  CARE
from oneat.NEATModels import NEATDynamic
from oneat.NEATModels.config import dynamic_config
from oneat.NEATUtils import helpers
from oneat.NEATUtils.helpers import load_json
from pathlib import Path

n_tiles = (1,2,2)
event_threshold = 0.9
event_confidence = 0.9
iou_threshold = 0.1
downsamplefactor = 1
#For a Z of 0 to 22 this setup takes the slices from 11 - 4 = 7 to 11 + 1 = 12
start_project_mid = 4
end_project_mid = 1
normalize = True
nms_function = 'iou'

imagedir = '/gpfsstore/rech/jsy/uzj81mi/Mari_Data_Oneat/raw/third_dataset/'
segdir = '/gpfsstore/rech/jsy/uzj81mi/Mari_Data_Oneat/seg/'
model_dir = '/gpfsstore/rech/jsy/uzj81mi/Mari_Models/Oneat/'
savedir= '/gpfsstore/rech/jsy/uzj81mi/Mari_Data_Oneat/oneat_results/'
model_name = 'Cellsplitdetectoroptimizedxenopus'

remove_markers = False
division_categories_json = model_dir + 'Cellsplitcategoriesxenopus.json'
catconfig = load_json(division_categories_json)
division_cord_json = model_dir + 'Cellsplitcordxenopus.json'
cordconfig = load_json(division_cord_json)
model = NEATDynamic(None, model_dir , model_name,catconfig, cordconfig)
Path(savedir).mkdir(exist_ok=True)
Raw_path = os.path.join(imagedir, '*tif')
X = glob.glob(Raw_path)
print(X) 
for imagename in X:
     print(imagename)   
     marker_tree =  model.get_markers(imagename, 
                                                segdir,
                                                start_project_mid = start_project_mid,
                                                end_project_mid = end_project_mid,  
                                                )

                                   
     model.predict(imagename,
                           savedir, 
                           n_tiles = n_tiles, 
                           event_threshold = event_threshold, 
                           event_confidence = event_confidence,
                           iou_threshold = iou_threshold,
                           marker_tree = marker_tree, 
                           remove_markers = remove_markers,
                           nms_function = nms_function,
                           downsamplefactor = downsamplefactor,
                           start_project_mid = start_project_mid,
                           end_project_mid = end_project_mid,
                           normalize = normalize)

