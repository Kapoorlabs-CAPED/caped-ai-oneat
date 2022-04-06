#!/usr/bin/env python
# coding: utf-8

# In[1]:



import sys
import os
import glob
from csbdeep.models import  CARE
from oneat.NEATModels import NEATDynamic
from oneat.NEATModels.config import dynamic_config
from oneat.NEATUtils import helpers
from oneat.NEATUtils.helpers import load_json
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from pathlib import Path


# In[4]:


n_tiles = (4,8,8)
event_threshold = 0.999
event_confidence = 0.9
iou_threshold = 0.4
fidelity = 4
downsamplefactor = 1
#For a Z of 0 to 22 this setup takes the slices from 11 - 4 = 7 to 11 + 1 = 12
start_project_mid = 4
end_project_mid = 1
normalize = True


# In[ ]:


imagedir = 'D:/TestDatasets/Oneat/Xenopus_oneat/'
segdir = 'D:/TestDatasets/Oneat/Xenopus_oneat/seg/'
model_dir = 'D:/TrainingModels/Oneat/'
savedir= 'D:/TestDatasets/Oneat/Xenopus_oneat/resultsGOLD_d29f32_fid4_iou0.4/'
model_name = 'Cellsplitsxenopus_xy64_tm1tp1_s3d29f32'
use_seg_only = True
remove_markers = True
division_categories_json = model_dir + 'Cellsplitcategoriesxenopus.json'
catconfig = load_json(division_categories_json)
division_cord_json = model_dir + 'Cellsplitcordxenopus.json'
cordconfig = load_json(division_cord_json)
model = NEATDynamic(None, model_dir , model_name,catconfig, cordconfig)

Path(savedir).mkdir(exist_ok=True)

Raw_path = os.path.join(imagedir, '*tif')
X = glob.glob(Raw_path)
 
for imagename in X:
   
     markers, marker_tree, watershed, stardist, mask =  model.get_markers(imagename, 
                                                                  savedir, 
                                                                  n_tiles = n_tiles, 
                                                                  segdir =segdir,
                                                                  use_seg_only = use_seg_only,
                                                                  remove_markers = remove_markers)
     model.predict(imagename,
                           savedir, 
                           n_tiles = n_tiles, 
                           event_threshold = event_threshold, 
                           event_confidence = event_confidence,
                           iou_threshold = iou_threshold,
                           fidelity = fidelity,
                           markers = markers, marker_tree = marker_tree, 
                           watershed = watershed, stardist = stardist, 
                           remove_markers = remove_markers,
                           downsamplefactor = downsamplefactor,
                           start_project_mid = start_project_mid,
                           end_project_mid = end_project_mid,
                           normalize = normalize)


# In[ ]:





# In[ ]:




