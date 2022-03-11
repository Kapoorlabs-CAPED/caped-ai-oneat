

import sys
import os
import glob
from oneat.NEATModels import NEATSynamic, nets
from oneat.NEATModels.config import dynamic_config
from oneat.NEATUtils import helpers
from oneat.NEATUtils.helpers import load_json
from tifffile import imread
from csbdeep.models import Config, CARE

os.environ["CUDA_VISIBLE_DEVICES"]="2"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from pathlib import Path




imagedir =  'images/'
maskimagedir =  'maskimages/'
model_dir = 'models/'
savedir= 'results/'

model_name = 'Celleventpredictor'
mask_name = '_Mask'
division_categories_json = model_dir + 'Celleventcategories.json'
catconfig = load_json(division_categories_json)
division_cord_json = model_dir + 'Celleventcord.json'
cordconfig = load_json(division_cord_json)
model = NEATSynamic(None, model_dir , model_name,catconfig, cordconfig)
Path(savedir).mkdir(exist_ok=True)
n_tiles = (4,4)
event_threshold = 1.0 - 1.0E-4 
iou_threshold = 0.3
dist_threshold = 30
yolo_v2 = False
downsample = 2
thresh = 5



Raw_path = os.path.join(imagedir, '*tif')
X = glob.glob(Raw_path)
X = sorted(X)

Mask_path = os.path.join(maskimagedir, '*tif')
Y = glob.glob(Mask_path)
Y = sorted(Y)

marker_dict = {}
for imagename in X:
  Name = os.path.basename(os.path.splitext(imagename)[0])  
  for maskimagename in Y:   
     MaskName = os.path.basename(os.path.splitext(maskimagename)[0]) 
     
     if MaskName == Name + mask_name:
          model.predict_synamic(imread(imagename), savedir, n_tiles = n_tiles, event_threshold = event_threshold, iou_threshold = iou_threshold, 
          downsamplefactor = downsample, thresh = thresh, maskimage = imread(maskimagename), dist_threshold = dist_threshold)






