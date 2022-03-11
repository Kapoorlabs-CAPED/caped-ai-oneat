import sys
import os
from glob import glob
from oneat.NEATModels import NEATDynamic, nets
from oneat.NEATModels.config import NeatConfig
from oneat.NEATUtils import helpers
from oneat.NEATUtils.helpers import load_json
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


imagename = 'oneat_images/image.tif'
model_dir = 'oneat_models/'
savedir = 'oneat_images/result/'
model_name = 'cellsplitdetector'
star_model_name = 'starmodel'
mask_model_name = 'maskmodel'
division_categories_json = model_dir + 'Celleventcategories.json'
catconfig = load_json(division_categories_json)
division_cord_json = model_dir + 'Celleventcord.json'
cordconfig = load_json(division_cord_json)
n_tiles = (1,1)
event_threshold = 1 - 1.0E-4
iou_threshold = 0.01
downsamplefactor = 1
fidelity = 5
maskfilter = 10
remove_normal_markers = True
model = NEATDynamic(None, model_dir , model_name,catconfig, cordconfig)


markers, marker_tree, watershed, mask = model.get_markers(imagename, star_model_name, mask_model_name, savedir, n_tiles, downsamplefactor = downsamplefactor, remove_markers = remove_normal_markers)

model.predict(imagename, savedir, n_tiles = n_tiles, event_threshold = event_threshold, iou_threshold = iou_threshold, 
fidelity = fidelity, downsamplefactor = downsamplefactor, maskfilter = maskfilter, markers = markers, marker_tree = marker_tree, watershed = watershed, maskimage = mask, remove_markers = remove_normal_markers)