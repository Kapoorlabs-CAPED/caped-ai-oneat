import os
import glob
import sys
import numpy as np
from tqdm import tqdm
from tifffile import imread, imwrite
from vollseg import StarDist3D, UNET, VollSeg, MASKUNET, CARE
from pathlib import Path
from natsort import natsorted
from oneat.NEATUtils import NEATViz
from oneat.NEATModels import NEATDynamic
from oneat.NEATModels.config import dynamic_config
from oneat.NEATUtils import helpers
from oneat.NEATUtils.helpers import load_json

image_dir = '/gpfsstore/rech/jsy/uzj81mi/Mari_Data_Oneat/raw/second_dataset/'
seg_dir = '/gpfsstore/rech/jsy/uzj81mi/Mari_Data_Oneat/seg/'
model_dir = '/gpfsstore/rech/jsy/uzj81mi/Mari_Models/'

split_image_dir = '/gpfsstore/rech/jsy/uzj81mi/Mari_Data_Oneat/raw/second_dataset_split/'
split_save_dir = '/gpfsstore/rech/jsy/uzj81mi/Mari_Data_Oneat/raw/second_dataset_split/seg/'
save_dir_oneat = '/gpfsstore/rech/jsy/uzj81mi/Mari_Data_Oneat/oneat_results/'

unet_model_name = 'Unet3D/Unet_Nuclei_Xenopus/'
star_model_name = 'StarDist3D/Nuclei_Xenopus_Mari/'
roi_model_name = 'MASKUNET/Roi_Nuclei_Xenopus/'
oneat_model_name  = 'Oneat/Cellsplitdetectoroptimizedxenopus'

unet_model = UNET(config = None, name = unet_model_name, basedir = model_dir)
star_model = StarDist3D(config = None, name = star_model_name, basedir = model_dir)
roi_model = MASKUNET(config = None, name = roi_model_name, basedir = model_dir)
division_categories_json = model_dir + 'Oneat/Cellsplitcategoriesxenopus.json'
catconfig = load_json(division_categories_json)
division_cord_json = model_dir + 'Oneat/Cellsplitcordxenopus.json'
cordconfig = load_json(division_cord_json)
oneat_model = NEATDynamic(None, model_dir , oneat_model_name,catconfig, cordconfig)

Path(seg_dir).mkdir(exist_ok=True)
Path(split_image_dir).mkdir(exist_ok=True)
Path(split_save_dir).mkdir(exist_ok=True)
Path(save_dir_oneat).mkdir(exist_ok=True)

Raw_path = os.path.join(image_dir, '*.tif')
filesRaw = glob.glob(Raw_path)

fileextension = '*tif'
#Minimum size in pixels for the cells to be segmented
min_size = 1
#Minimum size in pixels for the mask region, regions below this threshold would be removed
min_size_mask=10
#maximum size of the region, set this to veto regions above a certain size
max_size = 1000000
#Adjust the number of tiles depending on how good your GPU is, tiling ensures that your image tiles fit into the runtime
#memory
#If your Unet model is weak we will use the denoising model to obtain the semantic segmentation map, set this to False if this
#is the case else set tit o TRUE if you are using Unet to obtain the semantic segmentation map.
dounet = True
#If you want to do seedpooling from unet and stardist set this to true else it will only take stardist seeds
seedpool = True
#Wether unet create labelling in 3D or slice by slice can be set by this parameter, if true it will merge neighbouring slices
slice_merge = False
remove_markers = False
n_tiles = (2,8,8)
event_threshold = 0.99
event_confidence = 0.9
iou_threshold = 0.1
downsamplefactor = 1
#For a Z of 0 to 22 this setup takes the slices from 11 - 4 = 7 to 11 + 1 = 12
start_project_mid = 4
end_project_mid = 1
normalize = True
nms_function = 'iou'
#Use probability map for stardist to perform watershedding or use distance map
UseProbability = True
donormalize=True
lower_perc= 1
upper_perc=99.8
axes = 'ZYX'
result_type = 'StarDist'

prob_thresh = 0.672842
nms_thresh = 0.3
nms_space = 10
nms_time = 3
ExpandLabels = False
filesRaw = natsorted(filesRaw)

for imagename in filesRaw:
             
          image = imread(imagename)
          Name = os.path.basename(os.path.splitext(imagename)[0])
          for i in range(image.shape[0]):
                 imwrite(split_image_dir + '/' + Name + str(i) +  '.tif', image[i,:,:,:].astype('float32') )

          Raw_path = os.path.join(split_image_dir, '*.tif')
          filesRaw = glob.glob(Raw_path)
          filesRaw = natsorted(filesRaw)

          for fname in filesRaw:

                  image = imread(fname)
                  Name = os.path.basename(os.path.splitext(fname)[0])
                  VollSeg( image,
                          unet_model = unet_model,
                          star_model = star_model,
                          roi_model = roi_model,
                          seedpool = seedpool,
                          axes = axes,
                          min_size = min_size,
                          min_size_mask = min_size_mask,
                          max_size = max_size,
                          donormalize = donormalize,
                          lower_perc= lower_perc,
                          upper_perc=upper_perc,
                          n_tiles = n_tiles,
                          prob_thresh = prob_thresh,
                          nms_thresh = nms_thresh,
                          ExpandLabels = ExpandLabels,
                          slice_merge = slice_merge,
                          UseProbability = UseProbability,
                          save_dir = split_save_dir,
                          Name = Name,
                          dounet = dounet)
          split_save_dir = split_save_dir + result_type
          Seg_path = os.path.join(split_save_dir, '*.tif')
          filesSeg = glob.glob(Seg_path)
          filesSeg = natsorted(filesSeg)
          allseg = []
          for fname in filesSeg:
                segimage = imread(fname).astype('uint16')
                allseg.append(segimage)

          allseg = np.asarray(allseg)
          imwrite(seg_dir + '/' + Name +  '.tif', allseg.astype('uint16') )

          marker_tree =  oneat_model.get_markers(imagename,seg_dir,start_project_mid = start_project_mid,end_project_mid = end_project_mid)

          oneat_model.predict( imagename,
                          save_dir_oneat,
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
Vizdetections = NEATViz(image_dir,
            save_dir_oneat,
            division_categories_json,
            segimagedir = seg_dir,
            fileextension = fileextension,
            start_project_mid = start_project_mid,
            end_project_mid = end_project_mid,
            headless = True,
            event_threshold = event_threshold,
            nms_space = nms_space,
            nms_time = nms_time)
