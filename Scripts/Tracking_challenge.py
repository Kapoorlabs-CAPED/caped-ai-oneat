import os
import glob
import sys
import numpy as np
from tqdm import tqdm
from tifffile import imread, imwrite
from vollseg import StarDist3D, UNET, VollSeg, MASKUNET, CARE
from pathlib import Path


image_dir = '/gpfsstore/rech/jsy/uzj81mi/Tracking_challenge/split_dataset/'
model_dir = '/gpfsdsstore/projects/rech/jsy/uzj81mi/Segmentation_Models/'
save_dir = '/gpfsstore/rech/jsy/uzj81mi/Tracking_challenge/july_2022_segmentation_results/'
 
unet_model_name = 'Unet3D/Unet_Nuclei_Xenopus/'
star_model_name = 'StarDist3D/Carcinoma_cells/'
noise_model_name = 'CARE/denoising_generic_3D/'


unet_model = UNET(config = None, name = unet_model_name, basedir = model_dir)
star_model = StarDist3D(config = None, name = star_model_name, basedir = model_dir)
noise_model = CARE(config = None, name = noise_model_name, basedir = model_dir)



Raw_path = os.path.join(image_dir, '*.tif')
filesRaw = glob.glob(Raw_path)
filesRaw.sort
#Minimum size in pixels for the cells to be segmented
min_size = 1
#Minimum size in pixels for the mask region, regions below this threshold would be removed
min_size_mask=10
#maximum size of the region, set this to veto regions above a certain size
max_size = 1000000
#Adjust the number of tiles depending on how good your GPU is, tiling ensures that your image tiles fit into the runtime
#memory 
n_tiles = (4,4,4)
#If your Unet model is weak we will use the denoising model to obtain the semantic segmentation map, set this to False if this
#is the case else set tit o TRUE if you are using Unet to obtain the semantic segmentation map.
dounet = True
#If you want to do seedpooling from unet and stardist set this to true else it will only take stardist seeds
seedpool = True
#Wether unet create labelling in 3D or slice by slice can be set by this parameter, if true it will merge neighbouring slices
slice_merge = False
#Use probability map for stardist to perform watershedding or use distance map
UseProbability = True
donormalize=True
lower_perc= 1
upper_perc=99.8
axes = 'ZYX'
ExpandLabels = False
prob_thresh = 0.4
nms_thresh = 0.2
for fname in filesRaw:
     
     image = imread(fname)
     Name = os.path.basename(os.path.splitext(fname)[0])
     VollSeg( image, 
             unet_model = unet_model, 
             star_model = star_model, 
             noise_model= noise_model,
             seedpool = seedpool, 
             axes = axes, 
             min_size = min_size,  
             min_size_mask = min_size_mask,
             max_size = max_size,
             donormalize=donormalize,
             lower_perc= lower_perc,
             upper_perc=upper_perc,
             n_tiles = n_tiles,
             prob_thresh = prob_thresh,
             nms_thresh = nms_thresh,
             ExpandLabels = ExpandLabels,
             slice_merge = slice_merge, 
             UseProbability = UseProbability, 
             save_dir = save_dir, 
             Name = Name, 
             dounet = dounet)    
