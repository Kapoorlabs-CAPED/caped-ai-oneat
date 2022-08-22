import os
import glob
import sys
import numpy as np
from tqdm import tqdm
from tifffile import imread, imwrite
from vollseg import CARE, VollSeg
from pathlib import Path


image_dir = '/gpfsstore/rech/jsy/uzj81mi/Mari_Data_Oneat/raw/second_dataset_split/'
model_dir = '/gpfsstore/rech/jsy/uzj81mi/Mari_Models/'
save_dir = '/gpfsstore/rech/jsy/uzj81mi/Mari_Data_Oneat/raw/second_dataset_split/den/'
 
noise_model_name = 'CARE/denoising_nuclei_xenopus/'
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
axes = 'ZYX'

for fname in filesRaw:
     
     image = imread(fname)
     Name = os.path.basename(os.path.splitext(fname)[0])
     VollSeg( image, 
             noise_model = noise_model, 
             axes = axes, 
             min_size = min_size,  
             min_size_mask = min_size_mask,
             max_size = max_size,
             n_tiles = n_tiles,
             save_dir = save_dir, 
             Name = Name)    
