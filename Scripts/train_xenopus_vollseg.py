#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
from vollseg import SmartSeeds3D
from tifffile import imread, imwrite


# # In the cell below specify the following:
# 
# 

# In[ ]:


base_dir = '/gpfsstore/rech/jsy/uzj81mi/Mari_Data_Training/segmentation_training/'
npz_filename = 'XenopusNucleiSeg'
model_dir = '/gpfsstore/rech/jsy/uzj81mi/Mari_Data/Models/Unet3D/'
model_name = 'nuclei_xenopus_mari_d4f32'

raw_dir = 'raw/'
real_mask_dir = 'real_mask/' 
binary_mask_dir = 'binary_mask/'


# # In this cell choose the network training parameters for the Neural Network
# 
# 

# In[ ]:


#Network training parameters
depth = 4
epochs = 200
learning_rate = 0.0003
batch_size = 2
patch_x = 128
patch_y = 128
patch_z = 16
kern_size = 3
n_patches_per_image = 32
n_rays = 96
startfilter = 32
use_gpu_opencl = False
generate_npz = False
backbone = 'unet'
load_data_sequence = False
validation_split = 0.01
n_channel_in = 1
train_unet = True
train_star = False
train_loss = 'mae'



SmartSeeds3D(base_dir = base_dir, 
             npz_filename = npz_filename, 
             model_name = model_name, 
             model_dir = model_dir,
             raw_dir = raw_dir,
             real_mask_dir = real_mask_dir,
             binary_mask_dir = binary_mask_dir,
             n_channel_in = n_channel_in,
             backbone = backbone,
             load_data_sequence = load_data_sequence, 
             validation_split = validation_split, 
             n_patches_per_image = n_patches_per_image, 
             generate_npz = generate_npz,
             patch_x= patch_x, 
             patch_y= patch_y, 
             patch_z = patch_z,
             erosion_iterations = 0,  
             train_loss = train_loss,
             train_star = train_star,
             train_unet = train_unet,
             use_gpu = use_gpu_opencl,  
             batch_size = batch_size, 
             depth = depth, 
             kern_size = kern_size, 
             startfilter = startfilter, 
             n_rays = n_rays, 
             epochs = epochs, 
             learning_rate = learning_rate)

