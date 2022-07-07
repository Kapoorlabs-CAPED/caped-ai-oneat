#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import os
from glob import glob
from oneat.NEATModels import NEATStatic, nets
from oneat.NEATModels.Staticconfig  import static_config
from oneat.NEATUtils import helpers
from oneat.NEATUtils.helpers import save_json, load_json


# In[ ]:


npz_directory = '/gpfsscratch/rech/jsy/uzj81mi/MIDOG_RAW/MIDOG_Challenge_2022_p128/'
npz_name = 'MIDOG_128.npz'
npz_val_name = 'MIDOG_128val.npz'

#Read and Write the h5 file, directory location and name
model_dir =  '/gpfsstore/rech/jsy/uzj81mi/MIDOG_RAW/models/'
model_name = 'Midog128d29f32s3K3.h5'

static_categories_json = model_dir + 'Midogcategories.json'
key_categories = load_json(static_categories_json)
static_cord_json = model_dir + 'Midogcord.json'
key_cord = load_json(static_cord_json)

#For ORNET use residual = True and for OSNET use residual = False
residual = True
#NUmber of starting convolutional filters, is doubled down with increasing depth
startfilter = 32
#CNN network start layer, mid layers and lstm layer kernel size
start_kernel = 3
mid_kernel = 3
#Network depth has to be 9n + 2, n= 3 or 4 is optimal for Notum dataset
depth = 29
#Size of the gradient desent length vector, start small and use callbacks to get smaller when reaching the minima
learning_rate = 1.0E-3
#For stochastic gradient decent, the batch size used for computing the gradients
batch_size = 1
# Trainng image size
yolo_v0 = False
show = False
#Training epochs, longer the better with proper chosen learning rate
epochs = 250
nboxes = 1

#The inbuilt model stride which is equal to the nulber of times image was downsampled by the network
stage_number = 3
imagex = 128
imagey = 128

# In[ ]:


config = static_config(npz_directory =npz_directory, npz_name = npz_name, npz_val_name = npz_val_name,
                         key_categories = key_categories, key_cord = key_cord, imagex = imagex, imagey = imagey,
                         residual = residual, depth = depth, start_kernel = start_kernel, mid_kernel = mid_kernel,
                         startfilter = startfilter, nboxes = nboxes, gridx = 1, gridy = 1, show = show, multievent = False,
                         epochs = epochs, learning_rate = learning_rate,stage_number = stage_number,
                         batch_size = batch_size, model_name = model_name, yolo_v0 = yolo_v0)

config_json = config.to_json()

print(config)
save_json(config_json, model_dir + os.path.splitext(model_name)[0] + '_Parameter.json')


# In[ ]:


static_model = NEATStatic(config, model_dir, model_name, class_only = True, train_lstm = True)

static_model.loadData(sum_channels = False)

static_model.TrainModel()