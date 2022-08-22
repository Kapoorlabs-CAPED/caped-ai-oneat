#!/usr/bin/env python
# coding: utf-8



import sys
import os
from glob import glob
from oneat.NEATModels import NEATDynamic
from oneat.NEATModels.config import dynamic_config
from oneat.NEATUtils.helpers import save_json, load_json





npz_directory = '/gpfsstore/rech/jsy/uzj81mi/Mari_Data_Training/oneat_training/oneat_train_patches_m1p1/'
npz_name = 'Xenopus_oneat_training_m1p1.npz'
npz_val_name = 'Xenopus_oneat_training_m1p1val.npz'

#Read and Write the h5 file, directory location and name
model_dir =  '/gpfsstore/rech/jsy/uzj81mi/Mari_Models/Oneat/'
model_name = 'Cellsplitdetectortmtp11d29f32l32.h5'



#Neural network parameters
division_categories_json = model_dir + 'Cellsplitcategoriesxenopus.json'
key_categories = load_json(division_categories_json)
division_cord_json = model_dir + 'Cellsplitcordxenopus.json'
key_cord = load_json(division_cord_json)

#For ORNET use residual = True and for OSNET use residual = False
residual = True
#Number of starting convolutional filters, is doubled down with increasing depth
startfilter = 32
#CNN network start layer, mid layers and lstm layer kernel size
start_kernel = 7
lstm_kernel = 3
mid_kernel = 3
#Network depth has to be 9n + 2, n= 3 or 4 is optimal for Notum dataset
depth = 29
#Size of the gradient descent length vector, start small and use callbacks to get smaller when reaching the minima
learning_rate = 1.0E-3
#For stochastic gradient decent, the batch size used for computing the gradients
batch_size = 25
# use softmax for single event per box, sigmoid for multi event per box
lstm_hidden_unit = 32
#Training epochs, longer the better with proper chosen learning rate
epochs = 250
nboxes = 1
#The inbuilt model stride which is equal to the nulber of times image was downsampled by the network
show = False
stage_number = 3
size_tminus = 1
size_tplus = 1
imagex = 64
imagey = 64
yolo_v0 = False
yolo_v1 = True
yolo_v2 = False





config = dynamic_config(npz_directory =npz_directory, npz_name = npz_name, npz_val_name = npz_val_name, 
                         key_categories = key_categories, key_cord = key_cord, nboxes = nboxes, imagex = imagex,
                         imagey = imagey, size_tminus = size_tminus, size_tplus = size_tplus, epochs = epochs, yolo_v0 = yolo_v0, yolo_v1 = yolo_v1, yolo_v2 = yolo_v2,learning_rate = learning_rate,
                         residual = residual, depth = depth, start_kernel = start_kernel, mid_kernel = mid_kernel, stage_number = stage_number,
                         lstm_kernel = lstm_kernel, lstm_hidden_unit = lstm_hidden_unit, show = show,
                         startfilter = startfilter, batch_size = batch_size, model_name = model_name)

config_json = config.to_json()

print(config)
save_json(config_json, model_dir + os.path.splitext(model_name)[0] + '_Parameter.json')




Train = NEATDynamic(config, model_dir, model_name)

Train.loadData()

Train.TrainModel()






