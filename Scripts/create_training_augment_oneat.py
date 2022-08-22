#!/usr/bin/env python
# coding: utf-8



from pathlib import Path
from oneat.NEATUtils import TemporalAug
import os
from tifffile import imread, imwrite
import numpy as np

#Specify the directory containing images
image_dir = Path('/gpfsstore/rech/jsy/uzj81mi/Mari_Data_Training/oneat_training/oneat_train_raw/')
#Specify the directory contaiing csv files
csv_dir = Path('/gpfsstore/rech/jsy/uzj81mi/Mari_Data_Training/oneat_training/oneat_train_csv/')
#Specify the directory containing the segmentations
seg_image_dir = Path('/gpfsstore/rech/jsy/uzj81mi/Mari_Data_Training/oneat_training/oneat_train_seg/')
#Specify the directory for storing the augmented images
aug_image_dir = '/gpfsstore/rech/jsy/uzj81mi/Mari_Data_Training/oneat_training/oneat_train_raw_aug/'
#Specify the directory for storing the augmented labels
aug_seg_image_dir = '/gpfsstore/rech/jsy/uzj81mi/Mari_Data_Training/oneat_training/oneat_train_seg_aug/'
#Specify the directory for storing the augmented csv files
aug_csv_dir = '/gpfsstore/rech/jsy/uzj81mi/Mari_Data_Training/oneat_training/oneat_train_csv_aug/'

Path(aug_image_dir).mkdir(exist_ok = True)
Path(aug_seg_image_dir).mkdir(exist_ok = True)
Path(aug_csv_dir).mkdir(exist_ok = True)

pattern = '*.tif'

files_raw = list(image_dir.glob(pattern))
files_seg = list(seg_image_dir.glob(pattern))
files_csv = list(csv_dir.glob('*.csv'))

#Name of the  events
event_type_name = ["Normal", "Division"]
#Label corresponding to event
event_type_label = [0, 1]
csv_name_diff = 'ONEAT'
count = 0
rotation_angles = [3,6,9,12,15,18]

mean = 0.0
sigma = 10.0
distribution = 'Both'



for fname in files_raw:
                  
    name = os.path.basename(os.path.splitext(fname)[0])   
    for Segfname in files_seg:

      Segname = os.path.basename(os.path.splitext(Segfname)[0])

      if name == Segname:
        
        
        image = imread(fname)
        segimage = imread(Segfname)

        for csvfname in files_csv:
                count = 0  
                Csvname =  os.path.basename(os.path.splitext(csvfname)[0])
                for i in  range(0, len(event_type_name)):
                    event_name = event_type_name[i]
                    trainlabel = event_type_label[i]
                    classfound = (Csvname == csv_name_diff +  event_name + name)   
                    if classfound:
                        
                        #Rotate
                        for rotate_angle in rotation_angles:
                                
                                rotate_pixels = TemporalAug(rotate_angle = rotate_angle)

                                aug_rotate_pixels,aug_rotate_pixels_label, aug_rotate_pixels_csv  = rotate_pixels.build(image = np.copy(image), labelimage = segimage, labelcsv = csvfname)
                                
                               
                                save_name_raw = aug_image_dir + '/' + 'rotation_' +  str(rotate_angle) + name + '.tif'
                                save_name_seg = aug_seg_image_dir + '/' + 'rotation_' +  str(rotate_angle) + name + '.tif'
                                if os.path.exists(save_name_raw) == False:
                                    imwrite(save_name_raw, aug_rotate_pixels.astype('float32'))
                                if os.path.exists(save_name_seg) == False:    
                                    imwrite(save_name_seg, aug_rotate_pixels_label.astype('uint16'))
                                aug_rotate_pixels_csv.to_csv(aug_csv_dir + '/' + csv_name_diff + event_name + 'rotation_' +  str(rotate_angle) + name +  '.csv', index = False, mode = 'w')
                                count = count + 1
        for csvfname in files_csv:
                count = 0  
                Csvname =  os.path.basename(os.path.splitext(csvfname)[0])
                for i in  range(0, len(event_type_name)):
                    event_name = event_type_name[i]
                    trainlabel = event_type_label[i]
                    classfound = (Csvname == csv_name_diff +  event_name + name)   
                    if classfound: 
                        #Additive Noise
                        addnoise_pixels = TemporalAug(mean = mean, sigma = sigma, distribution = distribution)

                        aug_addnoise_pixels,aug_addnoise_pixels_label, aug_addnoise_pixels_csv  = addnoise_pixels.build(image = np.copy(image), labelimage = segimage, labelcsv = csvfname)
                        
                        save_name_raw = aug_image_dir + '/' + 'noise_' +  str(sigma) + name + '.tif'
                        save_name_seg = aug_seg_image_dir + '/' + 'noise_' +   str(sigma) + name + '.tif'
                        if os.path.exists(save_name_raw) == False:
                            imwrite(save_name_raw, aug_addnoise_pixels.astype('float32'))
                        if os.path.exists(save_name_seg) == False:    
                            imwrite(save_name_seg, aug_addnoise_pixels_label.astype('uint16'))
                        aug_addnoise_pixels_csv.to_csv(aug_csv_dir + '/' + csv_name_diff + event_name + 'noise_' +   str(sigma) + name +  '.csv', index = False, mode = 'w')
                        count = count + 1

        
