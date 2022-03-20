#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 16:05:15 2021

@author: aimachine
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 14:50:47 2021

@author: vkapoor
"""
from tifffile import imread, imwrite
import csv
import napari
import glob
import os
import sys
import numpy as np
import json
from pathlib import Path
from scipy import spatial
import itertools
from napari.qt.threading import thread_worker
import matplotlib.pyplot  as plt
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox, QPushButton, QSlider
import h5py
import pandas as pd
import imageio
from dask.array.image import imread as daskread
from matplotlib import cm
from scipy.ndimage.filters import median_filter, gaussian_filter, maximum_filter

class Zmapgen(object):

        def __init__(self, imagedir, savedir, append_name = "Mask" , fileextension = '*tif', show_after = 1, radius = 10):
            
            
               self.imagedir = imagedir
               self.savedir = savedir
               self.fileextension = fileextension
               self.show_after = show_after
               self.append_name = append_name
               self.radius = radius
               Path(self.savedir).mkdir(exist_ok=True)
               self.genmap()
               
        def genmap(self):
                 
                 
                 Raw_path = os.path.join(self.imagedir, self.fileextension)
                 X = glob.glob(Raw_path)
                 count = 0
                 for fname in X:
                     Name = os.path.basename(os.path.splitext(fname)[0])
                     if self.append_name in Name and '_Zmap' not in Name:
                        image = imread(fname)
                        count = count + 1
                        Signal_first = image[:,:,:,1]
                        Signal_second = image[:,:,:,2]
                        Sum_signal_first = gaussian_filter(np.sum(Signal_first, axis = 0), self.radius)
                        Sum_signal_first = normalizeZeroOne(Sum_signal_first)
                        Sum_signal_second = gaussian_filter(np.sum(Signal_second, axis = 0) , self.radius)
                        
                        Sum_signal_second = normalizeZeroOne(Sum_signal_second)
                        
                        Zmap = np.zeros([Sum_signal_first.shape[0],Sum_signal_first.shape[1],  3])
                        Zmap[:,:,0] = Sum_signal_first
                        Zmap[:,:,1] = Sum_signal_second
                        Zmap[:,:,2] = (Sum_signal_first + Sum_signal_second)/2
                        if count%self.show_after == 0:
                            doubleplot(Sum_signal_first, Sum_signal_second, Name + "First Channel Z map", "Second Channel Z map")
                        
                        imwrite(self.savedir + Name + '_Zmap' + '.tif', Zmap)
                        
  
def normalizeZeroOne(x):

     x = x.astype('float32')

     minVal = np.min(x)
     maxVal = np.max(x)
     
     x = ((x-minVal) / (maxVal - minVal + 1.0e-20))
     
     return x                        
                        
def doubleplot(imageA, imageB, titleA, titleB):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes.ravel()
    ax[0].imshow(imageA, cmap=cm.Spectral)
    ax[0].set_title(titleA)
    
    ax[1].imshow(imageB, cmap=cm.Spectral)
    ax[1].set_title(titleB)
    
    plt.tight_layout()
    plt.show()                         