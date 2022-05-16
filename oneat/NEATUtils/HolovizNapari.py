#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 14:50:47 2021

@author: vkapoor
"""
from tifffile import imread,  imwrite
import csv
import napari
import glob
import os
import cv2
import random
import sys
import numpy as np
import json

from pathlib import Path
from scipy import spatial
from skimage.measure import label
import matplotlib.pyplot  as plt

from qtpy.QtCore import Qt


from oneat.NEATUtils.oneat_animation._qt import OneatWidget
from dask.array.image import imread as daskread

default_reader = 'tifffile'


class NEATViz(object):

        def __init__(self, imagedir,   
                        savedir, 
                        categories_json, 
                        imagereader = default_reader , 
                        heatmapimagedir = None, 
                        segimagedir = None, 
                        heatname = '_Heat', 
                        eventname = '_Event', 
                        fileextension = '*tif', 
                        blur_radius = 5, 
                        start_project_mid = 0, 
                        end_project_mid = 0 ):
            
            
               self.imagedir = imagedir
               self.heatmapimagedir = heatmapimagedir
               self.segimagedir = segimagedir
               self.savedir = savedir
               self.heatname = heatname
               self.eventname = eventname
        
               self.categories_json = categories_json
               self.start_project_mid = start_project_mid
               self.end_project_mid = end_project_mid
               self.fileextension = fileextension
               self.blur_radius = blur_radius
               self.imagereader = imagereader
               if self.imagereader == default_reader:
                   self.use_dask = False
               else:
                   self.use_dask = True    
               Path(self.savedir).mkdir(exist_ok=True)
               self.viewer = napari.Viewer()
               
               self.time = 0
               self.load_json()
               self.key_categories = self.load_json()
               
               
               
               self.showNapari()
               
        def load_json(self):
            with open(self.categories_json, 'r') as f:
                return json.load(f)      
            
            
        
        def showNapari(self):
                 
                 self.oneat_widget = OneatWidget(self.viewer, self.savedir, 'Name', 
                 self.key_categories, use_dask = self.use_dask, segimagedir = self.segimagedir,
                 heatimagedir = self.heatmapimagedir, heatname = self.heatname, 
                 start_project_mid = self.start_project_mid,
                 end_project_mid = self.end_project_mid )
                 Raw_path = os.path.join(self.imagedir, self.fileextension)
                 X = glob.glob(Raw_path)
                 Imageids = []
                 self.oneat_widget.frameWidget.imageidbox.addItem('Select Image')
                 self.oneat_widget.frameWidget.eventidbox.addItem('Select Event')
                 for imagename in X:
                     Imageids.append(imagename)

                 for i in range(0, len(Imageids)):
                     self.oneat_widget.frameWidget.imageidbox.addItem(str(Imageids[i]))
                 
                 for (event_name,event_label) in self.key_categories.items():
                     if event_label > 0:
                         self.oneat_widget.frameWidget.eventidbox.addItem(event_name)

                 dock_widget = self.viewer.window.add_dock_widget(self.oneat_widget, area='right')
                 self.viewer.window._qt_window.resizeDocks([dock_widget], [200], Qt.Horizontal)  

                 napari.run()
                 
        