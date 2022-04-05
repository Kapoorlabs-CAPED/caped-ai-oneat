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
import cv2
import random
import sys
import numpy as np
import json
from scipy import spatial
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
from skimage import measure
from scipy.ndimage import gaussian_filter
import pandas as pd
import imageio
from oneat.NEATUtils.napari_animation._qt import AnimationWidget
from dask.array.image import imread as daskread
Boxname = 'ImageIDBox'
EventBoxname = 'EventIDBox'


class NEATViz(object):

        def __init__(self, imagedir, heatmapimagedir, savedir, categories_json, event_threshold, heatname = '_Heat', fileextension = '*tif', blur_radius = 5, start_project_mid = 0, end_project_mid = 0 ):
            
            
               self.imagedir = imagedir
               self.heatmapimagedir = heatmapimagedir
               self.savedir = savedir
               self.heatname = heatname
               self.event_threshold = event_threshold
               self.categories_json = categories_json
               self.start_project_mid = start_project_mid
               self.end_project_mid = end_project_mid
               self.fileextension = fileextension
               self.blur_radius = blur_radius
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
                 
                 
                 Raw_path = os.path.join(self.imagedir, self.fileextension)
                 X = glob.glob(Raw_path)
                 Imageids = []
                 
                 for imagename in X:
                     Imageids.append(imagename)
                 
                 
                 eventidbox = QComboBox()
                 eventidbox.addItem(EventBoxname)
                 for (event_name,event_label) in self.key_categories.items():
                     
                     eventidbox.addItem(event_name)
                    
                 imageidbox = QComboBox()   
                 imageidbox.addItem(Boxname)   
                 detectionsavebutton = QPushButton('Save Clicks')
                 
                 for i in range(0, len(Imageids)):
                     
                     
                     imageidbox.addItem(str(Imageids[i]))
                     
                     
                 self.figure = plt.figure(figsize=(4, 4))
                 self.multiplot_widget = FigureCanvas(self.figure)
                 self.ax = self.multiplot_widget.figure.subplots(1, 1)
                 width = 400
                 dock_widget = self.viewer.window.add_dock_widget(
                 self.multiplot_widget, name="EventStats", area='right')
                 self.multiplot_widget.figure.tight_layout()
                 self.viewer.window._qt_window.resizeDocks([dock_widget], [width], Qt.Horizontal)   

                 
                 self.animation_widget = AnimationWidget(self.viewer, self.savedir, 'Name', 0, 100)
                 self.viewer.window.add_dock_widget(self.animation_widget, area='right')
                 self.viewer.update_console({'animation': self.animation_widget.animation})
                 
                 eventidbox.currentIndexChanged.connect(lambda eventid = eventidbox : self.csv_add(
                         
                         
                         os.path.basename(os.path.splitext(imageidbox.currentText())[0]),
                         eventidbox.currentText()
                    
                )
            )    
                 
                 imageidbox.currentIndexChanged.connect(
                 lambda trackid = imageidbox: self.image_add(
                         
                         imageidbox.currentText(),
                         
                         os.path.basename(os.path.splitext(imageidbox.currentText())[0])
                    
                )
            )            
                 
                    
                 self.viewer.window.add_dock_widget(imageidbox, name="Image", area='left') 
                 self.viewer.window.add_dock_widget(eventidbox, name="Event", area='left')  
                 
                 
                        
        def csv_add(self, imagename, csv_event_name ):
            
            
            self.event_name = csv_event_name
            self.animation_widget = AnimationWidget(self.viewer, self.savedir, imagename + csv_event_name, 0, self.image.shape[0])
            self.csvname = None
            for (event_name,event_label) in self.key_categories.items():
                                
                                if event_label > 0 and csv_event_name == event_name:
                                     self.event_label = event_label                         
                                     for layer in list(self.viewer.layers):
                                          
                                         if 'Detections'  in layer.name or layer.name in 'Detections' :
                                                    self.viewer.layers.remove(layer)           
                                       
                                     
                                     self.csvname = self.savedir + "/" + event_name + "Location" + (os.path.splitext(os.path.basename(imagename))[0] + '.csv')
                                     
            self.dataset   = pd.read_csv(self.csvname, delimiter = ',')
            self.dataset_index = self.dataset.index
            self.ax.cla()
            #Data is written as T, Y, X, Score, Size, Confidence
            self.T = self.dataset[self.dataset.keys()[0]][0:]
            self.Z = self.dataset[self.dataset.keys()[1]][0:]
            self.Y = self.dataset[self.dataset.keys()[2]][0:]
            self.X = self.dataset[self.dataset.keys()[3]][0:]
            self.Score = self.dataset[self.dataset.keys()[4]][0:]
            self.Size = self.dataset[self.dataset.keys()[5]][0:]
            self.Confidence = self.dataset[self.dataset.keys()[6]][0:]
            
            
            
            timelist = []
            eventlist= []
            for i in range(0, self.image.shape[0]):
                
                   currentT   = np.round(self.dataset["T"]).astype('int')
                   currentZ = np.round(self.dataset["Z"]).astype('int')
                   currentScore = self.dataset["Score"]
                   currentConf = self.dataset["Confidence"]
                   condition = currentT == i
                   condition_indices = self.dataset_index[condition]
                   conditionScore = currentScore[condition_indices]
                
                   score_condition = conditionScore > self.event_threshold[self.event_label]
                
                   countT = len(conditionScore[score_condition])
                   timelist.append(i)
                   eventlist.append(countT)
                
                
            self.ax.plot(timelist, eventlist, '-r')
            self.ax.set_title(self.event_name + "Events")
            self.ax.set_xlabel("Time")
            self.ax.set_ylabel("Counts")
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()
            plt.savefig(self.savedir  + self.event_name   + '.png')        

            listtime = self.T.tolist()
            listz = self.Z.tolist()
            listy = self.Y.tolist()
            listx = self.X.tolist()
            
            listsize = self.Size.tolist()
            listscore = self.Score.tolist()
            listconfidence = self.Confidence.tolist()
            event_locations = []
            size_locations = []
            score_locations = []
            confidence_locations = []
            ndim = len(self.image)
            for i in (range(len(listtime))):
                 
                 tcenter = int(listtime[i])
                 zcenter = listz[i]
                 ycenter = listy[i]
                 xcenter = listx[i]
                 size = listsize[i]
                 score = listscore[i]
                 confidence = listconfidence[i]   
                 if score > self.event_threshold[self.event_label]:
                         event_locations.append([int(tcenter), int(ycenter), int(xcenter)])   
                         size_locations.append(size)
                         score_locations.append(score)
                         confidence_locations.append(confidence)
            point_properties = {'score' : np.array(score_locations), 'confidence' : np.array(confidence_locations)}    
            text_properties = {
            'text': self.event_name +': {score:.5f}' + '\n' + 'Confidence' +  ': {confidence:.5f}',
            'anchor': 'upper_left',
            'translation': [-5, 0],
            'size': 12,
            'color': 'pink',
        }
            for layer in list(self.viewer.layers):
                              
                             if 'Detections'  in layer.name or layer.name in 'Detections' :
                                        self.viewer.layers.remove(layer) 
            if len(score_locations) > 0:                             
                   self.viewer.add_points(event_locations, size = size_locations , properties = point_properties, text = text_properties,  name = 'Detections' + self.event_name, face_color = [0]*4, edge_color = "red", edge_width = 4) 
                   

                                     
                                        
            
        def image_add(self, image_toread, imagename):
                                    
                for layer in list(self.viewer.layers):
                                         if 'Image' in layer.name or layer.name in 'Image':
                                                    self.viewer.layers.remove(layer)
                                                    
                                                    
                self.image = imread(image_toread)
                
                if self.heatmapimagedir is not None:
                     self.heat_image = imread(self.heatmapimagedir + imagename + self.heatname + '.tif')
                    
                if len(self.image.shape) == 4:
                    self.image =  MidSlices(self.image, self.start_project_mid, self.end_project_mid, axis = 1)
                    
                
                self.viewer.add_image(self.image, name= 'Image' + imagename )
                if self.heatmapimagedir is not None:
                     self.viewer.add_image(self.heat_image, name= 'Image' + imagename + self.heatname )
                                                    
def MidSlices(Image, start_project_mid, end_project_mid, axis = 1):
    
    
    SmallImage = Image.take(indices = range(Image.shape[axis]//2 - start_project_mid, Image.shape[axis]//2 + end_project_mid), axis = axis)
    
    MaxProject = np.amax(SmallImage, axis = axis)
        
    return MaxProject  

def TruePositives(csv_gt, csv_pred, thresholdscore = 1 -  1.0E-6,  thresholdspace = 10, thresholdtime = 2):
    
            
            try:
                
                    tp = 0
                  

                    dataset_pred  = pd.read_csv(csv_pred, delimiter = ',')
                    dataset_pred_index = dataset_pred.index

                    T_pred = dataset_pred[dataset_pred.keys()[0]][0:]
                    Y_pred = dataset_pred[dataset_pred.keys()[1]][0:]
                    X_pred = dataset_pred[dataset_pred.keys()[2]][0:]
                    Score_pred = dataset_pred[dataset_pred.keys()[3]][0:]
                    
                    
                    listtime_pred = T_pred.tolist()
                    listy_pred = Y_pred.tolist()
                    listx_pred = X_pred.tolist()
                    listscore_pred = Score_pred.tolist()
                    location_pred = []
                    for i in range(len(listtime_pred)):

                        if listscore_pred[i] > thresholdscore:   
                            location_pred.append([listtime_pred[i], listy_pred[i], listx_pred[i]])

                    tree = spatial.cKDTree(location_pred)


                    dataset_gt  = pd.read_csv(csv_gt, delimiter = ',')
                    dataset_gt_index = dataset_gt.index

                    T_gt = dataset_gt[dataset_gt.keys()[0]][0:]
                    Y_gt = dataset_gt[dataset_gt.keys()[1]][0:]
                    X_gt = dataset_gt[dataset_gt.keys()[2]][0:]

                    listtime_gt = T_gt.tolist()
                    listy_gt = Y_gt.tolist()
                    listx_gt = X_gt.tolist()
                    location_gt = []
                    for i in range(len(listtime_gt)):
                        
                        index = [float(listtime_gt[i]), float(listy_gt[i]), float(listx_gt[i])]
                        closestpoint = tree.query(index)
                        spacedistance, timedistance = TimedDistance(index, location_pred[closestpoint[1]])
                        
                        if spacedistance < thresholdspace and timedistance < thresholdtime:
                            tp  = tp + 1
                    
                    fn = FalseNegatives(csv_pred, csv_gt, thresholdscore = thresholdscore, thresholdspace = thresholdspace, thresholdtime = thresholdtime)
                    fp = FalsePositives(csv_pred, csv_gt, thresholdscore = thresholdscore, thresholdspace = thresholdspace, thresholdtime = thresholdtime)
                    return tp/len(listtime_gt) * 100, fn, fp
                
            except:
                 
                 return 'File not found'
                 pass

 
def DownsampleData(image, DownsampleFactor):
                    
                    if DownsampleFactor!=1:  
                  
                        print('Downsampling Image in XY by', DownsampleFactor)
                        scale_percent = int(100/DownsampleFactor) # percent of original size
                        width = int(image.shape[2] * scale_percent / 100)
                        height = int(image.shape[1] * scale_percent / 100)
                        dim = (width, height)
                        smallimage = np.zeros([image.shape[0],  height,width])
                        for i in range(0, image.shape[0]):
                              # resize image
                              smallimage[i,:] = cv2.resize(image[i,:].astype('float32'), dim)         

                        return smallimage
                    else:

                        return image
                
def PatchGenerator(image,resultsdir,csv_gt,number_patches, patch_shape, size_tminus,size_tplus,DownsampleFactor = 1 ):
    
    
                    image = DownsampleData(image, DownsampleFactor)
                    dataset_gt  = pd.read_csv(csv_gt, delimiter = ',')
            
                    dataset_gt = dataset_gt.sample(frac = 1)
                    dataset_gt_index = dataset_gt.index
                    T_gt = dataset_gt[dataset_gt.keys()[0]][0:]
                    Y_gt = dataset_gt[dataset_gt.keys()[1]][0:]/DownsampleFactor
                    X_gt = dataset_gt[dataset_gt.keys()[2]][0:]/DownsampleFactor

                    listtime_gt = T_gt.tolist()
                    
                    listy_gt = Y_gt.tolist()
                    listx_gt = X_gt.tolist()
                    location_gt = []
                    fn = len(listtime_gt)
                    count = 0
                    Data = []
                    for i in range(len(listtime_gt)):
                        if count >  2 * number_patches:
                            break
                        time = int(float(listtime_gt[i])) - 1
                        y = float(listy_gt[i])
                        x = float(listx_gt[i])
                        
                        if x > 0.25 * image.shape[2] and x < 0.75* image.shape[2] and y > 0.25 * image.shape[1] and y < 0.75* image.shape[1]:
                                crop_Xminus = x - int(patch_shape[0] / 2)
                                crop_Xplus = x + int(patch_shape[0] / 2)
                                crop_Yminus = y - int(patch_shape[1] / 2)
                                crop_Yplus = y + int(patch_shape[1] / 2)

                          
                                randomy = np.random.randint(min(0.25 * image.shape[2],0.25 * image.shape[1]), high=max(0.25 * image.shape[2],0.25 * image.shape[1]))
                                randomx = np.random.randint(min(0.25 * image.shape[2],0.25 * image.shape[1]), high=max(0.25 * image.shape[2],0.25 * image.shape[1]))
                                random_crop_Xminus = randomx - int(patch_shape[0] / 2)
                                random_crop_Xplus = randomx + int(patch_shape[0] / 2)
                                random_crop_Yminus = randomy - int(patch_shape[1] / 2)
                                random_crop_Yplus = randomy + int(patch_shape[1] / 2)

                                region = (slice(int(time - size_tminus),int(time + size_tplus  + 1)),slice(int(crop_Yminus), int(crop_Yplus)),
                                          slice(int(crop_Xminus), int(crop_Xplus)))

                                random_region = (slice(int(time - size_tminus),int(time + size_tplus  + 1)),slice(int(random_crop_Yminus), int(random_crop_Yplus)),
                                          slice(int(random_crop_Xminus), int(random_crop_Xplus)))


                                crop_image = image[region] 
                                random_crop_image = image[random_region]
                                if(crop_image.shape[0] == size_tplus + size_tminus + 1 and crop_image.shape[1]== patch_shape[1] and crop_image.shape[2]== patch_shape[0]):
                                      Data.append([time, y * DownsampleFactor, x * DownsampleFactor])
                                      imwrite(resultsdir + 'Skeletor' + 'T' +  str(time) + 'Y' + str(y*DownsampleFactor) + 'X' + str(x*DownsampleFactor) + '.tif', crop_image.astype('float16'),metadata={'axes': 'TYX'})                
                                count = count + 1        
                                if(random_crop_image.shape[0] == size_tplus + size_tminus + 1 and random_crop_image.shape[1]== patch_shape[1] and random_crop_image.shape[2]== patch_shape[0]):
                                      Data.append([time, randomy * DownsampleFactor, randomx * DownsampleFactor])
                                      imwrite(resultsdir + 'Skeletor' + 'T' + str(time) + 'Y' + str(randomy*DownsampleFactor) + 'X' + str(randomx*DownsampleFactor) + '.tif', random_crop_image.astype('float16'),metadata={'axes': 'TYX'})
                                count = count + 1 
                    
                    writer = csv.writer(open(resultsdir + '/' + ('GTLocator') + ".csv", "w"))
                    writer.writerows(Data)
                    
                    
def FalseNegatives(csv_pred, csv_gt, thresholdscore = 1 -  1.0E-6, thresholdspace = 10, thresholdtime = 2):
    
            
            try:
                
                    
                  

                    dataset_pred  = pd.read_csv(csv_pred, delimiter = ',')
                    dataset_pred_index = dataset_pred.index

                    T_pred = dataset_pred[dataset_pred.keys()[0]][0:]
                    Y_pred = dataset_pred[dataset_pred.keys()[1]][0:]
                    X_pred = dataset_pred[dataset_pred.keys()[2]][0:]
                    Score_pred = dataset_pred[dataset_pred.keys()[3]][0:]
                    
                    listtime_pred = T_pred.tolist()
                    listy_pred = Y_pred.tolist()
                    listx_pred = X_pred.tolist()
                    listscore_pred = Score_pred.tolist()
                    location_pred = []
                    for i in range(len(listtime_pred)):
                        
                        
                        if listscore_pred[i] > thresholdscore:
                           location_pred.append([listtime_pred[i], listy_pred[i], listx_pred[i]])

                    tree = spatial.cKDTree(location_pred)


                    dataset_gt  = pd.read_csv(csv_gt, delimiter = ',')
                    dataset_gt_index = dataset_gt.index

                    T_gt = dataset_gt[dataset_gt.keys()[0]][0:]
                    Y_gt = dataset_gt[dataset_gt.keys()[1]][0:]
                    X_gt = dataset_gt[dataset_gt.keys()[2]][0:]

                    listtime_gt = T_gt.tolist()
                    listy_gt = Y_gt.tolist()
                    listx_gt = X_gt.tolist()
                    location_gt = []
                    fn = len(listtime_gt)
                    for i in range(len(listtime_gt)):
                        
                        index = [float(listtime_gt[i]), float(listy_gt[i]), float(listx_gt[i])]
                        closestpoint = tree.query(index)
                        spacedistance, timedistance = TimedDistance(index, location_pred[closestpoint[1]])

                        if spacedistance < thresholdspace and timedistance < thresholdtime:
                            fn  = fn - 1

                            


                    return fn/len(listtime_gt) * 100
                
            except:
                 
                 return 'File not found'
                 pass             
                
def FalsePositives(csv_pred, csv_gt, thresholdscore = 1 -  1.0E-6, thresholdspace = 10, thresholdtime = 2):
    
            
            try:
                
                    
                  

                    dataset_pred  = pd.read_csv(csv_pred, delimiter = ',')
                    dataset_pred_index = dataset_pred.index

                    T_pred = dataset_pred[dataset_pred.keys()[0]][0:]
                    Y_pred = dataset_pred[dataset_pred.keys()[1]][0:]
                    X_pred = dataset_pred[dataset_pred.keys()[2]][0:]
                    Score_pred = dataset_pred[dataset_pred.keys()[3]][0:]
                    
                    listtime_pred = T_pred.tolist()
                    listy_pred = Y_pred.tolist()
                    listx_pred = X_pred.tolist()
                    listscore_pred = Score_pred.tolist()
                    location_pred = []
                    


                    dataset_gt  = pd.read_csv(csv_gt, delimiter = ',')
                    dataset_gt_index = dataset_gt.index

                    T_gt = dataset_gt[dataset_gt.keys()[0]][0:]
                    Y_gt = dataset_gt[dataset_gt.keys()[1]][0:]
                    X_gt = dataset_gt[dataset_gt.keys()[2]][0:]

                    listtime_gt = T_gt.tolist()
                    listy_gt = Y_gt.tolist()
                    listx_gt = X_gt.tolist()
                    location_gt = []
                    fp = len(listtime_pred)
                    
                    for i in range(len(listtime_gt)):
                        
                     
                           location_gt.append([listtime_gt[i], listy_gt[i], listx_gt[i]])

                    tree = spatial.cKDTree(location_gt)
                    for i in range(len(listtime_pred)):
                        
                        index = [float(listtime_pred[i]), float(listy_pred[i]), float(listx_pred[i])]
                        closestpoint = tree.query(index)
                        spacedistance, timedistance = TimedDistance(index, location_gt[closestpoint[1]])

                        if spacedistance < thresholdspace and timedistance < thresholdtime:
                            fp  = fp - 1

                            


                    return fp/len(listtime_pred) * 100
                
            except:
                 
                 return 'File not found'
                 pass             
                                
 
def TimedDistance(pointA, pointB):

    
     spacedistance = float(np.sqrt( (pointA[1] - pointB[1] ) * (pointA[1] - pointB[1] ) + (pointA[2] - pointB[2] ) * (pointA[2] - pointB[2] )  ))
     
     timedistance = float(np.abs(pointA[0] - pointB[0]))
     
     
     return spacedistance, timedistance
                
                
def GetMarkers(image):
    
    
    MarkerImage = np.zeros(image.shape)
    waterproperties = measure.regionprops(image)                
    Coordinates = [prop.centroid for prop in waterproperties]
    Coordinates = sorted(Coordinates , key=lambda k: [k[0], k[1]])
    MarkerImage[tuple(coordinates_int.T)] = 1 + np.arange(len(Coordinates))

    markers = morphology.dilation(MarkerImage, morphology.disk(2))        
   
    return markers 