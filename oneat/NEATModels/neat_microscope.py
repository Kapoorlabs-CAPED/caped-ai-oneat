#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 13:32:04 2021

@author: vkapoor
"""
from oneat.NEATUtils import plotters
import numpy as np
from oneat.NEATUtils import helpers
from oneat.NEATUtils.helpers import load_json, yoloprediction, normalizeFloatZeroOne,microscope_dynamic_nms
from keras import callbacks
import os
import tensorflow as tf
import time
from oneat.NEATModels.nets import Concat
from tqdm import tqdm
# from IPython.display import clear_output
from pathlib import Path
from keras.models import load_model
import csv
from natsort import natsorted
import glob
import matplotlib.pyplot as plt
import h5py
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from oneat.NEATModels.neat_goldstandard import NEATDynamic
from tifffile import imread, imwrite

class NEATPredict(NEATDynamic):
    

    def __init__(self, config, model_dir, model_name, catconfig=None, cordconfig=None):

          super().__init__(config = config, model_dir = model_dir, model_name = model_name, catconfig = catconfig, cordconfig = cordconfig)


    def predict_microscope(self, imagedir, Z_imagedir,  start = 0,
                Z_start = 0, downsample=1, roi_start = 0, roi_end = 1, movie_name_list = {}, movie_input = {}, Z_movie_name_list = [], Z_movie_input = [],
                fileextension='*TIF', nb_prediction=3, n_tiles=(1, 1), Z_n_tiles=(1, 2, 2),
                overlap_percent=0.6, event_threshold = 0.5, event_confidence = 0.5, iou_threshold=0.01, projection_model=None, delay_projection=4,
                fidelity=4, jumpindex = 1, normalize = True,  optional_name = None):

        self.imagedir = imagedir
        self.basedirResults = self.imagedir + '/' + "live_results"
        Path(self.basedirResults).mkdir(exist_ok=True)
        # Recurrsion variables
        self.movie_name_list = movie_name_list
        self.movie_input = movie_input
        self.Z_movie_name_list = Z_movie_name_list
        self.delay_projection = delay_projection
        self.Z_movie_input = Z_movie_input
        self.Z_imagedir = Z_imagedir
        self.start = start
        self.jumpindex = jumpindex
        self.fidelity = fidelity
        self.optional_name = optional_name
        self.Z_start = Z_start
        self.projection_model = projection_model
        self.nb_prediction = nb_prediction
        self.fileextension = fileextension
        self.n_tiles = n_tiles
        self.roi_start = roi_start
        self.roi_end = roi_end
        self.Z_n_tiles = Z_n_tiles
        self.overlap_percent = overlap_percent
        self.iou_threshold = iou_threshold
        self.event_threshold = event_threshold
        self.event_confidence = event_confidence
        self.downsample = downsample
        self.normalize = normalize
        f = h5py.File(self.model_dir + self.model_name + '.h5', 'r+')
        data_p = f.attrs['training_config']
        data_p = data_p.decode().replace("learning_rate", "lr").encode()
        f.attrs['training_config'] = data_p
        f.close()
        self.model = load_model(self.model_dir + self.model_name + '.h5',
                                custom_objects={'loss': self.yololoss, 'Concat': Concat})

        # Z slice folder listener
        while 1:

            Z_Raw_path = os.path.join(self.Z_imagedir, self.fileextension)
            Z_filesRaw = glob.glob(Z_Raw_path)
            Z_filesRaw = natsorted(Z_filesRaw)

            Raw_path = os.path.join(self.imagedir, '*tif')
            filesRaw = glob.glob(Raw_path)
            filesRaw = natsorted(filesRaw)
        
            for Z_movie_name in Z_filesRaw:
                Z_Name = os.path.basename(os.path.splitext(Z_movie_name)[0])
                # Check for unique filename
                if self.optional_name is not None:
                        if Z_Name not in self.Z_movie_name_list and self.optional_name in Z_Name:
                            self.Z_movie_name_list.append(Z_Name)
                            self.Z_movie_input.append(Z_movie_name)
                else:
                        if Z_Name not in self.Z_movie_name_list:
                            self.Z_movie_name_list.append(Z_Name)
                            self.Z_movie_input.append(Z_movie_name)        

            for movie_name in filesRaw:
                Name = os.path.basename(os.path.splitext(movie_name)[0])
                # Check for unique filename
                if Name not in self.movie_name_list:
                    self.movie_name_list[Name] = Name
                    self.movie_input[Name] = movie_name

                    total_movies = len(self.movie_input)

            doproject = True

            for i in range(len(self.Z_movie_name_list)):

                Z_Name = self.Z_movie_name_list[i]
                Z_path = self.Z_movie_input[i]

                if Z_Name in self.movie_name_list:

                    Name = self.movie_name_list[Z_Name]

                    doproject = False
                else:
                    doproject = True

                if doproject:
                    time.sleep(self.delay_projection)
                    try:
                        start_time = time.time()
                        print('Reading Z stack for projection')
                        Z_image = imread(Z_path)
                        print('Read properly')
                    except:

                        Z_image = None
                    if Z_image is not None:
                        if self.projection_model is not None:
                            print('Projecting using the projection model')
                            projection = self.projection_model.predict(Z_image, 'ZYX', n_tiles=Z_n_tiles)
                        else:
                            print('Doing max projection')
                            projection = np.amax(Z_image, axis=0)
                        imwrite(self.imagedir + '/' + Z_Name + '.tif', projection.astype('float32'))
                        print(f'____ Projection took {(time.time() - start_time)} seconds ____ ')

                    else:
                        if Z_Name in self.Z_movie_name_list:
                            self.Z_movie_name_list.remove(Z_Name)
                        if Z_movie_name in self.Z_movie_input:
                            self.Z_movie_input.remove(Z_movie_name)

            self.movie_input_list = []
            for (k, v) in self.movie_input.items():
                self.movie_input_list.append(v)
            total_movies = len(self.movie_input_list)
            if total_movies > self.size_tminus + self.start:
                current_movies = imread(self.movie_input_list[self.start:self.start + self.size_tminus + 1])

                sizey = current_movies.shape[1]
                sizex = current_movies.shape[2]
                if self.downsample !=1:
                    scale_percent = 100/self.downsample
                    width = int(sizey * scale_percent / 100)
                    height = int(sizex * scale_percent / 100)
                    dim = (width, height)
                    sizex = height
                    sizey = width

                    current_movies_down = np.zeros([current_movies.shape[0], sizey, sizex])
                    # resize image
                    for j in range(current_movies.shape[0]):
                        current_movies_down[j, :] = cv2.resize(current_movies[j, :], dim, interpolation=cv2.INTER_AREA)
                else:
                    current_movies_down = current_movies
                # print(current_movies_down.shape)
                print('Predicting on Movies:', self.movie_input_list[self.start:self.start + self.size_tminus + 1])
                inputtime = self.start + self.size_tminus

                eventboxes = []
                classedboxes = {}
                smallimage = CreateVolume(current_movies_down, self.size_tminus + 1, 0)
                if self.normalize:
                   smallimage = normalizeFloatZeroOne(smallimage, 1, 99.8)
                # Break image into tiles if neccessary
                self.image = smallimage
                print('Doing ONEAT prediction')
                start_time = time.time()
                predictions, allx, ally = self.predict_main(smallimage)
                print(f'____ Prediction took {(time.time() - start_time)} seconds ____ ' )

                # Iterate over tiles
                for p in tqdm(range(0, len(predictions))):

                    sum_time_prediction = predictions[p]

                    if sum_time_prediction is not None:
                        for i in range(0, sum_time_prediction.shape[0]):
                            time_prediction = sum_time_prediction[i]

                            boxprediction = yoloprediction(ally[p], allx[p], time_prediction, self.stride, inputtime,
                                                           self.config, self.key_categories, self.key_cord, self.nboxes,
                                                           'prediction', 'dynamic')

                            if boxprediction is not None:
                                eventboxes = eventboxes + boxprediction

                for (event_name, event_label) in self.key_categories.items():

                    if event_label > 0:
                        current_event_box = []
                        for box in eventboxes:

                            event_prob = box[event_name]
                            event_confidence = box['confidence']
                            if event_prob >= self.event_threshold and event_confidence >= self.event_confidence:
                                current_event_box.append(box)
                        classedboxes[event_name] = [current_event_box]

                self.classedboxes = classedboxes
                self.eventboxes = eventboxes
                print('Performining non maximal supression')
                start_time = time.time()
                self.iou_classedboxes = classedboxes
                self.nms_microscope()
                print(f'____ NMS took {(time.time() - start_time)} seconds ____ ')
                print('Generating ini file')
                self.to_csv_microscope()
              
                self.predict_microscope(self.imagedir, self.Z_imagedir,
                              start = self.start, Z_start = self.Z_start,
                             fileextension=self.fileextension, downsample=self.downsample,
                             roi_start = self.roi_start, roi_end = self.roi_end,
                             movie_name_list = self.movie_name_list, movie_input = self.movie_input,
                             Z_movie_name_list = self.Z_movie_name_list, Z_movie_input = self.Z_movie_input,
                             nb_prediction=self.nb_prediction, n_tiles=self.n_tiles, Z_n_tiles=self.Z_n_tiles,
                             overlap_percent=self.overlap_percent, event_threshold=self.event_threshold, event_confidence = self.event_confidence,
                             iou_threshold=self.iou_threshold, projection_model=self.projection_model, delay_projection = self.delay_projection, 
                             fidelity = self.fidelity, jumpindex = self.jumpindex, normalize = self.normalize, optional_name = self.optional_name)

    def nms_microscope(self):
        
        
        best_iou_classedboxes = {}
        self.iou_classedboxes = {}
        self.start = self.start + self.jumpindex
        for (event_name,event_label) in self.key_categories.items():
            if event_label > 0:
              
               best_sorted_event_box = microscope_dynamic_nms( self.classedboxes, event_name, self.iou_threshold, self.event_threshold, self.imagex, self.imagey, self.fidelity)
               
               
               best_iou_classedboxes[event_name] = [best_sorted_event_box]
               
        self.iou_classedboxes = best_iou_classedboxes

    def to_csv_microscope(self):

        for (event_name, event_label) in self.key_categories.items():

            if event_label > 0:

                xlocations = []
                ylocations = []
                scores = []
                tlocations = []
                radiuses = []
                confidences = []

                iou_current_event_boxes = self.iou_classedboxes[event_name][0]
                bbox_left_y = self.image.shape[1] * self.roi_start
                bbox_right_y = self.image.shape[1] * self.roi_end

                bbox_left_x = self.image.shape[2] * self.roi_start
                bbox_right_x = self.image.shape[2] * self.roi_end
                iou_current_event_boxes = sorted(iou_current_event_boxes, key=lambda x: x[event_name], reverse=True)

                for iou_current_event_box in iou_current_event_boxes:
                    xcenter = iou_current_event_box['xcenter']
                    ycenter = iou_current_event_box['ycenter']
                    tcenter = iou_current_event_box['real_time_event']
                    score = iou_current_event_box[event_name]
                    radius = np.sqrt(
                        iou_current_event_box['height'] * iou_current_event_box['height'] + iou_current_event_box[
                            'width'] * iou_current_event_box['width']) // 2
                    confidence = iou_current_event_box['confidence']
                    if xcenter >= bbox_left_x and xcenter <= bbox_right_x and ycenter >= bbox_left_y and ycenter <= bbox_right_y:
                        print(round(xcenter), round(ycenter), score)
                        xlocations.append(round(xcenter))
                        ylocations.append(round(ycenter))
                        scores.append(score)
                        tlocations.append(tcenter)
                        radiuses.append(radius)
                        confidences.append(confidence)

                event_count = np.column_stack([xlocations, ylocations])
                total_event_count = np.column_stack([tlocations, ylocations, xlocations, scores, radiuses, confidences])
                csvname = self.basedirResults + "/" + event_name
                if len(xlocations) > 0:
                    writer = csv.writer(open(csvname + ".ini", 'w'))
                    writer.writerow(["[main]"])
                    writer.writerow(["nbPredictions=" + str(self.nb_prediction)])
                    live_event_data = []
                    count = 1

                    for line in event_count:
                        if len(live_event_data) > self.nb_prediction:
                            break
                        live_event_data.append(line)
                        writer.writerow(["[" + str(count) + "]"])
                        writer.writerow(["x=" + str(live_event_data[0][0])])
                        writer.writerow(["y=" + str(live_event_data[0][1])])
                        live_event_data = []

                        count = count + 1

                ImageResults = self.basedirResults + '/' + 'ImageLocations'
                Path(ImageResults).mkdir(exist_ok=True)

                event_data = []
                writer = csv.writer(open(csvname + ".csv", "a"))
                filesize = os.stat(csvname + ".csv").st_size
                if filesize < 1:
                    writer.writerow(['T', 'Y', 'X', 'Score', 'Size', 'Confidence'])
                for line in total_event_count:
                    if line not in event_data:
                        event_data.append(line)
                    writer.writerows(event_data)
                    event_data = []


def CreateVolume(patch, imaget, timepoint):
    starttime = timepoint
    endtime = timepoint + imaget
    smallimg = patch[starttime:endtime, :]

    return smallimg
