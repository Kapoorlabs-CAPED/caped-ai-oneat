from oneat.NEATUtils import plotters
import numpy as np
from oneat.NEATUtils import helpers
from oneat.NEATUtils.helpers import MidSlices, get_nearest,  load_json, yoloprediction, normalizeFloatZeroOne, GenerateMarkers, Generate_only_mask, MakeTrees, DownsampleData,save_dynamic_csv, dynamic_nms, gold_nms
from keras import callbacks
import os
import keras
from scipy.ndimage.morphology import binary_dilation, binary_erosion
import math
from tqdm import tqdm
from oneat.NEATModels import nets
from oneat.NEATModels.nets import Concat
from oneat.NEATModels.loss import dynamic_yolo_loss
from keras import backend as K
import tensorflow as tf
from tensorflow.keras import optimizers
from pathlib import Path
from keras.models import load_model
from tifffile import imread, imwrite
import napari
import glob
from skimage.morphology import erosion, dilation, disk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox, QPushButton
import cv2
from scipy import ndimage
from skimage.measure import label
from skimage import measure
Boxname = 'ImageIDBox'
EventBoxname = 'EventIDBox'


class NEATDynamic(object):
    """
    Parameters
    ----------
    
    NpzDirectory : Specify the location of npz file containing the training data with movies and labels
    
    TrainModelName : Specify the name of the npz file containing training data and labels
    
    ValidationModelName :  Specify the name of the npz file containing validation data and labels
    
    categories : Number of action classes
    
    Categories_Name : List of class names and labels
    
    model_dir : Directory location where trained model weights are to be read or written from
    
    model_name : The h5 file of CNN + LSTM + Dense Neural Network to be used for training
    
    model_keras : The model as it appears as a Keras function
    
    model_weights : If re-training model_weights = model_dir + model_name else None as default
    
    lstm_hidden_units : Number of hidden uniots for LSTm layer, 64 by default
    
    epochs :  Number of training epochs, 55 by default
    
    batch_size : batch_size to be used for training, 20 by default
    
    
    
    """

    def __init__(self, config, model_dir, model_name,  catconfig=None, cordconfig=None):

        self.config = config
        self.catconfig = catconfig
        self.cordconfig = cordconfig
        self.model_dir = model_dir
        self.model_name = model_name
        if self.config != None:
            self.npz_directory = config.npz_directory
            self.npz_name = config.npz_name
            self.npz_val_name = config.npz_val_name
            self.key_categories = config.key_categories
            self.stage_number = config.stage_number
            self.last_conv_factor = config.last_conv_factor
            self.show = config.show
            self.key_cord = config.key_cord
            self.box_vector = len(config.key_cord)
            self.categories = len(config.key_categories)
            self.depth = config.depth
            self.start_kernel = config.start_kernel
            self.mid_kernel = config.mid_kernel
            self.lstm_kernel = config.lstm_kernel
            self.learning_rate = config.learning_rate
            self.epochs = config.epochs
            self.residual = config.residual
            self.startfilter = config.startfilter
            self.batch_size = config.batch_size
            self.multievent = config.multievent
            self.imagex = config.imagex
            self.imagey = config.imagey
            self.imaget = config.size_tminus + config.size_tplus + 1
            self.size_tminus = config.size_tminus
            self.size_tplus = config.size_tplus

            self.nboxes = config.nboxes
            self.gridx = 1
            self.gridy = 1
            self.gridt = 1
            self.yolo_v0 = config.yolo_v0
            self.yolo_v1 = config.yolo_v1
            self.yolo_v2 = config.yolo_v2
            self.stride = config.stride
            self.lstm_hidden_unit = config.lstm_hidden_unit
        if self.config == None:

            try:
                self.config = load_json(self.model_dir + os.path.splitext(self.model_name)[0] + '_Parameter.json')
            except:
                self.config = load_json(self.model_dir + self.model_name + '_Parameter.json')

            self.npz_directory = self.config['npz_directory']
            self.npz_name = self.config['npz_name']
            self.npz_val_name = self.config['npz_val_name']
            self.key_categories = self.catconfig
            self.box_vector = self.config['box_vector']
            self.show = self.config['show']
            self.key_cord = self.cordconfig
            self.categories = len(self.catconfig)
            self.depth = self.config['depth']
            self.start_kernel = self.config['start_kernel']
            self.mid_kernel = self.config['mid_kernel']
            self.lstm_kernel = self.config['lstm_kernel']
            self.lstm_hidden_unit = self.config['lstm_hidden_unit']
            self.learning_rate = self.config['learning_rate']
            self.epochs = self.config['epochs']
            self.residual = self.config['residual']
            self.startfilter = self.config['startfilter']
            self.batch_size = self.config['batch_size']
            self.multievent = self.config['multievent']
            self.imagex = self.config['imagex']
            self.imagey = self.config['imagey']
            self.imaget = self.config['size_tminus'] + self.config['size_tplus'] + 1
            self.size_tminus = self.config['size_tminus']
            self.size_tplus = self.config['size_tplus']
            self.nboxes = self.config['nboxes']
            self.stage_number = self.config['stage_number']
            self.last_conv_factor = self.config['last_conv_factor']
            self.gridx = 1
            self.gridy = 1
            self.gridt = 1
            self.yolo_v0 = self.config['yolo_v0']
            self.yolo_v1 = self.config['yolo_v1']
            self.yolo_v2 = self.config['yolo_v2']
            self.stride = self.config['stride']
            self.lstm_hidden_unit = self.config['lstm_hidden_unit']

        self.X = None
        self.Y = None
        self.axes = None
        self.X_val = None
        self.Y_val = None
        self.Trainingmodel = None
        self.Xoriginal = None
        self.Xoriginal_val = None

        if self.residual:
            self.model_keras = nets.ORNET
        else:
            self.model_keras = nets.OSNET

        if self.multievent == True:
            self.last_activation = 'sigmoid'
            self.entropy = 'binary'

        if self.multievent == False:
            self.last_activation = 'softmax'
            self.entropy = 'notbinary'
        self.yololoss = dynamic_yolo_loss(self.categories, self.gridx, self.gridy, self.gridt, self.nboxes,
                                          self.box_vector, self.entropy, self.yolo_v0, self.yolo_v1, self.yolo_v2)

    def loadData(self):

        (X, Y), axes = helpers.load_full_training_data(self.npz_directory, self.npz_name, verbose=True)

        (X_val, Y_val), axes = helpers.load_full_training_data(self.npz_directory, self.npz_val_name, verbose=True)

        self.Xoriginal = X
        self.Xoriginal_val = X_val

        self.X = X
        self.Y = Y[:, :, 0]
        self.X_val = X_val
        self.Y_val = Y_val[:, :, 0]

        self.axes = axes
        self.Y = self.Y.reshape((self.Y.shape[0], 1, 1, self.Y.shape[1]))
        self.Y_val = self.Y_val.reshape((self.Y_val.shape[0], 1, 1, self.Y_val.shape[1]))

    def TrainModel(self):

        input_shape = (self.X.shape[1], self.X.shape[2], self.X.shape[3], self.X.shape[4])

        Path(self.model_dir).mkdir(exist_ok=True)

        if self.yolo_v2:

            for i in range(self.Y.shape[0]):

                if self.Y[i, :, :, 0] == 1:
                    self.Y[i, :, :, -1] = 1
            for i in range(self.Y_val.shape[0]):

                if self.Y_val[i, :, :, 0] == 1:
                    self.Y_val[i, :, :, -1] = 1
        Y_rest = self.Y[:, :, :, self.categories:]
 

        model_weights = self.model_dir + self.model_name
        if os.path.exists(model_weights):

            self.model_weights = model_weights
            print('loading weights')
        else:

            self.model_weights = None

        dummyY = np.zeros(
            [self.Y.shape[0], self.Y.shape[1], self.Y.shape[2], self.categories + self.nboxes * self.box_vector])
        dummyY[:, :, :, :self.Y.shape[3]] = self.Y

        dummyY_val = np.zeros([self.Y_val.shape[0], self.Y_val.shape[1], self.Y_val.shape[2],
                               self.categories + self.nboxes * self.box_vector])
        dummyY_val[:, :, :, :self.Y_val.shape[3]] = self.Y_val

        for b in range(1, self.nboxes):
            dummyY[:, :, :, self.categories + b * self.box_vector:self.categories + (b + 1) * self.box_vector] = self.Y[
                                                                                                                 :, :,
                                                                                                                 :,
                                                                                                                 self.categories: self.categories + self.box_vector]
            dummyY_val[:, :, :,
            self.categories + b * self.box_vector:self.categories + (b + 1) * self.box_vector] = self.Y_val[:, :, :,
                                                                                                 self.categories: self.categories + self.box_vector]

        self.Y = dummyY
        self.Y_val = dummyY_val


        self.Trainingmodel = self.model_keras(input_shape, self.categories, unit=self.lstm_hidden_unit,
                                              box_vector=Y_rest.shape[-1], nboxes=self.nboxes,
                                              stage_number=self.stage_number, last_conv_factor=self.last_conv_factor,
                                              depth=self.depth, start_kernel=self.start_kernel,
                                              mid_kernel=self.mid_kernel, lstm_kernel=self.lstm_kernel,
                                              startfilter=self.startfilter, input_weights=self.model_weights,
                                              last_activation=self.last_activation)

        sgd = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.Trainingmodel.compile(optimizer=sgd, loss=self.yololoss, metrics=['accuracy'])

        self.Trainingmodel.summary()
        # Keras callbacks
        lrate = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=4, verbose=1)
        hrate = callbacks.History()
        srate = callbacks.ModelCheckpoint(self.model_dir + self.model_name, monitor='loss', verbose=1,
                                          save_best_only=False, save_weights_only=False, mode='auto', period=1)
        prate = plotters.PlotHistory(self.Trainingmodel, self.X_val, self.Y_val, self.key_categories, self.key_cord,
                                     self.gridx, self.gridy, plot=self.show, nboxes=self.nboxes)

        # Train the model and save as a h5 file
        self.Trainingmodel.fit(self.X, self.Y, batch_size=self.batch_size,
                               epochs=self.epochs, validation_data=(self.X_val, self.Y_val), shuffle=True,
                               callbacks=[lrate, hrate, srate, prate])

        # Removes the old model to be replaced with new model, if old one exists
        if os.path.exists(self.model_dir + self.model_name):
            os.remove(self.model_dir + self.model_name)

        self.Trainingmodel.save(self.model_dir + self.model_name)

    def get_markers(self, imagename, savedir, n_tiles,  segdir = None, use_seg_only = True, starmodel = None, maskmodel = None, start_project_mid = 4, end_project_mid = 4,
     downsamplefactor = 1, remove_markers = True):

        self.starmodel = starmodel
        self.imagename = imagename
        self.image = imread(imagename)
        self.segdir = segdir
        self.maskmodel = maskmodel
        Name = os.path.basename(os.path.splitext(self.imagename)[0])
        self.savedir = savedir
        Path(self.savedir).mkdir(exist_ok=True)
        self.downsamplefactor = downsamplefactor
        self.n_tiles = n_tiles
        print('Obtaining Markers, Mask and Watershed image')
        if self.segdir is None:
            self.markers, self.watershed, self.stardist, self.mask = GenerateMarkers(self.image, self.n_tiles, model = self.starmodel, maskmodel = self.maskmodel, segimage = self.segimage, 
            start_project_mid = start_project_mid, end_project_mid = end_project_mid)
            self.segdir = self.savedir + '/' + 'Segmentation'
            Path(self.segdir).mkdir(exist_ok=True)
            imwrite(self.segdir + '/' + Name + '_markers' + '.tif', self.markers.astype('int32'))
            imwrite(self.segdir + '/' + Name + '_vollseg' + '.tif', self.watershed.astype('int32'))
            imwrite(self.segdir + '/' + Name + '_stardist' + '.tif', self.stardist.astype('int32'))
            imwrite(self.segdir + '/' + Name + '_mask' + '.tif', self.mask.astype('uint16'))


        else:
            try:
                self.markers = imread(self.segdir + '/' + Name + '_markers' + '.tif')
                self.watershed = imread(self.segdir + '/' + Name + '_vollseg' + '.tif')
                self.stardist = imread(self.segdir + '/' + Name + '_stardist' + '.tif')
                self.mask = imread(self.segdir + '/' + Name + '_mask' + '.tif')
                if remove_markers:
                    self.markers = DownsampleData(self.markers, self.downsamplefactor)
                for i in range(0, self.markers.shape[0]):
                    self.markers[i,:] = self.markers[i,:] > 0
                    self.markers[i,:] = label(self.markers[i,:].astype('uint16'))
                    


            except:
        
                if use_seg_only:
                    self.segimage = imread(self.segdir + '/' + Name + '.tif')
                self.markers, self.watershed, self.stardist, self.mask = GenerateMarkers(self.image, self.n_tiles, model = self.starmodel, maskmodel = self.maskmodel, segimage = self.segimage, 
            start_project_mid = start_project_mid, end_project_mid = end_project_mid)
                self.segdir = self.savedir + '/' + 'Segmentation'
                Path(self.segdir).mkdir(exist_ok=True)
                imwrite(self.segdir + '/' + Name + '_markers' + '.tif', self.markers.astype('int32'))
        self.marker_tree = MakeTrees(self.markers)


        return self.markers, self.marker_tree, self.watershed, self.stardist, self.mask
    
    def predict(self, imagename,  savedir, n_tiles=(1, 1), overlap_percent=0.8,
                event_threshold=0.5, event_confidence = 0.5, iou_threshold=0.1,  fidelity=1, downsamplefactor = 1, start_project_mid = 4, end_project_mid = 4,
                erosion_iterations = 1, markers = None, marker_tree = None, watershed = None, stardist = None, remove_markers = True, maskmodel = None, segdir = None, normalize = True):


        self.watershed = watershed
        self.stardist = stardist
        self.segdir = segdir
        self.imagename = imagename
        self.Name = os.path.basename(os.path.splitext(self.imagename)[0])
        if self.segdir is not None:
            try:
                self.maskimage = imread(self.segdir + '/' + self.Name + '_mask' + '.tif')
            except:
                self.maskimage = None    
        else:
            self.maskimage = None    
        self.image = imread(imagename)
        self.start_project_mid = start_project_mid
        self.end_project_mid = end_project_mid
        self.maskmodel = maskmodel
        self.ndim = len(self.image.shape)
        self.normalize = normalize
        self.z = 0
        if self.ndim == 4:
           self.z = self.image.shape[1]//2
           print(f'Image {self.image.shape} is {self.ndim} dimensional, projecting around the center {self.image.shape[1]//2} - {self.start_project_mid} to {self.image.shape[1]//2} + {self.end_project_mid}') 
           self.image =  MidSlices(self.image, self.start_project_mid, self.end_project_mid, axis = 1)
           
      
        self.erosion_iterations = erosion_iterations
        
        self.heatmap = np.zeros(self.image.shape, dtype = 'float32')  
        self.savedir = savedir
        if len(n_tiles) > 2:
            n_tiles = (n_tiles[-2], n_tiles[-1])
        self.n_tiles = n_tiles
        self.fidelity = fidelity
        self.overlap_percent = overlap_percent
        self.iou_threshold = iou_threshold
        self.event_threshold = event_threshold
        self.event_confidence = event_confidence
        self.downsamplefactor = downsamplefactor
        self.originalimage = self.image
        if self.maskmodel is not None and self.maskimage is None:

            print(f'Generating mask, hang on')
            self.segdir = self.savedir + '/' + 'Segmentation'
            Path(self.segdir).mkdir(exist_ok=True)
            self.maskimage = Generate_only_mask(self.image, self.maskmodel, self.n_tiles)
            self.maskimage = binary_erosion(self.maskimage, iterations = self.erosion_iterations)
            imwrite(self.segdir + '/' + self.Name + '_mask' + '.tif', self.maskimage.astype('float32'))
            print(f'Mask generated and saved at {self.segdir}')
        

        self.model = load_model(self.model_dir + self.model_name + '.h5',
                                custom_objects={'loss': self.yololoss, 'Concat': Concat})

        self.markers = markers
        self.marker_tree = marker_tree
        self.remove_markers = remove_markers
        if self.remove_markers:
             self.image = DownsampleData(self.image, self.downsamplefactor)
        
        if self.remove_markers == True:
           self.first_pass_predict()
           self.second_pass_predict()
        if self.remove_markers == False:  
           self.second_pass_predict()
        if self.remove_markers == None:
           self.default_pass_predict() 

    def default_pass_predict(self):
        eventboxes = []
        classedboxes = {}    
        count = 0
        heatsavename = self.savedir+ "/"  + (os.path.splitext(os.path.basename(self.imagename))[0])+ '_Heat' 

        print('Detecting event locations')
        self.image = DownsampleData(self.image, self.downsamplefactor)
        for inputtime in tqdm(range(0, self.image.shape[0])):
                    if inputtime < self.image.shape[0] - self.imaget:
                                count = count + 1
                                if inputtime%(self.image.shape[0]//4)==0 and inputtime > 0 or inputtime >= self.image.shape[0] - self.imaget - 1:
                                      
                                                                              
                                      
                                      imwrite((heatsavename + '.tif' ), self.heatmap)
                                      
                                smallimage = CreateVolume(self.image, self.imaget, inputtime)
                                if self.normalize:
                                    smallimage = normalizeFloatZeroOne(smallimage,1,99.8)
                                # Cut off the region for training movie creation
                                #Break image into tiles if neccessary
                                predictions, allx, ally = self.predict_main(smallimage)
                                #Iterate over tiles
                                for p in range(0,len(predictions)):   
                        
                                  sum_time_prediction = predictions[p]
                                  if sum_time_prediction is not None:
                                     #For each tile the prediction vector has shape N H W Categories + Training Vector labels
                                     for i in range(0, sum_time_prediction.shape[0]):
                                          time_prediction =  sum_time_prediction[i]
                                          boxprediction = yoloprediction(ally[p], allx[p], time_prediction, self.stride, inputtime , self.config, self.key_categories, self.key_cord, self.nboxes, 'detection', 'dynamic',marker_tree=self.marker_tree)
                                          
                                          if boxprediction is not None:
                                                  eventboxes = eventboxes + boxprediction
                                            
                                for (event_name,event_label) in self.key_categories.items(): 
                                                     
                                                if event_label > 0:
                                                     current_event_box = []
                                                     for box in eventboxes:
                                                       
                                                        event_prob = box[event_name]
                                                        event_confidence = box['confidence']
                                                        if event_prob >= self.event_threshold and event_confidence >= self.event_confidence:
                                                            
                                                            current_event_box.append(box)
                                                     classedboxes[event_name] = [current_event_box]
                                                 
                                self.classedboxes = classedboxes    
                                self.eventboxes =  eventboxes
                                #nms over time
                                if inputtime > 0 and inputtime%self.imaget == 0:
 
                                    self.nms()
                                    self.to_csv()
                                    eventboxes = []
                                    classedboxes = {}    
                                    count = 0


    def first_pass_predict(self):
        
        print('Detecting background event locations')
        eventboxes = []
        classedboxes = {}
        remove_candidates = {}
        
        
        for inputtime in tqdm(range(0, self.image.shape[0])):
            if inputtime < self.image.shape[0] - self.imaget:
                
                remove_candidates_list = []
                smallimage = CreateVolume(self.image, self.imaget, inputtime)
                if self.normalize: 
                   smallimage = normalizeFloatZeroOne(smallimage, 1, 99.8)
                # Cut off the region for training movie creation
                # Break image into tiles if neccessary
                predictions, allx, ally = self.predict_main(smallimage)
                # Iterate over tiles
                for p in range(0, len(predictions)):

                    sum_time_prediction = predictions[p]

                    if sum_time_prediction is not None:
                        # For each tile the prediction vector has shape N H W Categories + Training Vector labels
                        for i in range(0, sum_time_prediction.shape[0]):
                            time_prediction = sum_time_prediction[i]
                            boxprediction = yoloprediction(ally[p], allx[p], time_prediction, self.stride,
                                                           inputtime, self.config,
                                                           self.key_categories, self.key_cord, self.nboxes, 'detection',
                                                           'dynamic', self.marker_tree)
                                          
                            if boxprediction is not None:
                                eventboxes = eventboxes + boxprediction
            for (event_name, event_label) in self.key_categories.items():
                    
                    if event_label == 0:  
                                current_event_box = []              
                                for box in eventboxes:
                
                                    event_prob = box[event_name]
                                    event_confidence = box['confidence']
                                    if event_prob >= self.event_threshold and event_confidence >= self.event_confidence :

                                        current_event_box.append(box)
                                        classedboxes[event_name] = [current_event_box]

            self.classedboxes = classedboxes
            if len(self.classedboxes) > 0:
                self.fast_nms()
                for (event_name, event_label) in self.key_categories.items():
                    
                    if event_label == 0:
                            iou_current_event_boxes = self.iou_classedboxes[event_name][0]
                            iou_current_event_boxes = sorted(iou_current_event_boxes, key=lambda x: x[event_name], reverse=True)
                            for box in iou_current_event_boxes:
                                     closest_location = get_nearest(self.marker_tree, box['ycenter'], box['xcenter'], box['real_time_event'])
                                     if closest_location is not None:
                                        ycentermean, xcentermean = closest_location
                                        try:
                                            remove_candidates_list = remove_candidates[str(int(box['real_time_event']))]
                                            if (ycentermean * self.downsamplefactor, xcentermean * self.downsamplefactor) not in remove_candidates_list:
                                                    remove_candidates_list.append((ycentermean * self.downsamplefactor, xcentermean * self.downsamplefactor))
                                                    remove_candidates[str(int(box['real_time_event']))] = remove_candidates_list
                                        except:
                                            remove_candidates_list.append((ycentermean * self.downsamplefactor, xcentermean * self.downsamplefactor))
                                            remove_candidates[str(int(box['real_time_event']))]  = remove_candidates_list

                eventboxes = []
                classedboxes = {}                    
            #Image back to the same co ordinate system
        self.markers = DownsampleData(self.markers, int(1.0//self.downsamplefactor))
        self.image = DownsampleData(self.image, int(1.0//self.downsamplefactor))
        for i in range(0, self.markers.shape[0]):
                    self.markers[i,:] = self.markers[i,:] > 0
                    self.markers[i,:] = label(self.markers[i,:].astype('uint16'))

        new_markers = np.zeros_like(self.markers)
        for i in tqdm(range(0, new_markers.shape[0])):

                Clean_Coordinates = []
                try:
                   tree, location = self.marker_tree[str(int(i))]
                except:
                    location = []   
                try:
                   remove_location = remove_candidates[str(int(i))]
                except:
                    remove_location = []   
                if len(location) > 0:
                        for value in location:
                            if value not in remove_location:
                                Clean_Coordinates.append(value)

                                
                        Clean_Coordinates = sorted(Clean_Coordinates, key=lambda k: [k[1], k[0]])
                        Clean_Coordinates.append((0, 0))
                        Clean_Coordinates = np.asarray(Clean_Coordinates)
                            
                        coordinates_int = np.round(Clean_Coordinates).astype(int)
                        markers_raw = np.zeros_like(new_markers[i,:])
                        markers_raw[tuple(coordinates_int.T)] = 1 + np.arange(len(Clean_Coordinates))
                            
                        markers_current = dilation(markers_raw, disk(2))
                            
                        new_markers[i, :] = label(markers_current.astype('uint16'))                            
         
        markerdir = self.savedir + '/' + 'Clean_Markers'  
        Path(markerdir).mkdir(exist_ok=True)
        Name = os.path.basename(os.path.splitext(self.imagename)[0])
        print('Writing the clean markers')
        self.marker_tree = MakeTrees(new_markers)
        imwrite(markerdir + '/' + Name + '.tif', new_markers.astype('float32'))    
        



    def second_pass_predict(self):

        print('Detecting event locations')
        eventboxes = []
        classedboxes = {}
        self.n_tiles = (1,1)
        self.iou_threshold = 0.9
        heatsavename = self.savedir+ "/"  + (os.path.splitext(os.path.basename(self.imagename))[0])+ '_Heat'
 
        if self.normalize: 
           self.image = normalizeFloatZeroOne(self.image.astype('float32'), 1, 99.8)
  
        for inputtime in tqdm(range(0, self.image.shape[0])):
             if inputtime < self.image.shape[0] - self.imaget:   

                if inputtime%(self.image.shape[0]//4)==0 and inputtime > 0 or inputtime >= self.image.shape[0] - self.imaget - 1:



                                      imwrite((heatsavename + '.tif' ), self.heatmap) 
                tree, location = self.marker_tree[str(int(inputtime))]
                for i in range(len(location)):
                    
                    crop_xminus = location[i][1]  - int(self.imagex/2) * self.downsamplefactor - 1
                    crop_xplus = location[i][1]  + int(self.imagex/2) * self.downsamplefactor + 1
                    crop_yminus = location[i][0]  - int(self.imagey/2) * self.downsamplefactor - 1
                    crop_yplus = location[i][0]   + int(self.imagey/2) * self.downsamplefactor + 1
                    region =(slice(inputtime,inputtime + int(self.imaget)),slice(int(crop_yminus), int(crop_yplus)),
                          slice(int(crop_xminus), int(crop_xplus)))
                    
                    crop_image = self.image[region] 
                    if crop_image.shape[0] >= self.imaget and  crop_image.shape[1] >= self.imagey * self.downsamplefactor and crop_image.shape[2] >= self.imagex * self.downsamplefactor:                                                
                                #Now apply the prediction for counting real events
                                
                                crop_image = DownsampleData(crop_image, self.downsamplefactor)
                                ycenter = location[i][0]
                                xcenter = location[i][1]
                                  
                                predictions, allx, ally = self.predict_main(crop_image)
                                #Iterate over tiles
                                for p in range(0,len(predictions)):

                                  sum_time_prediction = predictions[p]
                                  if sum_time_prediction is not None:
                                     #For each tile the prediction vector has shape N H W Categories + Training Vector labels
                                     for i in range(0, sum_time_prediction.shape[0]):
                                          time_prediction =  sum_time_prediction[i]
                                          boxprediction = yoloprediction(0, 0 , time_prediction, self.stride,
                                                           inputtime, self.config,
                                                           self.key_categories, self.key_cord, self.nboxes, 'detection',
                                                           'dynamic')


                                if len(boxprediction) > 0:
                                      for i in range(0, len(boxprediction)):  
                                        boxprediction[i]['xcenter'] = xcenter
                                        boxprediction[i]['ycenter'] = ycenter
                                        boxprediction[i]['xstart'] = xcenter - int(self.imagex/2) * self.downsamplefactor
                                        boxprediction[i]['ystart'] = ycenter - int(self.imagey/2) * self.downsamplefactor  

                                 
                                                 
                                if boxprediction is not None:
                                          eventboxes = eventboxes + boxprediction
                for (event_name,event_label) in self.key_categories.items(): 
                                           
                                        if event_label > 0:
                                             current_event_box = []
                                             for box in eventboxes:
                                        
                                                event_prob = box[event_name]
                                                event_confidence = box['confidence']
                                                if event_prob >= self.event_threshold and event_confidence >= self.event_confidence :
                                                    
                                                          
                                                    current_event_box.append(box)
                                             classedboxes[event_name] = [current_event_box]

                if inputtime > 0 :                         
                        self.classedboxes = classedboxes    
                        self.eventboxes =  eventboxes
                        self.nms()
                        self.to_csv()
                        eventboxes = []
                        classedboxes = {}   


    def fast_nms(self):


        best_iou_classedboxes = {}
        self.iou_classedboxes = {}
        for (event_name,event_label) in self.key_categories.items():
            if event_label == 0:
               #best_sorted_event_box = self.classedboxes[event_name][0]
               best_sorted_event_box = dynamic_nms(self.heatmap,self.maskimage, self.classedboxes, event_name,  self.downsamplefactor, self.iou_threshold, self.event_threshold, self.imagex, self.imagey, self.fidelity )

               best_iou_classedboxes[event_name] = [best_sorted_event_box]

        self.iou_classedboxes = best_iou_classedboxes
                
    def nms(self):

        best_iou_classedboxes = {}
        self.iou_classedboxes = {}
        for (event_name,event_label) in self.key_categories.items():
            if event_label > 0:
               #best_sorted_event_box = self.classedboxes[event_name][0]
               if self.remove_markers is not None:
                   best_sorted_event_box = gold_nms(self.heatmap, self.classedboxes, event_name, 1, self.iou_threshold, self.event_threshold, self.imagex, self.imagey, 1 )

               if self.remove_markers == None:
                   best_sorted_event_box = dynamic_nms(self.heatmap,self.maskimage,  self.classedboxes, event_name,  self.downsamplefactor, self.iou_threshold, self.event_threshold, self.imagex, self.imagey, self.fidelity )

               best_iou_classedboxes[event_name] = [best_sorted_event_box]

        self.iou_classedboxes = best_iou_classedboxes



    def to_csv(self):
         if self.remove_markers is not None:
            save_dynamic_csv(self.imagename, self.key_categories, self.iou_classedboxes, self.savedir, 1, self.ndim, z = self.z, maskimage = self.maskimage)        
         if self.markers is None:
            save_dynamic_csv(self.imagename, self.key_categories, self.iou_classedboxes, self.savedir, self.downsamplefactor, self.ndim, z = self.z, maskimage = self.maskimage)          
  

    def saveimage(self, xlocations, ylocations, tlocations, angles, radius, scores):

        # Blue color in BGR
        textcolor = (255, 0, 0)

        # Line thickness of 2 px
        thickness = 2
        for j in range(len(xlocations)):
            startlocation = (int(xlocations[j] - radius[j]), int(ylocations[j] - radius[j]))
            endlocation = (int(xlocations[j] + radius[j]), int(ylocations[j] + radius[j]))
            Z = int(tlocations[j])

            image = self.Colorimage[Z, :, :, 1]
            if scores[j] >= 1.0 - 1.0E-5:
                color = (0, 0, 255)
                image = self.Colorimage[Z, :, :, 2]
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.rectangle(img, startlocation, endlocation, textcolor, thickness)

            cv2.putText(img, str('%.4f' % (scores[j])), startlocation, cv2.FONT_HERSHEY_SIMPLEX, 1, textcolor,
                        thickness, cv2.LINE_AA)
            if scores[j] >= 1.0 - 1.0E-5:
                self.Colorimage[Z, :, :, 2] = img[:, :, 0]
            else:
                self.Colorimage[Z, :, :, 1] = img[:, :, 0]
            if self.yolo_v2:
                x1 = xlocations[j]
                y1 = ylocations[j]

    def showNapari(self, imagedir, savedir, yolo_v2=False):

        Raw_path = os.path.join(imagedir, '*tif')
        X = glob.glob(Raw_path)
        self.savedir = savedir
        Imageids = []
        self.viewer = napari.Viewer()
        napari.run()
        for imagename in X:
            Imageids.append(imagename)

        eventidbox = QComboBox()
        eventidbox.addItem(EventBoxname)
        for (event_name, event_label) in self.key_categories.items():
            eventidbox.addItem(event_name)

        imageidbox = QComboBox()
        imageidbox.addItem(Boxname)
        detectionsavebutton = QPushButton(' Save detection Movie')

        for i in range(0, len(Imageids)):
            imageidbox.addItem(str(Imageids[i]))

        figure = plt.figure(figsize=(4, 4))
        multiplot_widget = FigureCanvas(figure)
        ax = multiplot_widget.figure.subplots(1, 1)
        width = 400
        dock_widget = self.viewer.window.add_dock_widget(
            multiplot_widget, name="EventStats", area='right')
        multiplot_widget.figure.tight_layout()
        self.viewer.window._qt_window.resizeDocks([dock_widget], [width], Qt.Horizontal)
        eventidbox.currentIndexChanged.connect(lambda eventid=eventidbox: EventViewer(
            self.viewer,
            imread(imageidbox.currentText()),
            eventidbox.currentText(),
            self.key_categories,
            os.path.basename(os.path.splitext(imageidbox.currentText())[0]),
            savedir,
            multiplot_widget,
            ax,
            figure,
            yolo_v2,

        )
                                               )

        imageidbox.currentIndexChanged.connect(
            lambda trackid=imageidbox: EventViewer(
                self.viewer,
                imread(imageidbox.currentText()),
                eventidbox.currentText(),
                self.key_categories,
                os.path.basename(os.path.splitext(imageidbox.currentText())[0]),
                savedir,
                multiplot_widget,
                ax,
                figure,
                yolo_v2,

            )
        )

        self.viewer.window.add_dock_widget(eventidbox, name="Event", area='left')
        self.viewer.window.add_dock_widget(imageidbox, name="Image", area='left')

    def overlaptiles(self, sliceregion):

        if self.n_tiles == (1, 1):
            patch = []
            rowout = []
            column = []
            patchx = sliceregion.shape[2] // self.n_tiles[0]
            patchy = sliceregion.shape[1] // self.n_tiles[1]
            patchshape = (patchy, patchx)
            smallpatch, smallrowout, smallcolumn = chunk_list(sliceregion, patchshape, self.stride, [0, 0])
            patch.append(smallpatch)
            rowout.append(smallrowout)
            column.append(smallcolumn)

        else:
            patchx = sliceregion.shape[2] // self.n_tiles[0]
            patchy = sliceregion.shape[1] // self.n_tiles[1]

            if patchx > self.imagex and patchy > self.imagey:
                if self.overlap_percent > 1 or self.overlap_percent < 0:
                    self.overlap_percent = 0.8

                jumpx = int(self.overlap_percent * patchx)
                jumpy = int(self.overlap_percent * patchy)

                patchshape = (patchy, patchx)
                rowstart = 0;
                colstart = 0
                pairs = []
                # row is y, col is x

                while rowstart < sliceregion.shape[1]:
                    colstart = 0
                    while colstart < sliceregion.shape[2]:
                        # Start iterating over the tile with jumps = stride of the fully convolutional network.
                        pairs.append([rowstart, colstart])
                        colstart += jumpx
                    rowstart += jumpy

                    # Include the last patch
                rowstart = sliceregion.shape[1] - patchy
                colstart = 0
                while colstart < sliceregion.shape[2] - patchx:
                    pairs.append([rowstart, colstart])
                    colstart += jumpx
                rowstart = 0
                colstart = sliceregion.shape[2] - patchx
                while rowstart < sliceregion.shape[1] - patchy:
                    pairs.append([rowstart, colstart])
                    rowstart += jumpy

                if sliceregion.shape[1] >= self.imagey and sliceregion.shape[2] >= self.imagex:

                    patch = []
                    rowout = []
                    column = []
                    for pair in pairs:
                        smallpatch, smallrowout, smallcolumn = chunk_list(sliceregion, patchshape, self.stride, pair)
                        if smallpatch.shape[1] >= self.imagey and smallpatch.shape[2] >= self.imagex:
                            patch.append(smallpatch)
                            rowout.append(smallrowout)
                            column.append(smallcolumn)

            else:

                patch = []
                rowout = []
                column = []
                patchx = sliceregion.shape[2] // self.n_tiles[0]
                patchy = sliceregion.shape[1] // self.n_tiles[1]
                patchshape = (patchy, patchx)
                smallpatch, smallrowout, smallcolumn = chunk_list(sliceregion, patchshape, self.stride, [0, 0])
                patch.append(smallpatch)
                rowout.append(smallrowout)
                column.append(smallcolumn)
        self.patch = patch
        self.sy = rowout
        self.sx = column

    def predict_main(self, sliceregion):
        try:
            self.overlaptiles(sliceregion)
            predictions = []
            allx = []
            ally = []
            if len(self.patch) > 0:
                for i in range(0, len(self.patch)):
                    sum_time_prediction = self.make_patches(self.patch[i])
                    predictions.append(sum_time_prediction)
                    allx.append(self.sx[i])
                    ally.append(self.sy[i])


            else:

                sum_time_prediction = self.make_patches(self.patch)
                predictions.append(sum_time_prediction)
                allx.append(self.sx)
                ally.append(self.sy)

        except tf.errors.ResourceExhaustedError:

            print('Out of memory, increasing overlapping tiles for prediction')
            self.list_n_tiles = list(self.n_tiles)
            self.list_n_tiles[0] = self.n_tiles[0] + 1
            self.list_n_tiles[1] = self.n_tiles[1] + 1
            self.n_tiles = tuple(self.list_n_tiles)

            self.predict_main(sliceregion)

        return predictions, allx, ally

    def make_patches(self, sliceregion):

        predict_im = np.expand_dims(sliceregion, 0)

        prediction_vector = self.model.predict(np.expand_dims(predict_im, -1), verbose=0)

        return prediction_vector

    def second_make_patches(self, sliceregion):

        predict_im = np.expand_dims(sliceregion, 0)

        prediction_vector = self.model.predict(np.expand_dims(predict_im, -1), verbose=0)

        return prediction_vector

    def make_batch_patches(self, sliceregion):

        prediction_vector = self.model.predict(np.expand_dims(sliceregion, -1), verbose=0)
        return prediction_vector


def CreateVolume(patch, imaget, timepoint):
    starttime = timepoint
    endtime = timepoint + imaget
    smallimg = patch[starttime:endtime, :]

    return smallimg


def chunk_list(image, patchshape, stride, pair):
    rowstart = pair[0]
    colstart = pair[1]

    endrow = rowstart + patchshape[0]
    endcol = colstart + patchshape[1]

    if endrow > image.shape[1]:
        endrow = image.shape[1]
    if endcol > image.shape[2]:
        endcol = image.shape[2]

    region = (slice(0, image.shape[0]), slice(rowstart, endrow),
              slice(colstart, endcol))

    # The actual pixels in that region.
    patch = image[region]

    # Always normalize patch that goes into the netowrk for getting a prediction score

    return patch, rowstart, colstart


class EventViewer(object):

    def __init__(self, viewer, image, event_name, key_categories, imagename, savedir, canvas, ax, figure, yolo_v2):

        self.viewer = viewer
        self.image = image
        self.event_name = event_name
        self.imagename = imagename
        self.canvas = canvas
        self.key_categories = key_categories
        self.savedir = savedir
        self.ax = ax
        self.yolo_v2 = yolo_v2
        self.figure = figure
        self.plot()

    def plot(self):

        self.ax.cla()

        for (event_name, event_label) in self.key_categories.items():
            if event_label > 0 and self.event_name == event_name:
                csvname = self.savedir + "/" + event_name + "Location" + (
                        os.path.splitext(os.path.basename(self.imagename))[0] + '.csv')
                event_locations, size_locations, angle_locations, line_locations, timelist, eventlist = self.event_counter(
                    csvname)

                for layer in list(self.viewer.layers):
                    if event_name in layer.name or layer.name in event_name or event_name + 'angle' in layer.name or layer.name in event_name + 'angle':
                        self.viewer.layers.remove(layer)
                    if 'Image' in layer.name or layer.name in 'Image':
                        self.viewer.layers.remove(layer)
                self.viewer.add_image(self.image, name='Image')
                self.viewer.add_points(np.asarray(event_locations), size=size_locations, name=event_name,
                                       face_color=[0] * 4, edge_color="red", edge_width=1)
                if self.yolo_v2:
                    self.viewer.add_shapes(np.asarray(line_locations), name=event_name + 'angle', shape_type='line',
                                           face_color=[0] * 4, edge_color="red", edge_width=1)
                self.viewer.theme = 'light'
                self.ax.plot(timelist, eventlist, '-r')
                self.ax.set_title(event_name + "Events")
                self.ax.set_xlabel("Time")
                self.ax.set_ylabel("Counts")
                self.figure.canvas.draw()
                self.figure.canvas.flush_events()
                plt.savefig(self.savedir + event_name + '.png')

    def event_counter(self, csv_file):

        time, y, x, score, size, confidence, angle = np.loadtxt(csv_file, delimiter=',', skiprows=1, unpack=True)

        radius = 10
        eventcounter = 0
        eventlist = []
        timelist = []
        listtime = time.tolist()
        listy = y.tolist()
        listx = x.tolist()
        listsize = size.tolist()
        listangle = angle.tolist()

        event_locations = []
        size_locations = []
        angle_locations = []
        line_locations = []
        for i in range(len(listtime)):
            tcenter = int(listtime[i])
            ycenter = listy[i]
            xcenter = listx[i]
            size = listsize[i]
            angle = listangle[i]
            eventcounter = listtime.count(tcenter)
            timelist.append(tcenter)
            eventlist.append(eventcounter)

            event_locations.append([tcenter, ycenter, xcenter])
            size_locations.append(size)

            xstart = xcenter + radius * math.cos(angle)
            xend = xcenter - radius * math.cos(angle)

            ystart = ycenter + radius * math.sin(angle)
            yend = ycenter - radius * math.sin(angle)
            line_locations.append([[tcenter, ystart, xstart], [tcenter, yend, xend]])
            angle_locations.append(angle)

        return event_locations, size_locations, angle_locations, line_locations, timelist, eventlist

