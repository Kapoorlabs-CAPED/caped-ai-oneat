from oneat.NEATUtils import plotters
import numpy as np
from oneat.NEATUtils import helpers
from oneat.NEATUtils.helpers import MidSlices, pad_timelapse,  MidSlicesSum, get_nearest,  load_json, yoloprediction, normalizeFloatZeroOne, GenerateMarkers, MakeTrees, DownsampleData,save_dynamic_csv, dynamic_nms, gold_nms
from keras import callbacks
import os
import keras
import sys
from scipy.ndimage.morphology import binary_dilation, binary_erosion
import math
from tqdm import tqdm
from oneat.NEATModels import nets
from oneat.NEATModels.nets import Concat
from oneat.NEATModels.loss import dynamic_yolo_loss
from keras import backend as K
import tensorflow as tf
from oneat.pretrained import get_registered_models, get_model_details, get_model_instance
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
            self.last_conv_factor = 2 ** (self.stage_number - 1)
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

            self.config = load_json(os.path.join(self.model_dir, self.model_name) + '_Parameter.json')
            

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
            self.last_conv_factor = 2 ** (self.stage_number - 1)
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

    @classmethod   
    def local_from_pretrained(cls, name_or_alias=None):
           try:
               print(cls)
               get_model_details(cls, name_or_alias, verbose=True)
               return get_model_instance(cls, name_or_alias)
           except ValueError:
               if name_or_alias is not None:
                   print("Could not find model with name or alias '%s'" % (name_or_alias), file=sys.stderr)
                   sys.stderr.flush()
               get_registered_models(cls, verbose=True)  

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
 

        model_weights = os.path.join(self.model_dir, self.model_name) 
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
                                              stage_number=self.stage_number,
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
        if os.path.exists(os.path.join(self.model_dir, self.model_name) ):
            os.remove(os.path.join(self.model_dir, self.model_name) )

        self.Trainingmodel.save(os.path.join(self.model_dir, self.model_name) )

    def get_markers(self, imagename, segdir, start_project_mid = 4, end_project_mid = 4,
     downsamplefactor = 1):

        self.imagename = imagename
        self.segdir = segdir
        Name = os.path.basename(os.path.splitext(self.imagename)[0])
        self.pad_width = (self.config['imagey'], self.config['imagex'])
        self.downsamplefactor = downsamplefactor
        print('Obtaining Markers')
        self.segimage = imread(self.segdir + '/' + Name + '.tif')
        self.markers = GenerateMarkers(self.segimage, 
        start_project_mid = start_project_mid, end_project_mid = end_project_mid, pad_width = self.pad_width)
        self.marker_tree = MakeTrees(self.markers)
        self.segimage = None         

        return self.marker_tree
    
    def predict(self, imagename,  savedir, n_tiles=(1, 1), overlap_percent=0.8,
                event_threshold=0.5, event_confidence = 0.5, iou_threshold=0.1,  fidelity=1, downsamplefactor = 1, start_project_mid = 4, end_project_mid = 4,
                erosion_iterations = 1,  marker_tree = None, remove_markers = False, normalize = True, center_oneat = True, nms_function = 'iou'):


        
        self.imagename = imagename
        self.Name = os.path.basename(os.path.splitext(self.imagename)[0])
        self.nms_function = nms_function 
        self.image = imread(imagename)
        self.start_project_mid = start_project_mid
        self.end_project_mid = end_project_mid
        self.ndim = len(self.image.shape)
        self.normalize = normalize
        self.z = 0
        if self.ndim == 4:
           self.z = self.image.shape[1]//2
           print(f'Image {self.image.shape} is {self.ndim} dimensional, projecting around the center {self.image.shape[1]//2} - {self.start_project_mid} to {self.image.shape[1]//2} + {self.end_project_mid}') 
           self.image =  MidSlicesSum(self.image, self.start_project_mid, self.end_project_mid, axis = 1)
           self.z = self.z - (self.start_project_mid + self.end_project_mid)//2
        if self.normalize: 
                    self.image = normalizeFloatZeroOne(self.image.astype('float32'), 1, 99.8)
        self.erosion_iterations = erosion_iterations
        
        self.heatmap = np.zeros(self.image.shape, dtype = 'float32')  
        self.eventmarkers = np.zeros(self.image.shape, dtype = 'uint16')
        self.savedir = savedir
        Path(self.savedir).mkdir(exist_ok=True)
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
        self.center_oneat = center_oneat
        
        self.model = load_model(os.path.join(self.model_dir, self.model_name) + '.h5',
                                custom_objects={'loss': self.yololoss, 'Concat': Concat})

        self.marker_tree = marker_tree
        self.remove_markers = remove_markers
        if self.remove_markers:
             self.image = DownsampleData(self.image, self.downsamplefactor)

        
        if self.remove_markers == True:
           self.generate_maps = False 

           self.image = pad_timelapse(self.image, self.pad_width)
           print(f'zero padded image shape ${self.image.shape}')
           self.first_pass_predict()
           self.second_pass_predict()
        if self.remove_markers == False:
           self.generate_maps = False 

           self.image = pad_timelapse(self.image, self.pad_width)
           print(f'zero padded image shape ${self.image.shape}')
           self.second_pass_predict()
        if self.remove_markers == None:
           self.generate_maps = True 
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
                                          boxprediction = yoloprediction(ally[p], 
                                          allx[p], 
                                          time_prediction, 
                                          self.stride, inputtime , 
                                          self.config, 
                                          self.key_categories, 
                                          self.key_cord, 
                                          self.nboxes, 'detection', 'dynamic',marker_tree=self.marker_tree, center_oneat = self.center_oneat)
                                          
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
                                
                                
                                if inputtime > 0 and inputtime%(self.imaget) == 0:
                                    
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
                                                           'dynamic', marker_tree = self.marker_tree, center_oneat = False)
                                          
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
        self.image = DownsampleData(self.image, int(1.0//self.downsamplefactor))
       
        



    def second_pass_predict(self):

        print('Detecting event locations')
        eventboxes = []
        classedboxes = {}
        self.n_tiles = (1,1)
        
        heatsavename = self.savedir+ "/"  + (os.path.splitext(os.path.basename(self.imagename))[0])+ '_Heat'
        eventsavename = self.savedir + "/" + (os.path.splitext(os.path.basename(self.imagename))[0])+ '_Event'
        
  
        for inputtime in tqdm(range(int(self.imaget)//2, self.image.shape[0])):
             if inputtime < self.image.shape[0] - self.imaget:   

                if inputtime%(self.image.shape[0]//4)==0 and inputtime > 0 or inputtime >= self.image.shape[0] - self.imaget - 1:
                                      markers_current = dilation(self.eventmarkers[inputtime,:], disk(2))
                                      self.eventmarkers[inputtime,:] = label(markers_current.astype('uint16')) 
                                      imwrite((heatsavename + '.tif' ), self.heatmap) 
                                      imwrite((eventsavename + '.tif' ), self.eventmarkers.astype('uint16'))
                tree, location = self.marker_tree[str(int(inputtime))]
                for i in range(len(location)):
                    
                    crop_xminus = location[i][1]  - int(self.imagex/2) * self.downsamplefactor 
                    crop_xplus = location[i][1]  + int(self.imagex/2) * self.downsamplefactor 
                    crop_yminus = location[i][0]  - int(self.imagey/2) * self.downsamplefactor 
                    crop_yplus = location[i][0]   + int(self.imagey/2) * self.downsamplefactor 
                    region =(slice(inputtime - int(self.imaget)//2,inputtime + int(self.imaget)//2 + 1),slice(int(crop_yminus), int(crop_yplus)),
                          slice(int(crop_xminus), int(crop_xplus)))
                    
                    crop_image = self.image[region] 
                    
                    if crop_image.shape[0] >= self.imaget and  crop_image.shape[1] >= self.imagey * self.downsamplefactor and crop_image.shape[2] >= self.imagex * self.downsamplefactor:                                                
                                #Now apply the prediction for counting real events
                               
                                crop_image = DownsampleData(crop_image, self.downsamplefactor)
                                ycenter = location[i][0]
                                xcenter = location[i][1]
                                  
                                predictions, allx, ally = self.predict_main(crop_image)
                                sum_time_prediction = predictions[0]
                                if sum_time_prediction is not None:
                                     #For each tile the prediction vector has shape N H W Categories + Training Vector labels
                                     time_prediction =  sum_time_prediction[0]
                                     boxprediction = yoloprediction(0, 0 , time_prediction, self.stride,
                                                           inputtime, self.config,
                                                           self.key_categories, self.key_cord, self.nboxes, 'detection',
                                                           'dynamic', center_oneat = self.center_oneat)
                                     if boxprediction is not None and len(boxprediction) > 0 and xcenter - self.pad_width[1] > 0 and ycenter - self.pad_width[0] > 0 and xcenter - self.pad_width[1] < self.originalimage.shape[2] and ycenter - self.pad_width[0] < self.originalimage.shape[1] :
                                            
                                                
                                                boxprediction[0]['real_time_event'] = inputtime
                                                boxprediction[0]['xcenter'] = xcenter - self.pad_width[1]
                                                boxprediction[0]['ycenter'] = ycenter - self.pad_width[0]
                                                boxprediction[0]['xstart'] = xcenter   - int(self.imagex/2) * self.downsamplefactor
                                                boxprediction[0]['ystart'] = ycenter   - int(self.imagey/2) * self.downsamplefactor  
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


                self.classedboxes = classedboxes    
                self.eventboxes =  eventboxes
                self.iou_classedboxes = classedboxes
                self.to_csv()
                eventboxes = []
                classedboxes = {}   


    def fast_nms(self):


        best_iou_classedboxes = {}
        self.iou_classedboxes = {}
        for (event_name,event_label) in self.key_categories.items():
            if event_label == 0:
               #best_sorted_event_box = self.classedboxes[event_name][0]
               best_sorted_event_box = dynamic_nms(self.heatmap, self.classedboxes, event_name,  self.downsamplefactor, self.iou_threshold, self.event_threshold, self.imagex, self.imagey, self.fidelity, generate_map = self.generate_maps, nms_function = self.nms_function )

               best_iou_classedboxes[event_name] = [best_sorted_event_box]

        self.iou_classedboxes = best_iou_classedboxes
               


    def nms(self):

        best_iou_classedboxes = {}
        self.iou_classedboxes = {}
        for (event_name,event_label) in self.key_categories.items():
            if event_label > 0:
               #best_sorted_event_box = self.classedboxes[event_name][0]
               if self.remove_markers is not None:
                   best_sorted_event_box = gold_nms(self.heatmap,self.eventmarkers, self.classedboxes, event_name, 1, self.iou_threshold, self.event_threshold, self.imagex, self.imagey,generate_map = self.generate_maps, nms_function = self.nms_function )

               if self.remove_markers is None:
                   best_sorted_event_box = dynamic_nms(self.heatmap,self.classedboxes, event_name,  self.downsamplefactor, self.iou_threshold, self.event_threshold, self.imagex, self.imagey, self.fidelity,generate_map = self.generate_maps, nms_function = self.nms_function )

               best_iou_classedboxes[event_name] = [best_sorted_event_box]

        self.iou_classedboxes = best_iou_classedboxes



    def to_csv(self):
         if self.remove_markers is not None:
            save_dynamic_csv(self.imagename, self.key_categories, self.iou_classedboxes, self.savedir, 1, self.ndim, z = self.z)        
         if self.remove_markers is None:
            save_dynamic_csv(self.imagename, self.key_categories, self.iou_classedboxes, self.savedir, self.downsamplefactor, self.ndim, z = self.z)          
  

    

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

