import os
import sys
import tensorflow as tf
from tqdm import tqdm
from oneat.NEATModels import nets
from oneat.NEATModels.nets import Concat
from oneat.NEATModels.loss import diamond_yolo_loss
from oneat.pretrained import get_registered_models, get_model_details, get_model_instance
from pathlib import Path
from oneat.NEATUtils.utils import  pad_timelapse, get_nearest_volume,  load_json, diamondyoloprediction, normalizeFloatZeroOne, GenerateVolumeMarkers, MakeForest,save_diamond_csv, diamond_dynamic_nms
import numpy as np
from keras import models
from keras.models import load_model
from tensorflow.keras.utils import plot_model
from tifffile import imread
import napari

class VizNEATEynamic(object):

    def __init__(self,  config, imagename, model_dir, model_name,  catconfig=None, cordconfig=None, timepoints = 4, layer_viz_start = 10, layer_viz_end = 20, dtype = np.uint8, n_tiles = (1,1,1), normalize = True):

        self.config = config
        self.model_dir = model_dir
        self.model_name = model_name
        self.n_tiles = n_tiles
        self.imagename = imagename
        self.dtype = dtype
        self.timepoints = timepoints
        self.catconfig = catconfig
        self.cordconfig = cordconfig
        self.layer_viz_start = layer_viz_start 
        self.layer_viz_end = layer_viz_end
        self.image = imread(imagename).astype(self.dtype)
        self.normalize = normalize
       
        self.viewer = napari.Viewer()   
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
            self.learning_rate = config.learning_rate
            self.epochs = config.epochs
            self.startfilter = config.startfilter
            self.batch_size = config.batch_size
            self.multievent = config.multievent
            self.imagex = config.imagex
            self.imagey = config.imagey
            self.imagez = config.imagez
            self.imaget = config.size_tminus + config.size_tplus + 1
            self.size_tminus = config.size_tminus
            self.size_tplus = config.size_tplus

            self.nboxes = config.nboxes
            self.gridx = 1
            self.gridy = 1
            self.gridz = 1
            self.yolo_v0 = config.yolo_v0
            self.yolo_v1 = config.yolo_v1
            self.yolo_v2 = config.yolo_v2
            self.stride = config.stride
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
            self.learning_rate = self.config['learning_rate']
            self.epochs = self.config['epochs']
            self.startfilter = self.config['startfilter']
            self.batch_size = self.config['batch_size']
            self.multievent = self.config['multievent']
            self.imagex = self.config['imagex']
            self.imagey = self.config['imagey']
            self.imagez = self.config['imagez']
            self.imaget = self.config['size_tminus'] + self.config['size_tplus'] + 1
            self.size_tminus = self.config['size_tminus']
            self.size_tplus = self.config['size_tplus']
            self.nboxes = self.config['nboxes']
            self.stage_number = self.config['stage_number']
            self.last_conv_factor = 2 ** (self.stage_number - 1)
            self.gridx = 1
            self.gridy = 1
            self.gridz = 1
            self.yolo_v0 = self.config['yolo_v0']
            self.yolo_v1 = self.config['yolo_v1']
            self.yolo_v2 = self.config['yolo_v2']
            self.stride = self.config['stride']


    def VizNets(self):
        self.model_keras = nets.VollNet

        if self.multievent == True:
            self.last_activation = 'sigmoid'
            self.entropy = 'binary'

        if self.multievent == False:
            self.last_activation = 'softmax'
            self.entropy = 'notbinary'

        self.yololoss = diamond_yolo_loss(self.categories, self.gridx, self.gridy, self.gridz, self.nboxes,
                                          self.box_vector, self.entropy, self.yolo_v0, self.yolo_v1, self.yolo_v2)

        if self.normalize: 
            self.image = normalizeFloatZeroOne(self.image, 1, 99.8, dtype = self.dtype)
       
        self.model = load_model(os.path.join(self.model_dir, self.model_name) + '.h5',
                                custom_objects={'loss': self.yololoss, 'Concat': Concat})    


        layer_outputs = [layer.output for layer in self.model.layers[self.layer_viz_start:self.layer_viz_end]]
        activation_model = models.Model(inputs= self.model.input, outputs=layer_outputs)

        for inputtime in tqdm(range(0, self.image.shape[0])):
               if inputtime < self.image.shape[0] - self.imaget and inputtime > int(self.imaget)//2:
                               
                                      
                                smallimage = CreateVolume(self.image, self.size_tminus, self.size_tplus, inputtime)
                                self.viewer.add_image(np.sum(smallimage, axis = 0), name= 'Image', blending= 'additive' )
                                smallimage = np.expand_dims(smallimage,0)
                                smallimage = tf.reshape(smallimage, (smallimage.shape[0], smallimage.shape[2], smallimage.shape[3],smallimage.shape[4], smallimage.shape[1]))
                                activations = activation_model.predict(smallimage)
                                print(type(activations), len(activations))
                                for count, activation in enumerate(activations):
                                    max_activation = np.sum(activation, axis = -1)
                                    self.viewer.add_image(max_activation, name= 'Activation' + str(count), blending= 'additive', colormap='inferno' )
        
        napari.run()

def CreateVolume(patch, size_tminus, size_tplus, timepoint):
    starttime = timepoint - int(size_tminus)
    endtime = timepoint + int(size_tplus) + 1
    #TZYX needs to be reshaed to ZYXT
    smallimg = patch[starttime:endtime,]
    return smallimg