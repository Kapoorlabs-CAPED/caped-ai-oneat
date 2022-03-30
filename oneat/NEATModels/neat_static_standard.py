#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 15:13:01 2020

@author: aimachine
"""

from oneat.NEATUtils import plotters
import numpy as np
from oneat.NEATUtils import helpers
from oneat.NEATUtils.helpers import save_json, load_json, yoloprediction, nonfcn_yoloprediction, normalizeFloatZeroOne, \
 goodboxes, save_static_csv, DownsampleData
from keras import callbacks
import os
from tqdm import tqdm
from oneat.NEATModels import nets
from oneat.NEATModels.nets import Concat
from oneat.NEATModels.loss import static_yolo_loss, static_yolo_loss_segfree
from keras import backend as K
import tensorflow as tf
# from IPython.display import clear_output
from keras import optimizers
from pathlib import Path
from keras.models import load_model
from tifffile import imread, imwrite
import csv
import napari
#from napari.qt.threading import thread_worker
import matplotlib.pyplot as plt
#from matplotlib.backends.backend_qt5agg import \
    #FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
#from qtpy.QtCore import Qt
#from qtpy.QtWidgets import QComboBox, QPushButton, QSlider
import glob
import h5py
#import cv2
import imageio

Boxname = 'ImageIDBox'
CellTypeBoxname = 'CellIDBox'


class NEATStatic(object):
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

    def __init__(self, staticconfig, model_dir, model_name, catconfig=None, cordconfig=None):

        self.staticconfig = staticconfig
        self.catconfig = catconfig
        self.cordconfig = cordconfig
        self.model_dir = model_dir
        self.model_name = model_name

        if self.staticconfig != None:
            self.npz_directory = staticconfig.npz_directory
            self.npz_name = staticconfig.npz_name
            self.npz_val_name = staticconfig.npz_val_name
            self.key_categories = staticconfig.key_categories
            self.box_vector = staticconfig.box_vector
            self.show = staticconfig.show
            self.key_cord = staticconfig.key_cord
            self.categories = len(staticconfig.key_categories)
            self.stage_number = staticconfig.stage_number
            self.last_conv_factor = staticconfig.last_conv_factor
            self.depth = staticconfig.depth
            self.start_kernel = staticconfig.start_kernel
            self.mid_kernel = staticconfig.mid_kernel
            self.learning_rate = staticconfig.learning_rate
            self.epochs = staticconfig.epochs
            self.residual = staticconfig.residual
            self.startfilter = staticconfig.startfilter
            self.batch_size = staticconfig.batch_size
            self.multievent = staticconfig.multievent
            self.imagex = staticconfig.imagex
            self.imagey = staticconfig.imagey
            self.nboxes = staticconfig.nboxes
            self.gridx = staticconfig.gridx
            self.gridy = staticconfig.gridy
            self.yolo_v0 = staticconfig.yolo_v0
            self.stride = staticconfig.stride

        if self.staticconfig == None:

            try:
                self.staticconfig = load_json(self.model_dir + os.path.splitext(self.model_name)[0] + '_Parameter.json')
            except:
                self.staticconfig = load_json(self.model_dir + self.model_name + '_Parameter.json')
            self.npz_directory = self.staticconfig['npz_directory']
            self.npz_name = self.staticconfig['npz_name']
            self.npz_val_name = self.staticconfig['npz_val_name']
            self.key_categories = self.catconfig
            self.box_vector = self.staticconfig['box_vector']
            self.show = self.staticconfig['show']
            self.key_cord = self.cordconfig
            self.categories = len(self.catconfig)
            self.depth = self.staticconfig['depth']
            self.start_kernel = self.staticconfig['start_kernel']
            self.mid_kernel = self.staticconfig['mid_kernel']
            self.learning_rate = self.staticconfig['learning_rate']
            self.epochs = self.staticconfig['epochs']
            self.residual = self.staticconfig['residual']
            self.startfilter = self.staticconfig['startfilter']
            self.batch_size = self.staticconfig['batch_size']
            self.multievent = self.staticconfig['multievent']
            self.imagex = self.staticconfig['imagex']
            self.imagey = self.staticconfig['imagey']
            self.nboxes = self.staticconfig['nboxes']
            self.gridx = self.staticconfig['gridx']
            self.gridy = self.staticconfig['gridy']
            self.yolo_v0 = self.staticconfig['yolo_v0']
            self.stride = self.staticconfig['stride']
            self.stage_number = self.staticconfig['stage_number']
            self.last_conv_factor = self.staticconfig['last_conv_factor']

        self.X = None
        self.Y = None
        self.axes = None
        self.X_val = None
        self.Y_val = None
        self.Trainingmodel = None
        self.Xoriginal = None
        self.Xoriginal_val = None

        if self.residual:
            self.model_keras = nets.resnet_v2
        else:
            self.model_keras = nets.seqnet_v2

        if self.multievent == True:
            self.last_activation = 'sigmoid'
            self.entropy = 'binary'

        if self.multievent == False:
            self.last_activation = 'softmax'
            self.entropy = 'notbinary'

        self.yolo_loss = static_yolo_loss_segfree(self.categories, self.gridx, self.gridy, self.nboxes, self.box_vector,
                                                  self.entropy, self.yolo_v0)

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

        input_shape = (self.X.shape[1], self.X.shape[2], self.X.shape[3])

        Path(self.model_dir).mkdir(exist_ok=True)

        Y_main = self.Y[:, :, :, 0:self.categories - 1]

        y_integers = np.argmax(Y_main, axis=-1)
        y_integers = y_integers[:, 0, 0]

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

        print(self.Y.shape, self.nboxes)

        self.Trainingmodel = self.model_keras(input_shape, self.categories, box_vector=self.box_vector,
                                              nboxes=self.nboxes, stage_number=self.stage_number,
                                              last_conv_factor=self.last_conv_factor, depth=self.depth,
                                              start_kernel=self.start_kernel, mid_kernel=self.mid_kernel,
                                              startfilter=self.startfilter, last_activation=self.last_activation,
                                              input_weights=self.model_weights)

        sgd = optimizers.SGD(lr=self.learning_rate, momentum=0.99, decay=1e-6, nesterov=True)
        self.Trainingmodel.compile(optimizer=sgd, loss=self.yolo_loss, metrics=['accuracy'])
        self.Trainingmodel.summary()

        # Keras callbacks
        lrate = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=4, verbose=1)
        hrate = callbacks.History()
        srate = callbacks.ModelCheckpoint(self.model_dir + self.model_name, monitor='loss', verbose=1,
                                          save_best_only=False, save_weights_only=False, mode='auto', period=1)
        prate = plotters.PlotStaticHistory(self.Trainingmodel, self.X_val, self.Y_val, self.key_categories,
                                           self.key_cord, self.gridx, self.gridy, plot=self.show, nboxes=self.nboxes)

        # Train the model and save as a h5 file
        self.Trainingmodel.fit(self.X, self.Y, batch_size=self.batch_size, epochs=self.epochs,
                               validation_data=(self.X_val, self.Y_val), shuffle=True,
                               callbacks=[lrate, hrate, srate, prate])
        # clear_output(wait=True)

        # Removes the old model to be replaced with new model, if old one exists
        if os.path.exists(self.model_dir + self.model_name):
            os.remove(self.model_dir + self.model_name)

        self.Trainingmodel.save(self.model_dir + self.model_name)

    def predict(self, imagename, savedir, event_threshold, n_tiles=(1, 1), overlap_percent=0.8, iou_threshold=0.01,
                fcn=True, height=None, width=None, RGB=False, fidelity = 1, downsamplefactor = 1, normalize = True):

        self.imagename = imagename
        self.image = imread(imagename)
        self.ColorimageDynamic = np.zeros([self.image.shape[0], self.image.shape[1], self.image.shape[2], 3],
                                          dtype='uint16')
        self.ColorimageDynamic[:, :, :, 0] = self.image
        self.ColorimageStatic = np.zeros([self.image.shape[0], self.image.shape[1], self.image.shape[2], 3],
                                         dtype='uint16')
        self.savedir = savedir
        self.n_tiles = n_tiles
        self.fcn = fcn
        self.RGB = RGB
        self.height = height
        self.width = width
        self.fidelity = fidelity
        self.downsamplefactor = downsamplefactor
        self.overlap_percent = overlap_percent
        self.iou_threshold = iou_threshold
        self.event_threshold = event_threshold
        self.originalimage = self.image
        self.image = DownsampleData(self.image, self.downsamplefactor)
        self.normalize = normalize
        f = h5py.File(self.model_dir + self.model_name + '.h5', 'r+')
        data_p = f.attrs['training_config']
        data_p = data_p.decode().replace("learning_rate", "lr").encode()
        f.attrs['training_config'] = data_p
        f.close()
        self.model = load_model(self.model_dir + self.model_name + '.h5',
                                custom_objects={'loss': self.yolo_loss, 'Concat': Concat})

        eventboxes = []
        classedboxes = {}
        count = 0
        savenameDynamic = self.savedir + "/" + (
        os.path.splitext(os.path.basename(self.imagename))[0]) + '_ColoredDynamic'
        savenameStatic = self.savedir + "/" + (
            os.path.splitext(os.path.basename(self.imagename))[0]) + '_ColoredStatic'
        if RGB == False:
            for inputtime in tqdm(range(0, self.image.shape[0])):
                if inputtime < self.image.shape[0]:
                    if inputtime >= self.image.shape[0] - 1:
                        imwrite((savenameDynamic + '.tif'), self.ColorimageDynamic)
                        imwrite((savenameStatic + '.tif'), self.ColorimageStatic)

                    count = count + 1
                smallimage = self.image[inputtime, :]
                if self.normalize:
                    smallimage = normalizeFloatZeroOne(smallimage, 1, 99.8)
                # Break image into tiles if neccessary
                if fcn:

                    predictions, allx, ally = self.predict_main(smallimage)
                else:

                    self.make_non_fcn(smallimage)
                    predictions, allx, ally = self.predict_nonfcn(smallimage)
                # Iterate over tiles
                for p in range(0, len(predictions)):

                    sum_time_prediction = predictions[p]

                    if sum_time_prediction is not None:
                        # For each tile the prediction vector has shape N H W Categories + Trainng Vector labels
                        for i in range(0, sum_time_prediction.shape[0]):
                            time_prediction = sum_time_prediction[i]
                            boxprediction = yoloprediction(ally[p], allx[p], time_prediction, self.stride, inputtime,
                                                           self.staticconfig, self.key_categories, self.key_cord,
                                                           self.nboxes, 'detection', 'static')

                            if boxprediction is not None:
                                eventboxes = eventboxes + boxprediction

                for (event_name, event_label) in self.key_categories.items():

                    if event_label > 0:
                        current_event_box = []
                        for box in eventboxes:

                            event_prob = box[event_name]
                            if event_prob >= self.event_threshold[event_label]:
                                current_event_box.append(box)
                        classedboxes[event_name] = [current_event_box]

                self.classedboxes = classedboxes
                self.eventboxes = eventboxes

                self.nms()
                self.to_csv()
                eventboxes = []
                classedboxes = {}
                count = 0

        if RGB:

            smallimage = self.image[:, :, 0]
            if self.normalize:
                smallimage = normalizeFloatZeroOne(smallimage, 1, 99.8)
            # Break image into tiles if neccessary
            if fcn:

                predictions, allx, ally = self.predict_main(smallimage)

            else:

                self.make_non_fcn(smallimage)
                predictions, allx, ally = self.predict_nonfcn(smallimage)
            # Iterate over tiles
            for p in range(0, len(predictions)):

                sum_time_prediction = predictions[p]

                if sum_time_prediction is not None:
                    # For each tile the prediction vector has shape N H W Categories + Trainng Vector labels
                    for i in range(0, sum_time_prediction.shape[0]):
                        time_prediction = sum_time_prediction[i]

                        if self.fcn:
                            boxprediction = yoloprediction(ally[p], allx[p], time_prediction, self.stride, 0,
                                                           self.staticconfig, self.key_categories, self.key_cord,
                                                           self.nboxes, 'detection', 'static')
                        else:
                            boxprediction = nonfcn_yoloprediction(ally[p], allx[p], time_prediction, self.stride, 0,
                                                                  self.staticconfig, self.key_categories, self.key_cord,
                                                                  self.nboxes, 'detection', 'static')

                        if boxprediction is not None:
                            eventboxes = eventboxes + boxprediction

            for (event_name, event_label) in self.key_categories.items():

                if event_label > 0:
                    current_event_box = []
                    for box in eventboxes:

                        event_prob = box[event_name]
                        if event_prob >= self.event_threshold:
                            current_event_box.append(box)
                    classedboxes[event_name] = [current_event_box]

            self.classedboxes = classedboxes
            self.eventboxes = eventboxes
            # self.iou_classedboxes = classedboxes
            self.nms()
            self.to_csv()
            eventboxes = []
            classedboxes = {}
            count = 0

    def nms(self):

        best_iou_classedboxes = {}
        self.iou_classedboxes = {}
        for (event_name, event_label) in self.key_categories.items():
            if event_label > 0:
                # Get all events
                sorted_event_box = self.classedboxes[event_name][0]
                scores = [sorted_event_box[i][event_name] for i in range(len(sorted_event_box))]
                best_sorted_event_box = goodboxes(sorted_event_box, scores, self.iou_threshold, self.event_threshold[event_label],self.imagex,
                                                   self.imagey, fidelity = self.fidelity)
             

                best_iou_classedboxes[event_name] = [best_sorted_event_box]

        self.iou_classedboxes = best_iou_classedboxes

    def to_csv(self):

        
        save_static_csv(self.ColorimageStatic, self.ColorimageDynamic, self.imagename, self.key_categories, self.iou_classedboxes, self.savedir, self.downsamplefactor)
   


    def showNapari(self, imagedir, savedir):

        Raw_path = os.path.join(imagedir, '*tif')
        X = glob.glob(Raw_path)
        self.savedir = savedir
        Imageids = []
        self.viewer = napari.Viewer()
        napari.run()
        for imagename in X:
            Imageids.append(imagename)

        celltypeidbox = QComboBox()
        celltypeidbox.addItem(CellTypeBoxname)
        for (event_name, event_label) in self.key_categories.items():
            celltypeidbox.addItem(event_name)

        imageidbox = QComboBox()
        imageidbox.addItem(Boxname)
        detectionsavebutton = QPushButton(' Save detection Movie')

        for i in range(0, len(Imageids)):
            imageidbox.addItem(str(Imageids[i]))

        figure = plt.figure(figsize=(4, 4))
        multiplot_widget = FigureCanvas(figure)
        ax = multiplot_widget.figure.subplots(1, 1)
        width = 400

        multiplot_widget.figure.tight_layout()

        celltypeidbox.currentIndexChanged.connect(lambda eventid=celltypeidbox: CellTypeViewer(
            self.viewer,
            imread(imageidbox.currentText()),
            celltypeidbox.currentText(),
            self.key_categories,
            os.path.basename(os.path.splitext(imageidbox.currentText())[0]),
            savedir,
            multiplot_widget,
            ax,
            figure,

        )
                                                  )

        imageidbox.currentIndexChanged.connect(
            lambda trackid=imageidbox: CellTypeViewer(
                self.viewer,
                imread(imageidbox.currentText()),
                celltypeidbox.currentText(),
                self.key_categories,
                os.path.basename(os.path.splitext(imageidbox.currentText())[0]),
                savedir,
                multiplot_widget,
                ax,
                figure,

            )
        )

        self.viewer.window.add_dock_widget(celltypeidbox, name="CellType", area='left')
        self.viewer.window.add_dock_widget(imageidbox, name="Image", area='left')

    def overlaptiles(self, sliceregion):

        if self.n_tiles == (1, 1):

            patchshape = (sliceregion.shape[0], sliceregion.shape[1])

            patch = []
            rowout = []
            column = []

            patch.append(sliceregion)
            rowout.append(0)
            column.append(0)

        else:

            patchx = sliceregion.shape[1] // self.n_tiles[0]
            patchy = sliceregion.shape[0] // self.n_tiles[1]

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

                while rowstart < sliceregion.shape[0] - patchy:
                    colstart = 0
                    while colstart < sliceregion.shape[1] - patchx:
                        # Start iterating over the tile with jumps = stride of the fully convolutional network.
                        pairs.append([rowstart, colstart])
                        colstart += jumpx
                    rowstart += jumpy

                    # Include the last patch
                rowstart = sliceregion.shape[0] - patchy
                colstart = 0
                while colstart < sliceregion.shape[1]:
                    pairs.append([rowstart, colstart])
                    colstart += jumpx
                rowstart = 0
                colstart = sliceregion.shape[1] - patchx
                while rowstart < sliceregion.shape[0]:
                    pairs.append([rowstart, colstart])
                    rowstart += jumpy

                if sliceregion.shape[0] >= self.imagey and sliceregion.shape[1] >= self.imagex:

                    patch = []
                    rowout = []
                    column = []
                    for pair in pairs:
                        smallpatch, smallrowout, smallcolumn = chunk_list(sliceregion, patchshape, pair)
                        patch.append(smallpatch)
                        rowout.append(smallrowout)
                        column.append(smallcolumn)

            else:

                patch = []
                rowout = []
                column = []
                patch.append(sliceregion)
                rowout.append(0)
                column.append(0)

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

    def predict_nonfcn(self, sliceregion):

        predictions = []
        allx = []
        ally = []
        if len(self.patch) > 0:
            for i in range(0, len(self.patch)):
                sum_time_prediction = self.make_patches(self.patch[i])

                predictions.append(sum_time_prediction)
                allx.append(self.sx[i])
                ally.append(self.sy[i])

        return predictions, allx, ally

    def make_patches(self, sliceregion):

        predict_im = np.expand_dims(sliceregion, 0)

        prediction_vector = self.model.predict(np.expand_dims(predict_im, -1), verbose=0)

        return prediction_vector

    def make_non_fcn(self, sliceregion):

        jumpx = int(self.overlap_percent / 4 * self.imagex)
        jumpy = int(self.overlap_percent / 4 * self.imagey)

        patchshape = (self.imagey, self.imagex)
        rowstart = 0;
        colstart = 0
        pairs = []
        # row is y, col is x

        while rowstart < sliceregion.shape[0] - self.imagey:
            colstart = 0
            while colstart < sliceregion.shape[1] - self.imagex:
                # Start iterating over the tile with jumps = stride of the fully convolutional network.
                pairs.append([rowstart, colstart])
                colstart += jumpx
            rowstart += jumpy

        patch = []
        rowout = []
        column = []
        for pair in pairs:
            smallpatch, smallrowout, smallcolumn = chunk_list(sliceregion, patchshape, pair)

            patch.append(smallpatch)
            rowout.append(smallrowout)
            column.append(smallcolumn)

        self.patch = patch
        self.sy = rowout
        self.sx = column


def chunk_list(image, patchshape, pair):
    rowstart = pair[0]
    colstart = pair[1]

    endrow = rowstart + patchshape[0]
    endcol = colstart + patchshape[1]

    if endrow > image.shape[0]:
        endrow = image.shape[0]
    if endcol > image.shape[1]:
        endcol = image.shape[1]

    region = (slice(rowstart, endrow),
              slice(colstart, endcol))

    # The actual pixels in that region.
    patch = image[region]
    # Always normalize patch that goes into the netowrk for getting a prediction score

    return patch, rowstart, colstart


class CellTypeViewer(object):

    def __init__(self, viewer, image, celltype_name, key_categories, imagename, savedir, canvas, ax, figure):

        self.viewer = viewer
        self.image = image
        self.celltype_name = celltype_name
        self.imagename = imagename
        self.canvas = canvas
        self.key_categories = key_categories
        self.savedir = savedir

        self.plot()

    def plot(self):

        for (celltype_name, event_label) in self.key_categories.items():
            if event_label > 0 and self.celltype_name == celltype_name:
                csvname = self.savedir + "/" + celltype_name + "Location" + (
                            os.path.splitext(os.path.basename(self.imagename))[0] + '.csv')
                event_locations, size_locations, timelist, eventlist = self.event_counter(csvname)

                for layer in list(self.viewer.layers):
                    if celltype_name in layer.name or layer.name in celltype_name:
                        self.viewer.layers.remove(layer)
                    if 'Image' in layer.name or layer.name in 'Image':
                        self.viewer.layers.remove(layer)
                self.viewer.add_image(self.image, name='Image')
                self.viewer.add_points(np.asarray(event_locations), size=size_locations, name=celltype_name,
                                       face_color=[0] * 4, edge_color="red", edge_width=1)
                self.viewer.theme = 'light'

    def event_counter(self, csv_file):

        time, y, x, score, size, confidence = np.loadtxt(csv_file, delimiter=',', skiprows=1, unpack=True)

        eventcounter = 0
        eventlist = []
        timelist = []
        listtime = time.tolist()
        listy = y.tolist()
        listx = x.tolist()
        listsize = size.tolist()

        event_locations = []
        size_locations = []

        for i in range(len(listtime)):
            tcenter = listtime[i]
            ycenter = listy[i]
            xcenter = listx[i]
            size = listsize[i]
            eventcounter = listtime.count(tcenter)
            timelist.append(tcenter)
            eventlist.append(eventcounter)

            event_locations.append([tcenter, ycenter, xcenter])
            if size > 1:
                size_locations.append(size)
            else:
                size_locations.append(2)

        return event_locations, size_locations, timelist, eventlist
