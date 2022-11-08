#!/usr/bin/env python3
"""
Created on Sat May 23 15:13:01 2020

@author: aimachine
"""

import os

# from IPython.display import clear_output
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras import callbacks
from tensorflow.keras.models import load_model
from tifffile import imread
from tqdm import tqdm
import datetime
from oneat.NEATModels import nets
from oneat.NEATModels.loss import class_yolo_loss, static_yolo_loss
from oneat.NEATModels.nets import Concat
from oneat.NEATUtils import plotters, utils
from oneat.NEATUtils.utils import (
    goodboxes,
    load_json,
    normalizeFloatZeroOne,
    save_static_csv,
    save_volume,
    yoloprediction,
)

Boxname = "ImageIDBox"
CellTypeBoxname = "CellIDBox"


class NEATResNet:
    """
    Parameters
    ----------

    NpzDirectory : Specify the location of npz file containing the training data with movies and labels

    TrainModelName : Specify the name of the npz file containing training data and labels

    ValidationModelName :  Specify the name of the npz file containing validation data and labels

    categories : Number of action classes

    Categories_Name : List of class names and labels

    model_dir : Directory location where trained model weights are to be read or written from


    model_keras : The model as it appears as a Keras function

    model_weights : If re-training model_weights = model_dir  else None as default

    lstm_hidden_units : Number of hidden uniots for LSTm layer, 64 by default

    epochs :  Number of training epochs, 55 by default

    batch_size : batch_size to be used for training, 20 by default



    """

    def __init__(
        self,
        staticconfig,
        model_dir,
        catconfig,
        cordconfig,
        class_only=False,
        train_lstm=False,
    ):

        self.staticconfig = staticconfig
        self.catconfig = catconfig
        self.cordconfig = cordconfig
        self.model_dir = model_dir
        self.class_only = class_only
        self.train_lstm = train_lstm
        self.key_cord = self.cordconfig
        self.categories = len(self.catconfig)
        self.key_categories = self.catconfig
        if self.staticconfig is not None:
            self.npz_directory = staticconfig.npz_directory
            self.npz_name = staticconfig.npz_name
            self.npz_val_name = staticconfig.npz_val_name
            self.key_categories = staticconfig.key_categories
            self.box_vector = staticconfig.box_vector
            self.show = staticconfig.show
            self.key_cord = staticconfig.key_cord
            self.categories = len(staticconfig.key_categories)
            self.stage_number = staticconfig.stage_number
            self.last_conv_factor = 2 ** (self.stage_number - 1)
            self.depth = staticconfig.depth
            self.start_kernel = staticconfig.start_kernel
            self.mid_kernel = staticconfig.mid_kernel
            self.learning_rate = staticconfig.learning_rate
            self.epochs = staticconfig.epochs
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

        if self.staticconfig is None:

            self.staticconfig = load_json(
                os.path.join(self.model_dir, "parameters.json")
            )

            self.npz_directory = self.staticconfig["npz_directory"]
            self.npz_name = self.staticconfig["npz_name"]
            self.npz_val_name = self.staticconfig["npz_val_name"]
            self.box_vector = self.staticconfig["box_vector"]
            self.show = self.staticconfig["show"]
            self.depth = self.staticconfig["depth"]
            self.start_kernel = self.staticconfig["start_kernel"]
            self.mid_kernel = self.staticconfig["mid_kernel"]
            self.learning_rate = self.staticconfig["learning_rate"]
            self.epochs = self.staticconfig["epochs"]
            self.startfilter = self.staticconfig["startfilter"]
            self.batch_size = self.staticconfig["batch_size"]
            self.multievent = self.staticconfig["multievent"]
            self.imagex = self.staticconfig["imagex"]
            self.imagey = self.staticconfig["imagey"]
            self.nboxes = self.staticconfig["nboxes"]
            self.gridx = self.staticconfig["gridx"]
            self.gridy = self.staticconfig["gridy"]
            self.yolo_v0 = self.staticconfig["yolo_v0"]
            self.stride = self.staticconfig["stride"]

            self.stage_number = self.staticconfig["stage_number"]
            self.last_conv_factor = 2 ** (self.stage_number - 1)

        self.X = None
        self.Y = None
        self.axes = None
        self.X_val = None
        self.Y_val = None
        self.Trainingmodel = None
        self.Xoriginal = None
        self.Xoriginal_val = None
        if not self.class_only:

            if not self.train_lstm:
                self.model_keras = nets.resnet_v2

            if self.train_lstm:
                self.model_keras = nets.resnet_lstm_v2

            if self.multievent:
                self.last_activation = "sigmoid"
                self.entropy = "binary"

            if not self.multievent:
                self.last_activation = "softmax"
                self.entropy = "notbinary"

            self.yolo_loss = static_yolo_loss(
                self.categories,
                self.gridx,
                self.gridy,
                self.nboxes,
                self.box_vector,
                self.entropy,
                self.yolo_v0,
            )

        if self.class_only:

            if not self.train_lstm:
                self.model_keras = nets.resnet_v2_class
            if self.train_lstm:
                self.model_keras = nets.resnet_lstm_v2_class

            if self.multievent:
                self.last_activation = "sigmoid"
                self.entropy = "binary"

            if not self.multievent:
                self.last_activation = "softmax"
                self.entropy = "notbinary"

            self.yolo_loss = class_yolo_loss(self.categories, self.entropy)

    def loadData(self, sum_channels=False):

        (X, Y), axes = utils.load_full_training_data(
            self.npz_directory, self.npz_name, verbose=True
        )

        (X_val, Y_val), axes = utils.load_full_training_data(
            self.npz_directory, self.npz_val_name, verbose=True
        )

        self.Xoriginal = X
        self.Xoriginal_val = X_val

        self.X = X
        if sum_channels:
            self.X = np.sum(X, -1)
            self.X = np.expand_dims(self.X, -1)
        self.Y = Y[:, :, 0]
        self.X_val = X_val
        if sum_channels:
            self.X_val = np.sum(X_val, -1)
            self.X_val = np.expand_dims(self.X_val, -1)
        self.Y_val = Y_val[:, :, 0]
        self.axes = axes
        self.Y = self.Y.reshape((self.Y.shape[0], 1, 1, self.Y.shape[1]))
        self.Y_val = self.Y_val.reshape(
            (self.Y_val.shape[0], 1, 1, self.Y_val.shape[1])
        )
        print(self.X.shape, self.Y.shape)

    def TrainModel(self):

        input_shape = (self.X.shape[1], self.X.shape[2], self.X.shape[3])

        Path(self.model_dir).mkdir(exist_ok=True)

        Y_main = self.Y[:, :, :, 0 : self.categories - 1]

        y_integers = np.argmax(Y_main, axis=-1)
        y_integers = y_integers[:, 0, 0]

        dummyY = np.zeros(
            [
                self.Y.shape[0],
                self.Y.shape[1],
                self.Y.shape[2],
                self.categories + self.nboxes * self.box_vector,
            ]
        )
        dummyY[:, :, :, : self.Y.shape[3]] = self.Y

        dummyY_val = np.zeros(
            [
                self.Y_val.shape[0],
                self.Y_val.shape[1],
                self.Y_val.shape[2],
                self.categories + self.nboxes * self.box_vector,
            ]
        )
        dummyY_val[:, :, :, : self.Y_val.shape[3]] = self.Y_val
        for b in range(1, self.nboxes):
            dummyY[
                :,
                :,
                :,
                self.categories
                + b * self.box_vector : self.categories
                + (b + 1) * self.box_vector,
            ] = self.Y[
                :, :, :, self.categories : self.categories + self.box_vector
            ]
            dummyY_val[
                :,
                :,
                :,
                self.categories
                + b * self.box_vector : self.categories
                + (b + 1) * self.box_vector,
            ] = self.Y_val[
                :, :, :, self.categories : self.categories + self.box_vector
            ]

        self.Y = dummyY
        self.Y_val = dummyY_val

        print(self.Y.shape, self.nboxes)

        self.Trainingmodel = self.model_keras(
            input_shape,
            self.categories,
            box_vector=self.box_vector,
            yolo_loss = self.yolo_loss, 
            nboxes=self.nboxes,
            stage_number=self.stage_number,
            depth=self.depth,
            start_kernel=self.start_kernel,
            mid_kernel=self.mid_kernel,
            startfilter=self.startfilter,
            last_activation=self.last_activation,
            input_model=self.model_dir,
        )

        sgd = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.Trainingmodel.compile(
            optimizer=sgd, loss=self.yolo_loss, metrics=["accuracy"]
        )
        self.Trainingmodel.summary()
        
        # Keras callbacks
        lrate = callbacks.ReduceLROnPlateau(
            monitor="loss", factor=0.1, patience=4, verbose=1
        )
        hrate = callbacks.History()
        srate = callbacks.ModelCheckpoint(
            self.model_dir,
            monitor="loss",
            verbose=1,
            save_best_only=False,
            save_weights_only=False,
            mode="auto",
            period=1,
        )
        prate = plotters.PlotStaticHistory(
            self.Trainingmodel,
            self.X_val,
            self.Y_val,
            self.key_categories,
            self.key_cord,
            self.gridx,
            self.gridy,
            plot=self.show,
            nboxes=self.nboxes,
            class_only=self.class_only,
        )
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        # Train the model and save as a h5 file
        self.Trainingmodel.fit(
            self.X,
            self.Y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(self.X_val, self.Y_val),
            shuffle=True,
            callbacks=[lrate, hrate, srate, prate, tensorboard_callback],
        )

        self.Trainingmodel.save(self.model_dir)

    def predict(
        self,
        image: np.ndarray,
        savedir: str = None,
        event_threshold: float = 0.5,
        event_confidence: float = 0.5,
        n_tiles: tuple = (1, 1),
        overlap_percent: float = 0.8,
        iou_threshold: float = 0.01,
        RGB: bool = False,
        fidelity: int = 1,
        normalize: bool = True,
        activations=False,
    ):

        self.image = imread(image)

        self.savedir = savedir
        self.n_tiles = n_tiles
        self.RGB = RGB
        self.fidelity = fidelity
        self.overlap_percent = overlap_percent
        self.iou_threshold = iou_threshold
        self.event_threshold = event_threshold
        self.event_confidence = event_confidence
        self.originalimage = self.image
        self.activations = activations

        self.normalize = normalize

        self.model = self._build()

        eventboxes = []
        classedboxes = {}
        self.iou_classedboxes = {}
        self.all_iou_classedboxes = {}
        if self.normalize:
            self.image = normalizeFloatZeroOne(self.image, 1, 99.8)

        if not RGB:
            for inputtime in tqdm(range(0, self.image.shape[0])):

                smallimage = self.image[inputtime, :]

                # Break image into tiles if neccessary
                predictions, allx, ally = self.predict_main(smallimage)

                # Iterate over tiles
                for p in range(0, len(predictions)):

                    sum_time_prediction = predictions[p]

                    if sum_time_prediction is not None:
                        # For each tile the prediction vector has shape N H W Categories + Trainng Vector labels
                        for i in range(0, sum_time_prediction.shape[0]):
                            time_prediction = sum_time_prediction[i]
                            boxprediction = yoloprediction(
                                ally[p],
                                allx[p],
                                time_prediction,
                                self.stride,
                                inputtime,
                                self.staticconfig,
                                self.key_categories,
                                self.key_cord,
                                self.nboxes,
                                "detection",
                                "static",
                            )

                            if boxprediction is not None:
                                eventboxes = eventboxes + boxprediction

                for (event_name, event_label) in self.key_categories.items():

                    if event_label > 0:
                        current_event_box = []
                        for box in eventboxes:

                            event_prob = box[event_name]
                            event_confidence = box["confidence"]
                            if (
                                event_prob >= self.event_threshold[event_label]
                                and event_confidence >= self.event_confidence
                            ):
                                current_event_box.append(box)
                        classedboxes[event_name] = [current_event_box]

                self.classedboxes = classedboxes
                self.eventboxes = eventboxes

                self.nms()
                if self.savedir is not None:
                    self.to_csv()
                if self.activations:
                    self.to_activations()
                eventboxes = []
                classedboxes = {}

        if RGB:

            smallimage = self.image[:, :, 0]
            if self.normalize:
                smallimage = normalizeFloatZeroOne(smallimage, 1, 99.8)
            # Break image into tiles if neccessary
            predictions, allx, ally = self.predict_main(smallimage)

            # Iterate over tiles
            for p in range(0, len(predictions)):

                sum_time_prediction = predictions[p]

                if sum_time_prediction is not None:
                    # For each tile the prediction vector has shape N H W Categories + Trainng Vector labels
                    for i in range(0, sum_time_prediction.shape[0]):
                        time_prediction = sum_time_prediction[i]

                        boxprediction = yoloprediction(
                            ally[p],
                            allx[p],
                            time_prediction,
                            self.stride,
                            0,
                            self.staticconfig,
                            self.key_categories,
                            self.key_cord,
                            self.nboxes,
                            "detection",
                            "static",
                        )

                        if boxprediction is not None:
                            eventboxes = eventboxes + boxprediction

            for (event_name, event_label) in self.key_categories.items():

                if event_label > 0:
                    current_event_box = []
                    for box in eventboxes:

                        event_prob = box[event_name]
                        event_confidence = box["confidence"]
                        if (
                            event_prob >= self.event_threshold
                            and event_confidence >= self.event_confidence
                        ):
                            current_event_box.append(box)
                    classedboxes[event_name] = [current_event_box]

            self.classedboxes = classedboxes
            self.eventboxes = eventboxes
            # self.iou_classedboxes = classedboxes
            self.nms()
            if self.savedir is not None:
                self.to_csv()
            if self.activations:
                self.to_activations()
            eventboxes = []
            classedboxes = {}

    def _build(self):

        Model = load_model(
            self.model_dir,
            custom_objects={"loss": self.yolo_loss, "Concat": Concat},
        )
        return Model

    def nms(self):

        best_iou_classedboxes = {}
        self.iou_classedboxes = {}
        for (event_name, event_label) in self.key_categories.items():
            if event_label > 0:
                # Get all events
                sorted_event_box = self.classedboxes[event_name][0]
                scores = [
                    sorted_event_box[i][event_name]
                    for i in range(len(sorted_event_box))
                ]
                best_sorted_event_box = goodboxes(
                    sorted_event_box,
                    scores,
                    self.iou_threshold,
                    self.event_threshold[event_label],
                    self.imagex,
                    self.imagey,
                    fidelity=self.fidelity,
                )

                best_iou_classedboxes[event_name] = [best_sorted_event_box]

        self.iou_classedboxes = best_iou_classedboxes

    def to_csv(self):

        save_static_csv(
            self.key_categories, self.iou_classedboxes, self.savedir
        )

    def to_activations(self):

        self.all_iou_classedboxes = save_volume(
            self.key_categories,
            self.iou_classedboxes,
            self.all_iou_classedboxes,
        )

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
                rowstart = 0
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

                if (
                    sliceregion.shape[0] >= self.imagey
                    and sliceregion.shape[1] >= self.imagex
                ):

                    patch = []
                    rowout = []
                    column = []
                    for pair in pairs:
                        smallpatch, smallrowout, smallcolumn = chunk_list(
                            sliceregion, patchshape, pair
                        )
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

            print("Out of memory, increasing overlapping tiles for prediction")

            self.list_n_tiles = list(self.n_tiles)
            self.list_n_tiles[0] = self.n_tiles[0] + 1
            self.list_n_tiles[1] = self.n_tiles[1] + 1
            self.n_tiles = tuple(self.list_n_tiles)
            self.predict_main(sliceregion)

        return predictions, allx, ally

    def make_patches(self, sliceregion):

        predict_im = np.expand_dims(sliceregion, 0)

        prediction_vector = self.model.predict(
            np.expand_dims(predict_im, -1), verbose=0
        )

        return prediction_vector


def chunk_list(image, patchshape, pair):
    rowstart = pair[0]
    colstart = pair[1]

    endrow = rowstart + patchshape[0]
    endcol = colstart + patchshape[1]

    if endrow > image.shape[0]:
        endrow = image.shape[0]
    if endcol > image.shape[1]:
        endcol = image.shape[1]

    region = (slice(rowstart, endrow), slice(colstart, endcol))

    # The actual pixels in that region.
    patch = image[region]
    # Always normalize patch that goes into the netowrk for getting a prediction score

    return patch, rowstart, colstart
