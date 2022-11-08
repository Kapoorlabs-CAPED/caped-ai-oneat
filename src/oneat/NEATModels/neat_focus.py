import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import callbacks, optimizers
from tensorflow.keras.models import load_model
from scipy.optimize import curve_fit
from tqdm import tqdm
import datetime
from oneat.NEATModels import nets
from oneat.NEATModels.loss import dynamic_yolo_loss
from oneat.NEATModels.nets import Concat
from oneat.NEATUtils import plotters, utils
from oneat.NEATUtils.utils import (
    focyoloprediction,
    load_json,
    normalizeFloatZeroOne,
    simpleaveragenms,
)

Boxname = "ImageIDBox"
EventBoxname = "EventIDBox"


class NEATFocus:

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


    epochs :  Number of training epochs, 55 by default

    batch_size : batch_size to be used for training, 20 by default



    """

    def __init__(self, config, model_dir, catconfig=None, cordconfig=None):

        self.config = config
        self.catconfig = catconfig
        self.cordconfig = cordconfig
        self.model_dir = model_dir
        if self.config is not None:
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
            self.learning_rate = config.learning_rate
            self.epochs = config.epochs
            self.startfilter = config.startfilter
            self.batch_size = config.batch_size
            self.multievent = config.multievent
            self.imagex = config.imagex
            self.imagey = config.imagey
            self.imagez = config.size_tminus + config.size_tplus + 1
            self.size_zminus = config.size_tminus
            self.size_zplus = config.size_tplus
            self.nboxes = 1
            self.gridx = 1
            self.gridy = 1
            self.gridz = 1
            self.yolo_v0 = True
            self.yolo_v1 = False
            self.yolo_v2 = False
            self.stride = config.last_conv_factor
        if self.config is None:

            self.config = load_json(
                os.path.join(self.model_dir, "parameters.json")
            )
            self.npz_directory = self.config["npz_directory"]
            self.npz_name = self.config["npz_name"]
            self.npz_val_name = self.config["npz_val_name"]
            self.key_categories = self.catconfig
            self.box_vector = self.config["box_vector"]
            self.show = self.config["show"]
            self.key_cord = self.cordconfig
            self.categories = len(self.catconfig)
            self.depth = self.config["depth"]
            self.start_kernel = self.config["start_kernel"]
            self.mid_kernel = self.config["mid_kernel"]
            self.learning_rate = self.config["learning_rate"]
            self.epochs = self.config["epochs"]
            self.startfilter = self.config["startfilter"]
            self.batch_size = self.config["batch_size"]
            self.multievent = self.config["multievent"]
            self.imagex = self.config["imagex"]
            self.imagey = self.config["imagey"]
            self.imagez = (
                self.config["size_tminus"] + self.config["size_tplus"] + 1
            )
            self.size_zminus = self.config["size_tminus"]
            self.size_zplus = self.config["size_tplus"]
            self.nboxes = 1
            self.stage_number = self.config["stage_number"]
            self.last_conv_factor = self.config["last_conv_factor"]
            self.gridx = 1
            self.gridy = 1
            self.gridz = 1
            self.yolo_v0 = False
            self.yolo_v1 = True
            self.yolo_v2 = False
            self.stride = self.config["last_conv_factor"]

        self.X = None
        self.Y = None
        self.axes = None
        self.X_val = None
        self.Y_val = None
        self.Trainingmodel = None
        self.Xoriginal = None
        self.Xoriginal_val = None

        self.model_keras = nets.VollNet
        if self.multievent:
            self.last_activation = "sigmoid"
            self.entropy = "binary"

        if not self.multievent:
            self.last_activation = "softmax"
            self.entropy = "notbinary"
        self.yolo_loss = dynamic_yolo_loss(
            self.categories,
            self.gridx,
            self.gridy,
            self.gridz,
            1,
            self.box_vector,
            self.entropy,
            self.yolo_v0,
            self.yolo_v1,
            self.yolo_v2,
        )

    def loadData(self):

        (X, Y), axes = utils.load_full_training_data(
            self.npz_directory, self.npz_name, verbose=True
        )

        (X_val, Y_val), axes = utils.load_full_training_data(
            self.npz_directory, self.npz_val_name, verbose=True
        )

        self.Xoriginal = X
        self.Xoriginal_val = X_val

        self.X = X
        self.Y = Y[:, :, 0]
        self.X_val = X_val
        self.Y_val = Y_val[:, :, 0]

        self.axes = axes
        self.Y = self.Y.reshape((self.Y.shape[0], 1, 1, 1, self.Y.shape[1]))
        self.Y_val = self.Y_val.reshape(
            (self.Y_val.shape[0], 1, 1, 1, self.Y_val.shape[1])
        )

    def TrainModel(self):

        input_shape = (
            self.X.shape[1],
            self.X.shape[2],
            self.X.shape[3],
            self.X.shape[4],
        )

        print(input_shape)
        print(self.last_activation)
        Path(self.model_dir).mkdir(exist_ok=True)

        model_weights = os.path.join(self.model_dir, "weights.h5")
        if os.path.exists(model_weights):

            self.model_weights = model_weights
            print("loading weights")
        else:

            self.model_weights = None

        dummyY = np.zeros(
            [
                self.Y.shape[0],
                1,
                self.Y.shape[1],
                self.Y.shape[2],
                self.categories + self.box_vector,
            ]
        )
        dummyY[:, :, :, :, : self.Y.shape[4]] = self.Y

        dummyY_val = np.zeros(
            [
                self.Y_val.shape[0],
                1,
                self.Y_val.shape[1],
                self.Y_val.shape[2],
                self.categories + self.box_vector,
            ]
        )
        dummyY_val[:, :, :, :, : self.Y_val.shape[4]] = self.Y_val

        self.Y = dummyY
        self.Y_val = dummyY_val

        self.Trainingmodel = self.model_keras(
            input_shape,
            self.categories,
            box_vector=self.box_vector,
            stage_number=self.stage_number,
            last_conv_factor=self.last_conv_factor,
            depth=self.depth,
            start_kernel=self.start_kernel,
            mid_kernel=self.mid_kernel,
            startfilter=self.startfilter,
            input_weights=self.model_weights,
            last_activation=self.last_activation,
        )

        sgd = optimizers.SGD(
            lr=self.learning_rate, momentum=0.99, decay=1e-6, nesterov=True
        )
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
        prate = plotters.PlotHistory(
            self.Trainingmodel,
            self.X_val,
            self.Y_val,
            self.key_categories,
            self.key_cord,
            self.gridx,
            self.gridy,
            plot=self.show,
            nboxes=1,
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

        self.Trainingmodel.save(model_weights)

    def predict(
        self,
        image: np.ndarray,
        savedir: str = None,
        n_tiles: tuple = (1, 1),
        overlap_percent: float = 0.8,
        event_threshold: float = 0.5,
        iou_threshold: float = 0.0001,
        radius: int = 10,
        normalize: bool = True,
        activations: bool = False,
    ):

        self.image = image
        self.Colorimage = np.zeros(
            [self.image.shape[0], self.image.shape[1], self.image.shape[2], 3],
            dtype="uint16",
        )
        self.Maskimage = np.zeros(
            [self.image.shape[0], self.image.shape[1], self.image.shape[2], 3],
            dtype="float32",
        )
        self.image_mask_c1 = np.zeros(self.image.shape, dtype="float32")
        self.image_mask_c2 = np.zeros(self.image.shape, dtype="float32")
        self.Colorimage[:, :, :, 0] = self.image
        self.Maskimage[:, :, :, 0] = self.image
        self.radius = radius
        self.savedir = savedir
        self.n_tiles = n_tiles
        self.overlap_percent = overlap_percent
        self.iou_threshold = iou_threshold
        self.event_threshold = event_threshold
        self.normalize = normalize
        self.activations = activations
        self.iou_classedboxes = {}
        self.all_iou_classedboxes = {}
        self.maskboxes = {}

        self.model = self._build()

        self.first_pass_predict()

    def _build(self):

        model_weights = os.path.join(self.model_dir, "weights.h5")
        Model = load_model(
            model_weights,
            custom_objects={"loss": self.yolo_loss, "Concat": Concat},
        )

        return Model

    def first_pass_predict(self):

        eventboxes = []
        classedboxes = {}
        print("Detecting focus planes")
        if self.normalize:
            self.image = normalizeFloatZeroOne(self.image, 1, 99.8)

        for inputz in tqdm(range(0, self.image.shape[0])):
            if inputz <= self.image.shape[0] - self.imagez:

                eventboxes = []
                self.currentZ = inputz
                smallimage = CreateVolume(self.image, self.imagez, inputz)

                # self.current_Zpoints = [(j,k) for j in range(smallimage.shape[1]) for k in range(smallimage.shape[2]) ]
                # Cut off the region for training movie creation
                # Break image into tiles if neccessary
                predictions, allx, ally = self.predict_main(smallimage)
                # Iterate over tiles
                for p in range(0, len(predictions)):

                    sum_z_prediction = predictions[p]

                    if sum_z_prediction is not None:
                        # For each tile the prediction vector has shape N H W Categories + Training Vector labels
                        for i in range(0, sum_z_prediction.shape[0]):
                            z_prediction = sum_z_prediction[i]
                            boxprediction = focyoloprediction(
                                ally[p],
                                allx[p],
                                z_prediction,
                                self.stride,
                                inputz,
                                self.config,
                                self.key_categories,
                            )

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
                self.nms()
                if self.savedir is not None:
                    self.to_csv()
                self.draw()
                eventboxes = []
                classedboxes = {}

        self.print_planes()
        self.fit_curve()
        self.genmap()

    def nms(self):

        best_iou_classedboxes = {}
        all_best_iou_classedboxes = {}
        self.all_iou_classedboxes = {}
        self.iou_classedboxes = {}
        for (event_name, event_label) in self.key_categories.items():
            if event_label > 0:
                # Get all events

                sorted_event_box = self.classedboxes[event_name][0]

                sorted_event_box = sorted(
                    sorted_event_box, key=lambda x: x[event_name], reverse=True
                )

                scores = [
                    sorted_event_box[i][event_name]
                    for i in range(len(sorted_event_box))
                ]
                best_sorted_event_box, all_boxes = simpleaveragenms(
                    sorted_event_box,
                    scores,
                    self.iou_threshold,
                    self.event_threshold,
                    event_name,
                )
                all_best_iou_classedboxes[event_name] = [all_boxes]
                best_iou_classedboxes[event_name] = [best_sorted_event_box]
                if self.activations:
                    if event_name in self.all_iou_classedboxes:
                        boxes = self.all_iou_classedboxes[event_name]
                        boxes.append(best_sorted_event_box)
                        self.all_iou_classedboxes[event_name] = boxes
                    else:
                        self.all_iou_classedboxes[
                            event_name
                        ] = best_sorted_event_box
        self.iou_classedboxes = best_iou_classedboxes
        self.all_iou_classedboxes = all_best_iou_classedboxes

    def to_csv(self):

        for (event_name, event_label) in self.key_categories.items():

            if event_label > 0:
                zlocations = []
                scores = []
                max_scores = []
                iou_current_event_box = self.iou_classedboxes[event_name][0]
                zcenter = iou_current_event_box["real_z_event"]
                max_score = iou_current_event_box["max_score"]
                score = iou_current_event_box[event_name]

                zlocations.append(zcenter)
                scores.append(score)
                max_scores.append(max_score)
                print(zlocations, scores)
                event_count = np.column_stack([zlocations, scores, max_scores])
                event_count = sorted(
                    event_count, key=lambda x: x[0], reverse=False
                )
                event_data = []
                csvname = self.savedir + "/" + event_name + "_FocusQuality"
                writer = csv.writer(open(csvname + ".csv", "a"))
                filesize = os.stat(csvname + ".csv").st_size
                if filesize < 1:
                    writer.writerow(["Z", "Score", "Max_score"])
                for line in event_count:
                    if line not in event_data:
                        event_data.append(line)
                    writer.writerows(event_data)
                    event_data = []

    def fit_curve(self):

        for (event_name, event_label) in self.key_categories.items():

            if event_label > 0:
                readcsvname = self.savedir + "/" + event_name + "_FocusQuality"
                self.dataset = pd.read_csv(readcsvname, delimiter=",")
                self.dataset_index = self.dataset.index

                Z = self.dataset[self.dataset.keys()[0]][1:]
                score = self.dataset[self.dataset.keys()[1]][1:]

                H, A, mu0, sigma = gauss_fit(np.array(Z), np.array(score))
                csvname = (
                    self.savedir + "/" + event_name + "_GaussFitFocusQuality"
                )
                writer = csv.writer(open(csvname + ".csv", "a"))
                filesize = os.stat(csvname + ".csv").st_size
                if filesize < 1:
                    writer.writerow(["Amplitude", "Mean", "Sigma"])
                    writer.writerow([A, mu0, sigma])

    def print_planes(self):
        for (event_name, event_label) in self.key_categories.items():
            if event_label > 0:
                csvfname = (
                    self.savedir + "/" + event_name + "_focus_quality" + ".csv"
                )
                dataset = pd.read_csv(csvfname, skiprows=0)
                z = dataset[dataset.keys()[0]][1:]
                score = dataset[dataset.keys()[1]][1:]
                maxz = z[np.argmax(score)] + 2
                print("Best Zs" + "for" + event_name + "at" + str(maxz))

    def draw(self):

        for (event_name, event_label) in self.key_categories.items():

            if event_label > 0:

                xlocations = []
                ylocations = []
                scores = []
                zlocations = []
                heights = []
                widths = []
                iou_current_event_boxes = self.all_iou_classedboxes[
                    event_name
                ][0]

                for iou_current_event_box in iou_current_event_boxes:

                    xcenter = iou_current_event_box["xcenter"]
                    ycenter = iou_current_event_box["ycenter"]
                    zcenter = iou_current_event_box["real_z_event"]
                    xstart = iou_current_event_box["xstart"]
                    ystart = iou_current_event_box["ystart"]
                    xend = xstart + iou_current_event_box["width"]
                    yend = ystart + iou_current_event_box["height"]
                    score = iou_current_event_box[event_name]

                    if event_label == 1:
                        for x in range(int(xstart), int(xend)):
                            for y in range(int(ystart), int(yend)):
                                if (
                                    y < self.image.shape[1]
                                    and x < self.image.shape[2]
                                ):
                                    self.Maskimage[int(zcenter), y, x, 1] = (
                                        self.Maskimage[int(zcenter), y, x, 1]
                                        + score
                                    )
                    else:

                        for x in range(int(xstart), int(xend)):
                            for y in range(int(ystart), int(yend)):
                                if (
                                    y < self.image.shape[1]
                                    and x < self.image.shape[2]
                                ):
                                    self.Maskimage[int(zcenter), y, x, 2] = (
                                        self.Maskimage[int(zcenter), y, x, 2]
                                        + score
                                    )

                    if score > 0.9:

                        xlocations.append(round(xcenter))
                        ylocations.append(round(ycenter))
                        scores.append(score)
                        zlocations.append(zcenter)
                        heights.append(iou_current_event_box["height"])
                        widths.append(iou_current_event_box["width"])

    def overlaptiles(self, sliceregion):

        if self.n_tiles == (1, 1):
            patch = []
            rowout = []
            column = []
            patchx = sliceregion.shape[2] // self.n_tiles[0]
            patchy = sliceregion.shape[1] // self.n_tiles[1]
            patchshape = (patchy, patchx)
            smallpatch, smallrowout, smallcolumn = chunk_list(
                sliceregion, patchshape, self.stride, [0, 0]
            )
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
                rowstart = 0
                colstart = 0
                pairs = []
                # row is y, col is x

                while rowstart < sliceregion.shape[1] - patchy:
                    colstart = 0
                    while colstart < sliceregion.shape[2] - patchx:

                        # Start iterating over the tile with jumps = stride of the fully convolutional network.
                        pairs.append([rowstart, colstart])
                        colstart += jumpx
                    rowstart += jumpy

                # Include the last patch
                rowstart = sliceregion.shape[1] - patchy
                colstart = 0
                while colstart < sliceregion.shape[2]:
                    pairs.append([rowstart, colstart])
                    colstart += jumpx
                rowstart = 0
                colstart = sliceregion.shape[2] - patchx
                while rowstart < sliceregion.shape[1]:
                    pairs.append([rowstart, colstart])
                    rowstart += jumpy

                if (
                    sliceregion.shape[1] >= self.imagey
                    and sliceregion.shape[2] >= self.imagex
                ):

                    patch = []
                    rowout = []
                    column = []
                    for pair in pairs:
                        smallpatch, smallrowout, smallcolumn = chunk_list(
                            sliceregion, patchshape, self.stride, pair
                        )
                        if (
                            smallpatch.shape[1] >= self.imagey
                            and smallpatch.shape[2] >= self.imagex
                        ):
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
                smallpatch, smallrowout, smallcolumn = chunk_list(
                    sliceregion, patchshape, self.stride, [0, 0]
                )
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

    def make_batch_patches(self, sliceregion):

        prediction_vector = self.model.predict(
            np.expand_dims(sliceregion, -1), verbose=0
        )
        return prediction_vector


def chunk_list(image, patchshape, stride, pair):
    rowstart = pair[0]
    colstart = pair[1]

    endrow = rowstart + patchshape[0]
    endcol = colstart + patchshape[1]

    if endrow > image.shape[1]:
        endrow = image.shape[1]
    if endcol > image.shape[2]:
        endcol = image.shape[2]

    region = (
        slice(0, image.shape[0]),
        slice(rowstart, endrow),
        slice(colstart, endcol),
    )

    # The actual pixels in that region.
    patch = image[region]

    # Always normalize patch that goes into the netowrk for getting a prediction score

    return patch, rowstart, colstart


def CreateVolume(patch, imagez, timepoint):

    starttime = timepoint
    endtime = timepoint + imagez
    smallimg = patch[starttime:endtime, :]

    return smallimg


def normalizeZeroOne(x):
    x = x.astype("float32")

    minVal = np.min(x)
    maxVal = np.max(x)

    x = (x - minVal) / (maxVal - minVal + 1.0e-20)

    return x


def gauss(x, H, A, x0, sigma):
    return H + A * np.exp(-((x - x0) ** 2) / (2 * sigma**2))


def gauss_fit(x, y):
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    popt, pcov = curve_fit(gauss, x, y, p0=[min(y), max(y), mean, sigma])
    return popt
