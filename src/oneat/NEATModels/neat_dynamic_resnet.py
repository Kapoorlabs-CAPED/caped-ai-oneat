import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras import callbacks
from tensorflow.keras.models import load_model
from skimage.measure import label
from skimage.morphology import dilation, disk
from tifffile import imwrite
from tqdm import tqdm
import datetime
from oneat.NEATModels import nets
from oneat.NEATModels.loss import static_yolo_loss
from oneat.NEATModels.nets import Concat
from oneat.NEATUtils import plotters, utils
from oneat.NEATUtils.utils import (
    GenerateMarkers,
    MakeTrees,
    MidSlicesSum,
    dynamic_nms,
    get_nearest,
    gold_nms,
    load_json,
    normalizeFloatZeroOne,
    pad_timelapse,
    save_dynamic_csv,
    yoloprediction,
)
from oneat.pretrained import (
    get_model_details,
    get_model_instance,
    get_registered_models,
)

Boxname = "ImageIDBox"
EventBoxname = "EventIDBox"


class NEATTResNet:
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

    def __init__(
        self, config: dict, model_dir: str, catconfig: dict, cordconfig: dict
    ):

        self.config = config
        self.catconfig = catconfig
        self.cordconfig = cordconfig
        self.model_dir = model_dir
        self.key_cord = self.cordconfig
        self.categories = len(self.catconfig)
        self.key_categories = self.catconfig
        if self.config is not None:
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
        if self.config is None:

            self.config = load_json(
                os.path.join(self.model_dir, "parameters.json")
            )
            self.npz_directory = self.config["npz_directory"]
            self.npz_name = self.config["npz_name"]
            self.npz_val_name = self.config["npz_val_name"]
            self.box_vector = self.config["box_vector"]
            self.show = self.config["show"]
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
            self.imaget = (
                self.config["size_tminus"] + self.config["size_tplus"] + 1
            )
            self.size_tminus = self.config["size_tminus"]
            self.size_tplus = self.config["size_tplus"]
            self.nboxes = self.config["nboxes"]
            self.stage_number = self.config["stage_number"]
            self.last_conv_factor = 2 ** (self.stage_number - 1)
            self.gridx = 1
            self.gridy = 1
            self.gridt = 1
            self.yolo_v0 = self.config["yolo_v0"]
            self.yolo_v1 = self.config["yolo_v1"]
            self.yolo_v2 = self.config["yolo_v2"]
            self.stride = self.config["stride"]

        self.X = None
        self.Y = None
        self.axes = None
        self.X_val = None
        self.Y_val = None
        self.Trainingmodel = None
        self.Xoriginal = None
        self.Xoriginal_val = None

        self.model_keras = nets.resnet_v2

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

    @classmethod
    def local_from_pretrained(cls, name_or_alias=None):
        try:
            print(cls)
            get_model_details(cls, name_or_alias, verbose=True)
            return get_model_instance(cls, name_or_alias)
        except ValueError:
            if name_or_alias is not None:
                print(
                    "Could not find model with name or alias '%s'"
                    % (name_or_alias),
                    file=sys.stderr,
                )
                sys.stderr.flush()
            get_registered_models(cls, verbose=True)

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
        self.X = tf.reshape(
            self.X,
            (
                self.X.shape[0],
                self.X.shape[2],
                self.X.shape[3],
                self.X.shape[1],
            ),
        )
        self.Y = Y[:, :, 0]
        self.X_val = X_val
        self.X_val = tf.reshape(
            self.X_val,
            (
                self.X_val.shape[0],
                self.X_val.shape[2],
                self.X_val.shape[3],
                self.X_val.shape[1],
            ),
        )
        self.Y_val = Y_val[:, :, 0]

        self.axes = axes
        self.Y = self.Y.reshape((self.Y.shape[0], 1, 1, self.Y.shape[1]))
        self.Y_val = self.Y_val.reshape(
            (self.Y_val.shape[0], 1, 1, self.Y_val.shape[1])
        )

    def TrainModel(self):

        input_shape = (self.X.shape[1], self.X.shape[2], self.X.shape[3])

        Path(self.model_dir).mkdir(exist_ok=True)

        if self.yolo_v2:

            for i in range(self.Y.shape[0]):

                if self.Y[i, :, :, 0] == 1:
                    self.Y[i, :, :, -1] = 1
            for i in range(self.Y_val.shape[0]):

                if self.Y_val[i, :, :, 0] == 1:
                    self.Y_val[i, :, :, -1] = 1
        Y_rest = self.Y[:, :, :, self.categories :]

        model_weights = os.path.join(self.model_dir, "weights.h5")
        if os.path.exists(model_weights):

            self.model_weights = model_weights
            print("loading weights")
        else:

            self.model_weights = None

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

        self.Trainingmodel = self.model_keras(
            input_shape,
            self.categories,
            box_vector=Y_rest.shape[-1],
            nboxes=self.nboxes,
            stage_number=self.stage_number,
            depth=self.depth,
            start_kernel=self.start_kernel,
            mid_kernel=self.mid_kernel,
            startfilter=self.startfilter,
            input_weights=self.model_weights,
            last_activation=self.last_activation,
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

    def get_markers(
        self,
        segimage,
        start_project_mid=4,
        end_project_mid=4,
        downsamplefactor=1,
        dtype="uint16",
    ):

        self.segimage = segimage
        self.pad_width = (self.config["imagey"], self.config["imagex"])
        self.downsamplefactor = downsamplefactor
        print("Obtaining Markers")
        self.segimage = segimage.astype(dtype)
        self.markers = GenerateMarkers(
            self.segimage,
            start_project_mid=start_project_mid,
            end_project_mid=end_project_mid,
            pad_width=self.pad_width,
        )
        self.marker_tree = MakeTrees(self.markers)

        return self.marker_tree

    def predict(
        self,
        image: np.ndarray,
        savedir: str = None,
        n_tiles: tuple = (1, 1),
        overlap_percent: float = 0.8,
        dtype: np.dtype = np.uint8,
        event_threshold: float = 0.5,
        event_confidence: float = 0.5,
        iou_threshold: float = 0.1,
        fidelity: int = 1,
        start_project_mid: int = 4,
        end_project_mid: int = 4,
        erosion_iterations: int = 1,
        marker_tree: dict = None,
        remove_markers: bool = False,
        normalize: bool = True,
        center_oneat: bool = True,
        nms_function: str = "iou",
        activations: bool = False,
    ):

        self.dtype = dtype
        self.nms_function = nms_function
        self.image = image.astype(self.dtype)
        self.start_project_mid = start_project_mid
        self.end_project_mid = end_project_mid
        self.ndim = len(self.image.shape)
        self.normalize = normalize
        self.activations = activations
        self.z = 0
        if self.ndim == 4:
            self.z = self.image.shape[1] // 2
            print(
                f"Image {self.image.shape} is {self.ndim} dimensional, projecting around the center {self.image.shape[1]//2} - {self.start_project_mid} to {self.image.shape[1]//2} + {self.end_project_mid}"
            )
            self.image = MidSlicesSum(
                self.image,
                self.start_project_mid,
                self.end_project_mid,
                axis=1,
            )
            self.z = (
                self.z - (self.start_project_mid + self.end_project_mid) // 2
            )

        self.erosion_iterations = erosion_iterations
        self.heatmap = np.zeros(self.image.shape, dtype="uint16")
        self.eventmarkers = np.zeros(self.image.shape, dtype="uint16")
        self.savedir = savedir
        if self.savedir is not None:
            Path(self.savedir).mkdir(exist_ok=True)
        if len(n_tiles) > 2:
            n_tiles = (n_tiles[-2], n_tiles[-1])
        self.n_tiles = n_tiles
        self.fidelity = fidelity
        self.overlap_percent = overlap_percent
        self.iou_threshold = iou_threshold
        self.event_threshold = event_threshold
        self.event_confidence = event_confidence
        self.originalimage = self.image
        self.center_oneat = center_oneat
        self.iou_classedboxes = {}
        self.all_iou_classedboxes = {}
        self.model = self._build()
        self.marker_tree = marker_tree
        self.remove_markers = remove_markers

        if self.normalize:
            self.image = normalizeFloatZeroOne(
                self.image, 1, 99.8, dtype=self.dtype
            )
        if self.remove_markers:
            self.generate_maps = False

            self.image = pad_timelapse(self.image, self.pad_width)
            print(f"zero padded image shape ${self.image.shape}")
            self.first_pass_predict()
            self.second_pass_predict()
        if not self.remove_markers:
            self.generate_maps = False

            self.image = pad_timelapse(self.image, self.pad_width)
            print(f"zero padded image shape ${self.image.shape}")
            self.second_pass_predict()
        if self.remove_markers is None:
            self.generate_maps = True
            self.default_pass_predict()

    def _build(self):

        model_weights = os.path.join(self.model_dir, "weights.h5")
        Model = load_model(
            model_weights,
            custom_objects={"loss": self.yolo_loss, "Concat": Concat},
        )
        return Model

    def default_pass_predict(self):
        eventboxes = []
        classedboxes = {}
        count = 0
        if self.savedir is not None:
            heatsavename = self.savedir + "/" + "_Heat"

        print("Detecting event locations")
        for inputtime in tqdm(range(0, self.image.shape[0])):
            if (
                inputtime < self.image.shape[0] - self.imaget
                and inputtime > int(self.imaget) // 2
            ):
                count = count + 1
                if (
                    inputtime % (self.image.shape[0] // 4) == 0
                    and inputtime > 0
                    or inputtime >= self.image.shape[0] - self.imaget - 1
                ):

                    imwrite((heatsavename + ".tif"), self.heatmap)

                smallimage = CreateVolume(
                    self.image, self.size_tminus, self.size_tplus, inputtime
                )
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
                            boxprediction = yoloprediction(
                                ally[p],
                                allx[p],
                                time_prediction,
                                self.stride,
                                inputtime,
                                self.config,
                                self.key_categories,
                                self.key_cord,
                                self.nboxes,
                                "detection",
                                "static",
                                marker_tree=self.marker_tree,
                                center_oneat=self.center_oneat,
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

                if inputtime > 0 and inputtime % (self.imaget) == 0:

                    self.nms()
                    if self.savedir is not None:
                        self.to_csv()
                    eventboxes = []
                    classedboxes = {}
                    count = 0

    def first_pass_predict(self):

        print("Detecting background event locations")
        eventboxes = []
        classedboxes = {}
        remove_candidates = {}

        for inputtime in tqdm(range(0, self.image.shape[0])):
            if (
                inputtime < self.image.shape[0] - self.imaget
                and inputtime > int(self.imaget) // 2
            ):

                remove_candidates_list = []
                smallimage = CreateVolume(
                    self.image, self.size_tminus, self.size_tplus, inputtime
                )
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
                            boxprediction = yoloprediction(
                                ally[p],
                                allx[p],
                                time_prediction,
                                self.stride,
                                inputtime,
                                self.config,
                                self.key_categories,
                                self.key_cord,
                                self.nboxes,
                                "detection",
                                "static",
                                marker_tree=self.marker_tree,
                                center_oneat=False,
                            )

                            if boxprediction is not None:
                                eventboxes = eventboxes + boxprediction
            for (event_name, event_label) in self.key_categories.items():

                if event_label == 0:
                    current_event_box = []
                    for box in eventboxes:

                        event_prob = box[event_name]
                        if event_prob >= 0.5:

                            current_event_box.append(box)
                            classedboxes[event_name] = [current_event_box]

            self.classedboxes = classedboxes
            if len(self.classedboxes) > 0:
                self.fast_nms()
                for (event_name, event_label) in self.key_categories.items():

                    if event_label == 0:
                        iou_current_event_boxes = self.iou_classedboxes[
                            event_name
                        ][0]
                        iou_current_event_boxes = sorted(
                            iou_current_event_boxes,
                            key=lambda x: x[event_name],
                            reverse=True,
                        )
                        for box in iou_current_event_boxes:
                            closest_location = get_nearest(
                                self.marker_tree,
                                box["ycenter"],
                                box["xcenter"],
                                box["real_time_event"],
                            )
                            if closest_location is not None:
                                ycentermean, xcentermean = closest_location
                                try:
                                    remove_candidates_list = remove_candidates[
                                        str(int(box["real_time_event"]))
                                    ]
                                    if (
                                        ycentermean * self.downsamplefactor,
                                        xcentermean * self.downsamplefactor,
                                    ) not in remove_candidates_list:
                                        remove_candidates_list.append(
                                            (
                                                ycentermean
                                                * self.downsamplefactor,
                                                xcentermean
                                                * self.downsamplefactor,
                                            )
                                        )
                                        remove_candidates[
                                            str(int(box["real_time_event"]))
                                        ] = remove_candidates_list
                                except ValueError:
                                    remove_candidates_list.append(
                                        (
                                            ycentermean
                                            * self.downsamplefactor,
                                            xcentermean
                                            * self.downsamplefactor,
                                        )
                                    )
                                    remove_candidates[
                                        str(int(box["real_time_event"]))
                                    ] = remove_candidates_list

                eventboxes = []
                classedboxes = {}

    def second_pass_predict(self):

        print("Detecting event locations")
        eventboxes = []
        classedboxes = {}
        self.n_tiles = (1, 1)
        if self.savedir is not None:
            heatsavename = self.savedir + "/" + "_Heat"

        for inputtime in tqdm(
            range(int(self.imaget) // 2, self.image.shape[0])
        ):
            if inputtime < self.image.shape[0] - self.imaget:

                smallimage = CreateVolume(
                    self.image, self.size_tminus, self.size_tplus, inputtime
                )

                if (
                    inputtime % (self.image.shape[0] // 4) == 0
                    and inputtime > 0
                    or inputtime >= self.image.shape[0] - self.imaget - 1
                ):
                    markers_current = dilation(
                        self.eventmarkers[inputtime, :], disk(2)
                    )
                    self.eventmarkers[inputtime, :] = label(
                        markers_current.astype("uint16")
                    )
                    imwrite((heatsavename + ".tif"), self.heatmap)
                if str(int(inputtime)) in self.marker_tree:
                    tree, location = self.marker_tree[str(int(inputtime))]
                    for i in range(len(location)):

                        crop_xminus = (
                            location[i][1]
                            - int(self.imagex / 2) * self.downsamplefactor
                        )
                        crop_xplus = (
                            location[i][1]
                            + int(self.imagex / 2) * self.downsamplefactor
                        )
                        crop_yminus = (
                            location[i][0]
                            - int(self.imagey / 2) * self.downsamplefactor
                        )
                        crop_yplus = (
                            location[i][0]
                            + int(self.imagey / 2) * self.downsamplefactor
                        )
                        region = (
                            slice(0, smallimage.shape[0]),
                            slice(int(crop_yminus), int(crop_yplus)),
                            slice(int(crop_xminus), int(crop_xplus)),
                        )

                        crop_image = smallimage[region]

                        if (
                            crop_image.shape[0] >= self.imaget
                            and crop_image.shape[1]
                            >= self.imagey * self.downsamplefactor
                            and crop_image.shape[2]
                            >= self.imagex * self.downsamplefactor
                        ):
                            # Now apply the prediction for counting real events

                            ycenter = location[i][0]
                            xcenter = location[i][1]

                            predictions, allx, ally = self.predict_main(
                                crop_image
                            )
                            sum_time_prediction = predictions[0]
                            if sum_time_prediction is not None:
                                # For each tile the prediction vector has shape N H W Categories + Training Vector labels
                                time_prediction = sum_time_prediction[0]
                                boxprediction = yoloprediction(
                                    0,
                                    0,
                                    time_prediction,
                                    self.stride,
                                    inputtime,
                                    self.config,
                                    self.key_categories,
                                    self.key_cord,
                                    self.nboxes,
                                    "detection",
                                    "static",
                                    center_oneat=self.center_oneat,
                                )
                                if (
                                    boxprediction is not None
                                    and len(boxprediction) > 0
                                    and xcenter - self.pad_width[1] > 0
                                    and ycenter - self.pad_width[0] > 0
                                    and xcenter - self.pad_width[1]
                                    < self.image.shape[2]
                                    and ycenter - self.pad_width[0]
                                    < self.image.shape[1]
                                ):

                                    boxprediction[0][
                                        "real_time_event"
                                    ] = inputtime
                                    boxprediction[0]["xcenter"] = (
                                        xcenter - self.pad_width[1]
                                    )
                                    boxprediction[0]["ycenter"] = (
                                        ycenter - self.pad_width[0]
                                    )
                                    boxprediction[0]["xstart"] = (
                                        boxprediction[0]["xcenter"]
                                        - int(self.imagex / 2)
                                        * self.downsamplefactor
                                    )
                                    boxprediction[0]["ystart"] = (
                                        boxprediction[0]["ycenter"]
                                        - int(self.imagey / 2)
                                        * self.downsamplefactor
                                    )
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
                self.iou_classedboxes = classedboxes
                self.to_csv()
                eventboxes = []
                classedboxes = {}

    def fast_nms(self):

        best_iou_classedboxes = {}
        self.iou_classedboxes = {}
        for (event_name, event_label) in self.key_categories.items():
            if event_label == 0:
                # best_sorted_event_box = self.classedboxes[event_name][0]
                best_sorted_event_box = dynamic_nms(
                    self.heatmap,
                    self.classedboxes,
                    event_name,
                    self.downsamplefactor,
                    self.iou_threshold,
                    self.event_threshold,
                    self.imagex,
                    self.imagey,
                    self.fidelity,
                    generate_map=self.generate_maps,
                    nms_function=self.nms_function,
                )

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

    def nms(self):

        best_iou_classedboxes = {}
        self.iou_classedboxes = {}
        for (event_name, event_label) in self.key_categories.items():
            if event_label > 0:
                # best_sorted_event_box = self.classedboxes[event_name][0]
                if self.remove_markers is not None:
                    best_sorted_event_box = gold_nms(
                        self.heatmap,
                        self.eventmarkers,
                        self.classedboxes,
                        event_name,
                        1,
                        self.iou_threshold,
                        self.event_threshold,
                        self.imagex,
                        self.imagey,
                        generate_map=self.generate_maps,
                        nms_function=self.nms_function,
                    )

                if self.remove_markers is None:
                    best_sorted_event_box = dynamic_nms(
                        self.heatmap,
                        self.classedboxes,
                        event_name,
                        self.downsamplefactor,
                        self.iou_threshold,
                        self.event_threshold,
                        self.imagex,
                        self.imagey,
                        self.fidelity,
                        generate_map=self.generate_maps,
                        nms_function=self.nms_function,
                    )

                best_iou_classedboxes[event_name] = [best_sorted_event_box]

        self.iou_classedboxes = best_iou_classedboxes

    def to_csv(self):
        if self.remove_markers is not None:
            save_dynamic_csv(
                self.key_categories,
                self.iou_classedboxes,
                self.savedir,
                z=self.z,
            )
        if self.remove_markers is None:
            save_dynamic_csv(
                self.key_categories,
                self.iou_classedboxes,
                self.savedir,
                z=self.z,
            )

    def overlaptiles(self, sliceregion: np.ndarray):

        if self.n_tiles == (1, 1):
            patch = []
            rowout = []
            column = []
            patchx = sliceregion.shape[2] // self.n_tiles[0]
            patchy = sliceregion.shape[1] // self.n_tiles[1]
            patchshape = (patchy, patchx)
            smallpatch, smallrowout, smallcolumn = chunk_list(
                sliceregion, patchshape, [0, 0]
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

                if (
                    sliceregion.shape[1] >= self.imagey
                    and sliceregion.shape[2] >= self.imagex
                ):

                    patch = []
                    rowout = []
                    column = []
                    for pair in pairs:
                        smallpatch, smallrowout, smallcolumn = chunk_list(
                            sliceregion, patchshape, pair
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
                    sliceregion, patchshape, [0, 0]
                )
                patch.append(smallpatch)
                rowout.append(smallrowout)
                column.append(smallcolumn)
        self.patch = patch
        self.sy = rowout
        self.sx = column

    def predict_main(self, sliceregion: np.ndarray):
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

    def make_patches(self, sliceregion: np.ndarray):

        smallimg = np.expand_dims(sliceregion, 0)
        smallimg = tf.reshape(
            smallimg,
            (
                smallimg.shape[0],
                smallimg.shape[2],
                smallimg.shape[3],
                smallimg.shape[1],
            ),
        )
        prediction_vector = self.model.predict(smallimg, verbose=0)

        return prediction_vector


def CreateVolume(
    patch: np.ndarray, size_tminus: int, size_tplus: int, timepoint: int
):
    starttime = timepoint - int(size_tminus)
    endtime = timepoint + int(size_tplus) + 1
    smallimg = patch[starttime:endtime, :]
    return smallimg


def chunk_list(image: np.ndarray, patchshape: tuple, pair: tuple):
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
