import random

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import models
from matplotlib import cm


class PlotHistory(keras.callbacks.Callback):
    def __init__(
        self,
        Trainingmodel: models,
        X: np.ndarray,
        Y: np.ndarray,
        key_categories: dict,
        key_cord: dict,
        gridx: int,
        gridy: int,
        plot: bool = False,
        nboxes: int = 1,
    ):
        self.Trainingmodel = Trainingmodel
        self.X = X
        self.Y = Y
        self.plot = plot
        self.gridx = gridx
        self.gridy = gridy
        self.nboxes = nboxes
        self.key_cord = key_cord
        self.key_categories = key_categories

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        if self.plot:
            self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):

        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get("loss"))
        self.val_losses.append(logs.get("val_loss"))
        self.acc.append(logs.get("accuracy"))
        self.val_acc.append(logs.get("val_accuracy"))

        self.i += 1
        if self.plot:
            f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

            ax1.set_yscale("log")
            ax1.plot(self.x, self.losses, label="loss")
            ax1.plot(self.x, self.val_losses, label="val_loss")
            ax1.legend()

            ax2.plot(self.x, self.acc, label="accuracy")
            ax2.plot(self.x, self.val_acc, label="val_accuracy")
            ax2.legend()
            plt.show()
            # clear_output(True)
        idx = random.randint(1, self.X.shape[0] - 1)
        Printpredict(
            idx,
            self.Trainingmodel,
            self.X,
            self.Y,
            self.key_categories,
            self.key_cord,
            self.gridx,
            self.gridy,
            plot=self.plot,
            nboxes=self.nboxes,
        )


def Printpredict(
    idx,
    model,
    data,
    Truelabel,
    key_categories,
    key_cord,
    plot=False,
    nboxes=1,
):

    Image = data[idx]
    Truelabel = Truelabel[idx]
    data = np.expand_dims(Image, 0)
    prediction = model.predict(data)
    cols = 5

    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, data.shape[1], figsize=(5 * cols, 5))
        fig.figsize = (20, 10)
    # The prediction vector is (1, categories + box_vector) dimensional, input data is (N T H W C) C is 1 in our case
    for j in range(0, data.shape[1]):

        img = Image[j, :, :, 0]
        if plot:
            ax[j].imshow(img, cm.Spectral)
    for i in range(0, prediction.shape[0]):

        try:
            maxevent = np.argmax(
                prediction[i, :, :, : len(key_categories)], axis=-1
            )
            trueevent = np.argmax(
                Truelabel[0, 0, : len(key_categories)], axis=-1
            )
        except ValueError:

            maxevent = np.argmax(
                prediction[i, :, :, :, : len(key_categories)], axis=-1
            )
            trueevent = np.argmax(
                Truelabel[0, 0, 0, : len(key_categories)], axis=-1
            )
        for (k, v) in key_categories.items():
            if v == maxevent:
                maxlabel = k
            if v == trueevent:
                truelabel = k
        try:
            print(
                "Predicted cell:",
                maxlabel,
                "Probability:",
                prediction[i, 0, 0, maxevent],
            )
            print("True Cell type:", truelabel)
        except ValueError:
            print(prediction[i, 0, 0, :])
            print("True Cell type:", truelabel)
            # print( "Predicted cell:", maxlabel , "Probability:", prediction[i,0,0,0,maxevent])
            # print('True Cell type:', truelabel)

        if nboxes > 1:
            for b in range(1, nboxes - 1):
                try:
                    prediction[
                        i,
                        :,
                        :,
                        len(key_categories) : len(key_categories)
                        + len(key_cord),
                    ] += prediction[
                        i,
                        :,
                        :,
                        len(key_categories)
                        + b * len(key_cord) : len(key_categories)
                        + (b + 1) * len(key_cord),
                    ]
                    prediction[
                        i,
                        :,
                        :,
                        len(key_categories) : len(key_categories)
                        + len(key_cord),
                    ] = prediction[
                        i,
                        :,
                        :,
                        len(key_categories) : len(key_categories)
                        + len(key_cord),
                    ] / (
                        nboxes - 1
                    )
                except ValueError:
                    prediction[
                        i,
                        :,
                        :,
                        :,
                        len(key_categories) : len(key_categories)
                        + len(key_cord),
                    ] += prediction[
                        i,
                        :,
                        :,
                        len(key_categories)
                        + b * len(key_cord) : len(key_categories)
                        + (b + 1) * len(key_cord),
                    ]
                    prediction[
                        i,
                        :,
                        :,
                        :,
                        len(key_categories) : len(key_categories)
                        + len(key_cord),
                    ] = prediction[
                        i,
                        :,
                        :,
                        len(key_categories) : len(key_categories)
                        + len(key_cord),
                    ] / (
                        nboxes - 1
                    )
        for (k, v) in key_cord.items():
            try:
                print(k, prediction[i, :, :, len(key_categories) + v])
                print(
                    "True positional value",
                    k,
                    Truelabel[0, 0, len(key_categories) + v],
                )
            except ValueError:

                print(k, prediction[i, :, :, :, len(key_categories) + v])
                print(
                    "True positional value",
                    k,
                    Truelabel[0, 0, 0, len(key_categories) + v],
                )

    if plot:
        plt.show()


class PlotSilverHistory(keras.callbacks.Callback):
    def __init__(
        self,
        Trainingmodel: models,
        X: np.ndarray,
        Y: np.ndarray,
        key_categories: dict,
        key_cord: dict,
        gridx: int,
        gridy: int,
        plot: bool = False,
        nboxes: int = 1,
    ):
        self.Trainingmodel = Trainingmodel
        self.X = X
        self.Y = Y
        self.plot = plot
        self.gridx = gridx
        self.gridy = gridy
        self.nboxes = nboxes
        self.key_cord = key_cord
        self.key_categories = key_categories

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        if self.plot:
            self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):

        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get("loss"))
        self.val_losses.append(logs.get("val_loss"))
        self.acc.append(logs.get("accuracy"))
        self.val_acc.append(logs.get("val_accuracy"))

        self.i += 1
        if self.plot:
            f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

            ax1.set_yscale("log")
            ax1.plot(self.x, self.losses, label="loss")
            ax1.plot(self.x, self.val_losses, label="val_loss")
            ax1.legend()

            ax2.plot(self.x, self.acc, label="accuracy")
            ax2.plot(self.x, self.val_acc, label="val_accuracy")
            ax2.legend()
            plt.show()
            # clear_output(True)
        idx = random.randint(1, self.X.shape[0] - 1)
        Printsilverpredict(
            idx,
            self.Trainingmodel,
            self.X,
            self.Y,
            self.key_categories,
            self.key_cord,
            self.gridx,
            self.gridy,
            plot=self.plot,
            nboxes=self.nboxes,
        )


def Printsilverpredict(
    idx,
    model,
    data,
    Truelabel,
    key_categories,
    key_cord,
    plot=False,
    nboxes=1,
):

    Image = data[idx]
    Truelabel = Truelabel[idx]
    data = np.expand_dims(Image, 0)
    prediction = model.predict(data)
    cols = 5

    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, data.shape[1], figsize=(5 * cols, 5))
        fig.figsize = (20, 10)
    # The prediction vector is (1, categories + box_vector) dimensional, input data is (N H W T) C is T in our case
    for j in range(0, data.shape[-1]):

        img = Image[0, :, :, j]
        if plot:
            ax[j].imshow(img, cm.Spectral)
    for i in range(0, prediction.shape[0]):

        try:
            maxevent = np.argmax(
                prediction[i, :, :, : len(key_categories)], axis=-1
            )
            trueevent = np.argmax(
                Truelabel[0, 0, : len(key_categories)], axis=-1
            )
        except ValueError:

            maxevent = np.argmax(
                prediction[i, :, :, :, : len(key_categories)], axis=-1
            )
            trueevent = np.argmax(
                Truelabel[0, 0, 0, : len(key_categories)], axis=-1
            )
        for (k, v) in key_categories.items():
            if v == maxevent:
                maxlabel = k
            if v == trueevent:
                truelabel = k
        try:
            print(
                "Predicted cell:",
                maxlabel,
                "Probability:",
                prediction[i, 0, 0, maxevent],
            )
            print("True Cell type:", truelabel)
        except ValueError:
            print(prediction[i, 0, 0, :])
            print("True Cell type:", truelabel)
            # print( "Predicted cell:", maxlabel , "Probability:", prediction[i,0,0,0,maxevent])
            # print('True Cell type:', truelabel)

        if nboxes > 1:
            for b in range(1, nboxes - 1):
                try:
                    prediction[
                        i,
                        :,
                        :,
                        len(key_categories) : len(key_categories)
                        + len(key_cord),
                    ] += prediction[
                        i,
                        :,
                        :,
                        len(key_categories)
                        + b * len(key_cord) : len(key_categories)
                        + (b + 1) * len(key_cord),
                    ]
                    prediction[
                        i,
                        :,
                        :,
                        len(key_categories) : len(key_categories)
                        + len(key_cord),
                    ] = prediction[
                        i,
                        :,
                        :,
                        len(key_categories) : len(key_categories)
                        + len(key_cord),
                    ] / (
                        nboxes - 1
                    )
                except ValueError:
                    prediction[
                        i,
                        :,
                        :,
                        :,
                        len(key_categories) : len(key_categories)
                        + len(key_cord),
                    ] += prediction[
                        i,
                        :,
                        :,
                        len(key_categories)
                        + b * len(key_cord) : len(key_categories)
                        + (b + 1) * len(key_cord),
                    ]
                    prediction[
                        i,
                        :,
                        :,
                        :,
                        len(key_categories) : len(key_categories)
                        + len(key_cord),
                    ] = prediction[
                        i,
                        :,
                        :,
                        len(key_categories) : len(key_categories)
                        + len(key_cord),
                    ] / (
                        nboxes - 1
                    )
        for (k, v) in key_cord.items():
            try:
                print(k, prediction[i, :, :, len(key_categories) + v])
                print(
                    "True positional value",
                    k,
                    Truelabel[0, 0, len(key_categories) + v],
                )
            except ValueError:

                print(k, prediction[i, :, :, :, len(key_categories) + v])
                print(
                    "True positional value",
                    k,
                    Truelabel[0, 0, 0, len(key_categories) + v],
                )

    if plot:
        plt.show()


class PlotVolumeHistory(keras.callbacks.Callback):
    def __init__(
        self,
        Trainingmodel: models,
        X: np.ndarray,
        Y: np.ndarray,
        key_categories: dict,
        key_cord: dict,
        gridx: int,
        gridy: int,
        gridz: int,
        plot: bool = False,
        nboxes: int = 1,
    ):
        self.Trainingmodel = Trainingmodel
        self.X = X
        self.Y = Y
        self.plot = plot
        self.gridx = gridx
        self.gridy = gridy
        self.gridz = gridz
        self.nboxes = nboxes
        self.key_cord = key_cord
        self.key_categories = key_categories

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        if self.plot:
            self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self,epoch,logs={}):

        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get("loss"))
        self.val_losses.append(logs.get("val_loss"))
        self.acc.append(logs.get("accuracy"))
        self.val_acc.append(logs.get("val_accuracy"))

        self.i += 1
        if self.plot:
            f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

            ax1.set_yscale("log")
            ax1.plot(self.x, self.losses, label="loss")
            ax1.plot(self.x, self.val_losses, label="val_loss")
            ax1.legend()

            ax2.plot(self.x, self.acc, label="accuracy")
            ax2.plot(self.x, self.val_acc, label="val_accuracy")
            ax2.legend()
            plt.show()
            # clear_output(True)
        idx = random.randint(1, self.X.shape[0] - 1)
        PrintVolumepredict(
            idx,
            self.Trainingmodel,
            self.X,
            self.Y,
            self.key_categories,
            self.key_cord,
            plot=self.plot,
            nboxes=self.nboxes,
        )


def PrintVolumepredict(
    idx,
    model,
    data,
    Truelabel,
    key_categories,
    key_cord,
    plot=False,
    nboxes=1,
):

    Image = data[idx]
    Truelabel = Truelabel[idx]
    data = np.expand_dims(Image, 0)
    prediction = model.predict(data)
    cols = 5

    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, data.shape[1], figsize=(5 * cols, 5))
        fig.figsize = (20, 10)
    # The prediction vector is (1, categories + box_vector) dimensional, input data is (N Z H W T) C is 1 in our case
    for j in range(0, data.shape[-1]):

        img = Image[Image.shape[0] // 2, :, :, j]
        if plot:
            ax[j].imshow(img, cm.Spectral)
    for i in range(0, prediction.shape[0]):

        try:
            maxevent = np.argmax(
                prediction[i, :, :, :, : len(key_categories)], axis=-1
            )
            trueevent = np.argmax(
                Truelabel[0, 0, 0, : len(key_categories)], axis=-1
            )
        except ValueError:

            maxevent = np.argmax(
                prediction[i, :, :, :, :, : len(key_categories)], axis=-1
            )
            trueevent = np.argmax(
                Truelabel[0, 0, 0, 0, : len(key_categories)], axis=-1
            )
        for (k, v) in key_categories.items():
            if v == maxevent:
                maxlabel = k
            if v == trueevent:
                truelabel = k
        try:
            print(
                "Predicted cell:",
                maxlabel,
                "Probability:",
                prediction[i, 0, 0, 0, maxevent],
            )
            print("True Cell type:", truelabel)
        except ValueError:
            print(prediction[i, 0, 0, 0, :])
            print("True Cell type:", truelabel)
            # print( "Predicted cell:", maxlabel , "Probability:", prediction[i,0,0,0,maxevent])
            # print('True Cell type:', truelabel)

        if nboxes > 1:
            for b in range(1, nboxes - 1):
                try:
                    prediction[
                        i,
                        :,
                        :,
                        :,
                        len(key_categories) : len(key_categories)
                        + len(key_cord),
                    ] += prediction[
                        i,
                        :,
                        :,
                        :,
                        len(key_categories)
                        + b * len(key_cord) : len(key_categories)
                        + (b + 1) * len(key_cord),
                    ]
                    prediction[
                        i,
                        :,
                        :,
                        :,
                        len(key_categories) : len(key_categories)
                        + len(key_cord),
                    ] = prediction[
                        i,
                        :,
                        :,
                        :,
                        len(key_categories) : len(key_categories)
                        + len(key_cord),
                    ] / (
                        nboxes - 1
                    )
                except ValueError:
                    prediction[
                        i,
                        :,
                        :,
                        :,
                        len(key_categories) : len(key_categories)
                        + len(key_cord),
                    ] += prediction[
                        i,
                        :,
                        :,
                        :,
                        len(key_categories)
                        + b * len(key_cord) : len(key_categories)
                        + (b + 1) * len(key_cord),
                    ]
                    prediction[
                        i,
                        :,
                        :,
                        :,
                        len(key_categories) : len(key_categories)
                        + len(key_cord),
                    ] = prediction[
                        i,
                        :,
                        :,
                        :,
                        len(key_categories) : len(key_categories)
                        + len(key_cord),
                    ] / (
                        nboxes - 1
                    )
        for (k, v) in key_cord.items():
            try:
                print(k, prediction[i, :, :, :, len(key_categories) + v])
                print(
                    "True positional value",
                    k,
                    Truelabel[0, 0, 0, len(key_categories) + v],
                )
            except ValueError:

                print(k, prediction[i, :, :, :, :, len(key_categories) + v])
                print(
                    "True positional value",
                    k,
                    Truelabel[0, 0, 0, 0, len(key_categories) + v],
                )

    if plot:
        plt.show()


class PlotStaticHistory(keras.callbacks.Callback):
    def __init__(
        self,
        Trainingmodel: models,
        X: np.ndarray,
        Y: np.ndarray,
        key_categories: dict,
        key_cord: dict,
        gridx: int,
        gridy: int,
        plot: bool = False,
        nboxes: int = 1,
        class_only: bool = False,
    ):
        self.Trainingmodel = Trainingmodel
        self.X = X
        self.Y = Y
        self.gridx = gridx
        self.gridy = gridy
        self.plot = plot
        self.nboxes = nboxes
        self.key_cord = key_cord
        self.key_categories = key_categories
        self.class_only = class_only

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        if self.plot:
            self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, logs={}):
        print(logs)
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get("loss"))
        self.val_losses.append(logs.get("val_loss"))
        self.acc.append(logs.get("acc"))
        self.val_acc.append(logs.get("val_acc"))

        self.i += 1
        if self.plot:
            f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

            ax1.set_yscale("log")
            ax1.plot(self.x, self.losses, label="loss")
            ax1.plot(self.x, self.val_losses, label="val_loss")
            ax1.legend()
            ax2.plot(self.x, self.acc, label="acc")
            ax2.plot(self.x, self.val_acc, label="val_acc")
            ax2.legend()
            plt.show()
            # clear_output(True)
        idx = random.randint(1, self.X.shape[0] - 1)
        PrintStaticpredict(
            idx,
            self.Trainingmodel,
            self.X,
            self.Y,
            self.key_categories,
            self.key_cord,
            plot=self.plot,
            nboxes=self.nboxes,
            class_only=self.class_only,
        )


def PrintStaticpredict(
    idx,
    model: models,
    data: list,
    Truelabel: list,
    key_categories: dict,
    key_cord: dict,
    plot: bool = False,
    nboxes: int = 1,
    class_only: bool = False,
):

    Image = data[idx]
    Truelabel = Truelabel[idx]
    data = np.expand_dims(Image, 0)
    prediction = model.predict(data)

    # The prediction vector is (1, categories + box_vector) dimensional, input data is (N H W C) C is 1 in our case

    img = Image[:, :, 0]
    if plot:
        plt.imshow(img, cm.Spectral)
    for i in range(0, prediction.shape[0]):
        maxevent = np.argmax(
            prediction[i, :, :, : len(key_categories)], axis=-1
        )
        trueevent = np.argmax(Truelabel[0, 0, : len(key_categories)], axis=-1)
        for (k, v) in key_categories.items():
            if v == maxevent:
                maxlabel = k
            if v == trueevent:
                truelabel = k

        print(
            "Predicted cell:",
            maxlabel,
            "Probability:",
            prediction[i, 0, 0, maxevent],
        )
        print("True Cell type:", truelabel)
        if nboxes > 1:
            for b in range(1, nboxes - 1):
                prediction[
                    i,
                    :,
                    :,
                    len(key_categories) : len(key_categories) + len(key_cord),
                ] += prediction[
                    i,
                    :,
                    :,
                    len(key_categories)
                    + b * len(key_cord) : len(key_categories)
                    + (b + 1) * len(key_cord),
                ]
            prediction[
                i,
                :,
                :,
                len(key_categories) : len(key_categories) + len(key_cord),
            ] = prediction[
                i,
                :,
                :,
                len(key_categories) : len(key_categories) + len(key_cord),
            ] / (
                nboxes - 1
            )
        if class_only is False:
            for (k, v) in key_cord.items():

                print(k, prediction[i, :, :, len(key_categories) + v])
                print(
                    "True positional value",
                    k,
                    Truelabel[0, 0, len(key_categories) + v],
                )

    if plot:
        plt.show()
