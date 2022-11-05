import argparse

import numpy as np


class static_config(argparse.Namespace):
    def __init__(
        self,
        npz_directory=None,
        npz_name=None,
        npz_val_name=None,
        key_categories=None,
        key_cord=None,
        stage_number=3,
        gridx=1,
        gridy=1,
        nboxes=1,
        depth=29,
        start_kernel=3,
        mid_kernel=3,
        startfilter=32,
        show=True,
        imagex=64,
        imagey=64,
        epochs=100,
        learning_rate=1.0e-4,
        batch_size=10,
        model_name="NEATModel",
        yolo_v0=True,
        multievent=True,
        **kwargs
    ):

        self.npz_directory = npz_directory
        self.npz_name = npz_name
        self.npz_val_name = npz_val_name
        self.key_categories = key_categories
        self.key_cord = key_cord
        self.depth = depth
        
        self.start_kernel = start_kernel
        self.mid_kernel = mid_kernel
        self.startfilter = startfilter
        self.gridx = gridx
        self.gridy = gridy
        self.nboxes = nboxes
        self.show = show
        self.imagex = imagex
        self.imagey = imagey
        self.epochs = epochs
        self.stage_number = stage_number
        self.last_conv_factor = 2 ** (self.stage_number - 1)
        self.yolo_v0 = yolo_v0
        self.categories = len(self.key_categories)
        self.box_vector = len(self.key_cord)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model_name = model_name
        self.multievent = multievent

        self.is_valid()

    def to_json(self):

        config = {
            "npz_directory": self.npz_directory,
            "npz_name": self.npz_name,
            "npz_val_name": self.npz_val_name,
            "depth": self.depth,
            "depth_dense_0": self.depth_dense_0,
            "depth_dense_1": self.depth_dense_1,
            "depth_dense_2": self.depth_dense_2,
            "depth_dense_3": self.depth_dense_3,
            "start_kernel": self.start_kernel,
            "mid_kernel": self.mid_kernel,
            "startfilter": self.startfilter,
            "gridx": self.gridx,
            "gridy": self.gridy,
            "nboxes": self.nboxes,
            "imagex": self.imagex,
            "imagey": self.imagey,
            "epochs": self.epochs,
            "categories": self.categories,
            "box_vector": self.box_vector,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "model_name": self.model_name,
            "multievent": self.multievent,
            "yolo_v0": self.yolo_v0,
            "show": self.show,
            "stage_number": self.stage_number,
            "last_conv_factor": self.last_conv_factor,
        }
        for (k, v) in self.key_categories.items():
            config[k] = v

        for (k, v) in self.key_cord.items():
            config[k] = v

        return config

    def is_valid(self, return_invalid=False):
        """Check if configuration is valid.
        Returns
        -------
        bool
        Flag that indicates whether the current configuration values are valid.
        """

        def _is_int(v, low=None, high=None):
            return (
                isinstance(v, int)
                and (True if low is None else low <= v)
                and (True if high is None else v <= high)
            )

        ok = {}
        ok["npz_directory"] = isinstance(self.npz_directory, str)
        ok["npz_name"] = isinstance(self.npz_name, str)
        ok["npz_val_name"] = isinstance(self.npz_val_name, str)
        ok["yolo_v0"] = isinstance(self.yolo_v0, bool)
        ok["depth"] = _is_int(self.depth, 1)
       
        ok["start_kernel"] = _is_int(self.start_kernel, 1)
        ok["mid_kernel"] = _is_int(self.mid_kernel, 1)
        ok["startfilter"] = _is_int(self.startfilter, 1)
        ok["stage_number"] = _is_int(self.stage_number, 1)
        ok["last_conv_factor"] = _is_int(self.last_conv_factor, 1)
        ok["epochs"] = _is_int(self.epochs, 1)
        ok["nboxes"] = _is_int(self.nboxes, 1)
        ok["gridx"] = _is_int(self.gridx, 1)
        ok["gridy"] = _is_int(self.gridy, 1)
        ok["imagex"] = _is_int(self.imagex, 1)
        ok["imagey"] = _is_int(self.imagey, 1)
        ok["show"] = isinstance(self.show, bool)
        ok["learning_rate"] = (
            np.isscalar(self.learning_rate) and self.learning_rate > 0
        )
        ok["multievent"] = isinstance(self.multievent, bool)
        ok["categories"] = _is_int(len(self.key_categories), 1)
        ok["box_vector"] = _is_int(self.box_vector, 1)

        if return_invalid:
            return all(ok.values()), tuple(k for (k, v) in ok.items() if not v)
        else:
            return all(ok.values())
