#!/usr/bin/env python3
"""
Created on Tue Jul  7 15:25:10 2020

@author: aimachine
"""
import argparse

import numpy as np


class lstm_config(argparse.Namespace):
    def __init__(
        self,
        npz_directory=None,
        npz_name=None,
        npz_val_name=None,
        key_categories=None,
        key_cord=None,
        stage_number=3,
        last_conv_factor=4,
        imagex=128,
        imagey=128,
        size_tminus=3,
        size_tplus=0,
        nboxes=1,
        depth=29,
        start_kernel=3,
        mid_kernel=3,
        startfilter=48,
        lstm_hidden_unit=16,
        epochs=100,
        learning_rate=1.0e-4,
        batch_size=10,
        model_name="NEATModel",
        yolo_v0=False,
        yolo_v1=True,
        yolo_v2=False,
        multievent=False,
        pure_lstm=False,
        show=True,
        **kwargs
    ):

        self.npz_directory = npz_directory
        self.npz_name = npz_name
        self.npz_val_name = npz_val_name
        self.key_categories = key_categories
        self.key_cord = key_cord
        self.yolo_v0 = yolo_v0
        self.yolo_v1 = yolo_v1
        self.yolo_v2 = yolo_v2
        self.nboxes = nboxes
        self.multievent = multievent
        self.categories = len(self.key_categories)
        self.box_vector = len(self.key_cord)
        self.depth = depth
        self.start_kernel = start_kernel
        self.mid_kernel = mid_kernel
        self.startfilter = startfilter
        self.lstm_hidden_unit = lstm_hidden_unit
        self.epochs = epochs
        self.stride = last_conv_factor
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model_name = model_name
        self.show = show
        self.imagex = imagex
        self.imagey = imagey
        self.stage_number = stage_number
        self.last_conv_factor = last_conv_factor
        self.size_tminus = size_tminus
        self.size_tplus = size_tplus
        self.pure_lstm = pure_lstm
        self.is_valid()

    def to_json(self):

        config = {
            "npz_directory": self.npz_directory,
            "npz_name": self.npz_name,
            "npz_val_name": self.npz_val_name,
            "model_name": self.model_name,
            "multievent": self.multievent,
            "yolo_v0": self.yolo_v0,
            "yolo_v1": self.yolo_v1,
            "yolo_v2": self.yolo_v2,
            "nboxes": self.nboxes,
            "imagex": self.imagex,
            "imagey": self.imagey,
            "size_tminus": self.size_tminus,
            "size_tplus": self.size_tplus,
            "stride": self.stride,
            "depth": self.depth,
           
            "categories": self.categories,
            "box_vector": self.box_vector,
            "start_kernel": self.start_kernel,
            "mid_kernel": self.mid_kernel,
            "startfilter": self.startfilter,
            "lstm_hidden_unit": self.lstm_hidden_unit,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "show": self.show,
            "stage_number": self.stage_number,
            "last_conv_factor": self.last_conv_factor,
            "pure_lstm": self.pure_lstm,
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
        ok["pure_lstm"] = isinstance(self.pure_lstm, bool)
        ok["yolo_v0"] = isinstance(self.yolo_v0, bool)
        ok["yolo_v1"] = isinstance(self.yolo_v1, bool)
        ok["yolo_v2"] = isinstance(self.yolo_v2, bool)
        ok["depth"] = _is_int(self.depth, 1)
        
        ok["stride"] = _is_int(self.stride, 1)
        ok["start_kernel"] = _is_int(self.start_kernel, 1)
        ok["mid_kernel"] = _is_int(self.mid_kernel, 1)
        ok["startfilter"] = _is_int(self.startfilter, 1)
        ok["stage_number"] = _is_int(self.stage_number, 1)
        ok["last_conv_factor"] = _is_int(self.last_conv_factor, 1)
        ok["epochs"] = _is_int(self.epochs, 1)
        ok["nboxes"] = _is_int(self.nboxes, 1)

        ok["imagex"] = _is_int(self.imagex, 1)
        ok["imagey"] = _is_int(self.imagey, 1)
        ok["size_tminus"] = _is_int(self.size_tminus, 1)
        ok["size_tplus"] = _is_int(self.size_tplus, 1)

        ok["learning_rate"] = (
            np.isscalar(self.learning_rate) and self.learning_rate > 0
        )
        ok["multievent"] = isinstance(self.multievent, bool)
        ok["show"] = isinstance(self.show, bool)
        ok["categories"] = _is_int(len(self.key_categories), 1)
        ok["box_vector"] = _is_int(self.box_vector, 1)

        if return_invalid:
            return all(ok.values()), tuple(k for (k, v) in ok.items() if not v)
        else:
            return all(ok.values())


class volume_config(argparse.Namespace):
    def __init__(
        self,
        npz_directory=None,
        npz_name=None,
        npz_val_name=None,
        pure_lstm=False,
        key_categories=None,
        key_cord=None,
        stage_number=3,
        last_conv_factor=4,
        imagex=64,
        imagey=64,
        imagez=4,
        size_tminus=1,
        size_tplus=1,
        nboxes=1,
        depth= {'depth_0': 12, 'depth_1': 24, 'depth_2': 16},
        reduction = 0.5,
        weight_decay=1e-4,
        start_kernel=7,
        mid_kernel=3,
        startfilter=48,
        epochs=100,
        learning_rate=1.0e-4,
        batch_size=10,
        model_name="NEATModel",
        yolo_v0=False,
        yolo_v1=True,
        yolo_v2=False,
        multievent=False,
        show=True,
        **kwargs
    ):

        self.npz_directory = npz_directory
        self.npz_name = npz_name
        self.npz_val_name = npz_val_name
        self.key_categories = key_categories
        self.key_cord = key_cord
        self.yolo_v0 = yolo_v0
        self.yolo_v1 = yolo_v1
        self.yolo_v2 = yolo_v2
        self.nboxes = nboxes
        self.multievent = multievent
        self.categories = len(self.key_categories)
        self.box_vector = len(self.key_cord)
        self.depth = depth
        self.reduction = reduction
        self.weight_decay=weight_decay
        
        self.start_kernel = start_kernel
        self.mid_kernel = mid_kernel
        self.startfilter = startfilter
        self.epochs = epochs
        self.stride = last_conv_factor
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model_name = model_name
        self.show = show
        self.imagex = imagex
        self.imagey = imagey
        self.imagez = imagez
        self.stage_number = stage_number
        self.last_conv_factor = last_conv_factor
        self.size_tminus = size_tminus
        self.size_tplus = size_tplus
        self.pure_lstm = pure_lstm
        self.is_valid()

    def to_json(self):

        config = {
            "npz_directory": self.npz_directory,
            "npz_name": self.npz_name,
            "npz_val_name": self.npz_val_name,
            "model_name": self.model_name,
            "multievent": self.multievent,
            "yolo_v0": self.yolo_v0,
            "yolo_v1": self.yolo_v1,
            "yolo_v2": self.yolo_v2,
            "nboxes": self.nboxes,
            "imagex": self.imagex,
            "imagey": self.imagey,
            "imagez": self.imagez,
            "size_tminus": self.size_tminus,
            "size_tplus": self.size_tplus,
            "stride": self.stride,
            "depth": self.depth,
            "reduction": self.reduction,
            "weight_decay": self.weight_decay,
            "categories": self.categories,
            "box_vector": self.box_vector,
            "start_kernel": self.start_kernel,
            "mid_kernel": self.mid_kernel,
            "startfilter": self.startfilter,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "show": self.show,
            "stage_number": self.stage_number,
            "last_conv_factor": self.last_conv_factor,
            "pure_lstm": self.pure_lstm,
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
        ok["yolo_v1"] = isinstance(self.yolo_v1, bool)
        ok["yolo_v2"] = isinstance(self.yolo_v2, bool)
        ok["pure_lstm"] = isinstance(self.pure_lstm, bool)
        ok["depth"] = isinstance(self.pure_lstm, dict)
        ok["reduction"] = isinstance(self.reduction, float)
        ok["weight_decay"]=isinstance(self.weight_decay, float)
        ok["stride"] = _is_int(self.stride, 1)
        ok["start_kernel"] = _is_int(self.start_kernel, 1)
        ok["mid_kernel"] = _is_int(self.mid_kernel, 1)
        ok["startfilter"] = _is_int(self.startfilter, 1)
        ok["stage_number"] = _is_int(self.stage_number, 1)
        ok["last_conv_factor"] = _is_int(self.last_conv_factor, 1)
        ok["epochs"] = _is_int(self.epochs, 1)
        ok["nboxes"] = _is_int(self.nboxes, 1)

        ok["imagex"] = _is_int(self.imagex, 1)
        ok["imagey"] = _is_int(self.imagey, 1)
        ok["imagez"] = _is_int(self.imagez, 1)
        ok["size_tminus"] = _is_int(self.size_tminus, 1)
        ok["size_tplus"] = _is_int(self.size_tplus, 1)

        ok["learning_rate"] = (
            np.isscalar(self.learning_rate) and self.learning_rate > 0
        )
        ok["multievent"] = isinstance(self.multievent, bool)
        ok["show"] = isinstance(self.show, bool)
        ok["categories"] = _is_int(len(self.key_categories), 1)
        ok["box_vector"] = _is_int(self.box_vector, 1)

        if return_invalid:
            return all(ok.values()), tuple(k for (k, v) in ok.items() if not v)
        else:
            return all(ok.values())
