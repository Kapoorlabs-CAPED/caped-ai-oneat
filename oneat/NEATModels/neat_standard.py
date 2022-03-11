from ..NEATUtils import plotters
import numpy as np
from ..NEATUtils import helpers
from ..NEATUtils.helpers import get_nearest, save_json, load_json, yoloprediction, normalizeFloatZeroOne, GenerateMarkers, \
    DensityCounter, MakeTrees, nonfcn_yoloprediction, fastnms, averagenms, DownsampleData, save_dynamic_csv, dynamic_nms
from keras import callbacks
import os
import math
import tensorflow as tf
from tqdm import tqdm
from ..NEATModels import nets
from ..NEATModels.nets import Concat
from ..NEATModels.loss import dynamic_yolo_loss
from keras import backend as K
# from IPython.display import clear_output
from keras import optimizers
from pathlib import Path
from keras.models import load_model
from tifffile import imread, imwrite
import csv
import napari
import glob
from scipy import spatial
import itertools
from napari.qt.threading import thread_worker
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox, QPushButton, QSlider
import h5py
import cv2
import imageio

Boxname = 'ImageIDBox'
EventBoxname = 'EventIDBox'
from .neat_goldstandard import NEATDynamic

class NEATSDynamic(NEATDynamic):
   

    def __init__(self, config, model_dir, model_name, catconfig=None, cordconfig=None):

                super().__init__(config = config, model_dir = model_dir, model_name = model_name, catconfig = catconfig, cordconfig = cordconfig)

    


    def predict_standard(self, imagename, markers, marker_tree, savedir, n_tiles=(1, 1), overlap_percent=0.8,
                event_threshold=0.5, iou_threshold=0.1, fidelity = 5, downsamplefactor = 1, watershed = None, maskimage = None, maskfilter = 10):

        self.predict(imagename,savedir,n_tiles = n_tiles, overlap_percent = overlap_percent, event_threshold = event_threshold, iou_threshold = iou_threshold, 
        fidelity = fidelity, downsamplefactor = downsamplefactor,  maskfilter = maskfilter, markers = markers, marker_tree = marker_tree, watershed = watershed, maskimage = maskimage,
         remove_markers = None )

def CreateVolume(patch, imaget, timepoint):
    starttime = timepoint
    endtime = timepoint + imaget
    smallimg = patch[starttime:endtime, :]

    return smallimg

