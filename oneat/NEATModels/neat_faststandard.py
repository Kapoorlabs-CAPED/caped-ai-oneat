#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 13:49:35 2021

@author: vkapoor
"""

from .neat_goldstandard import NEATDynamic

class NEATSynamic(NEATDynamic):
    

    def __init__(self, config, model_dir, model_name, catconfig = None, cordconfig = None):

        super().__init__(config = config, model_dir = model_dir, model_name = model_name, catconfig = catconfig, cordconfig = cordconfig)
        
    def predict_synamic(self,imagename, savedir, n_tiles = (1,1), overlap_percent = 0.8, event_threshold = 0.5, event_confidence = 0.5, iou_threshold = 0.1, fidelity = 5,
     downsamplefactor = 1, maskfilter = 10,start_project_mid = 4, end_project_mid = 4, maskmodel = None, maskdir = None):

        self.predict(imagename,savedir,n_tiles = n_tiles, overlap_percent = overlap_percent, event_threshold = event_threshold, event_confidence = event_confidence, iou_threshold = iou_threshold, 
        fidelity = fidelity, start_project_mid = start_project_mid, end_project_mid = end_project_mid, downsamplefactor = downsamplefactor, maskfilter = maskfilter,  markers = None, marker_tree = None,
         remove_markers = None, maskmodel = maskmodel, maskdir = maskdir )