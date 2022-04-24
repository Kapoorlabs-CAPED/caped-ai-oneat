#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 13:49:35 2021

@author: vkapoor
"""

from .neat_goldstandard import NEATDynamic
from oneat.pretrained import get_registered_models, get_model_details, get_model_instance
import sys
class NEATSynamic(NEATDynamic):
    

    def __init__(self, config, model_dir, model_name, catconfig = None, cordconfig = None):

        super().__init__(config = config, model_dir = model_dir, model_name = model_name, catconfig = catconfig, cordconfig = cordconfig)
    @classmethod   
    def local_from_pretrained(cls, target, name_or_alias=None):
           try:
               print('class', cls, 'name' , name_or_alias )
               get_model_details(cls, name_or_alias, verbose=True)
               return get_model_instance(cls,  name_or_alias, target)
           except ValueError:
               if name_or_alias is not None:
                   print("Could not find model with name or alias '%s'" % (name_or_alias), file=sys.stderr)
                   sys.stderr.flush()
               get_registered_models(cls, verbose=True)    

    def predict_synamic(self,imagename, savedir, n_tiles = (1,1), overlap_percent = 0.8, event_threshold = 0.5, event_confidence = 0.5, iou_threshold = 0.1, fidelity = 5,
     downsamplefactor = 1, erosion_iterations = 10,start_project_mid = 4, end_project_mid = 4, maskmodel = None, segdir = None, normalize = True, center_oneat = True, nms_function = 'iou'):

        self.predict(imagename,savedir,n_tiles = n_tiles, overlap_percent = overlap_percent, event_threshold = event_threshold, event_confidence = event_confidence, iou_threshold = iou_threshold, 
        fidelity = fidelity, start_project_mid = start_project_mid, end_project_mid = end_project_mid, downsamplefactor = downsamplefactor, erosion_iterations = erosion_iterations, marker_tree = None,
         remove_markers = None,  normalize = normalize, center_oneat = center_oneat, nms_function = nms_function )

