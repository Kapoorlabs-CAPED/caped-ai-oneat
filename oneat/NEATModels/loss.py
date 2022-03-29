#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 16:05:34 2021
@author: vkapoor
"""

import tensorflow as tf
import numpy as np
from keras import backend as K

lambdaobject = 1
lambdanoobject = 1
lambdacoord = 1
lambdaclass = 1
lambdaangle = 1

def get_event_grid(grid_h, grid_w, grid_t, boxes):
    
    event_grid = np.array([ [[float(y),float(x),float(t)]]*boxes  for y in range(grid_h) for x in range(grid_w) for t in range(grid_t)])
    
    return event_grid


def extract_ground_event_truth(y_true, categories, grid_h, grid_w, grid_t, nboxes, box_vector,yolo_v0,yolo_v1,yolo_v2):
    
    true_box_class = y_true[...,0:categories]
    
    true_nboxes = K.reshape(y_true[...,categories:], (-1, grid_h * grid_w * grid_t, nboxes, box_vector))
    
   

    true_box_xyt = true_nboxes[...,0:3]

    true_box_wh =  true_nboxes[...,3:5] 
    
    if yolo_v0:
        true_box_conf = 1
        true_box_angle = 2
        
    
    else:
            if yolo_v1 or yolo_v2:    
               true_box_conf = true_nboxes[...,5]
               true_box_angle = 2
            if yolo_v2:
                true_box_angle = true_nboxes[...,6]
                
    return true_box_class, true_box_xyt, true_box_wh, true_box_conf, true_box_angle 
               
    
def extract_ground_event_pred(y_pred, categories, grid_h, grid_w, grid_t, event_grid, nboxes, box_vector, yolo_v0,yolo_v1,yolo_v2):
    
    pred_box_class = y_pred[...,0:categories]
    
    pred_nboxes = K.reshape(y_pred[...,categories:], (-1, grid_h * grid_w * grid_t, nboxes, box_vector))
    
    pred_box_xyt = pred_nboxes[...,0:3] + event_grid
    pred_box_wh = pred_nboxes[...,3:5]
        
    if yolo_v0:
        
        pred_box_conf = 1
        pred_box_angle = 2
        
    
    else:
            if yolo_v1 or yolo_v2:    
               pred_box_conf = pred_nboxes[...,5]
               pred_box_angle = 2
            if yolo_v2:
                pred_box_angle = pred_nboxes[...,6]
                
    return pred_box_class, pred_box_xyt, pred_box_wh, pred_box_conf, pred_box_angle 
            
    
def get_cell_grid(grid_h, grid_w, boxes):
    
    cell_grid = np.array([ [[float(x),float(y)]]*boxes   for y in range(grid_h) for x in range(grid_w)])
    
    return cell_grid

    
def extract_ground_cell_pred(y_pred, categories, grid_h, grid_w, cell_grid, nboxes, box_vector, yolo_v0):

        pred_box_class = y_pred[...,0:categories]
        
        pred_nboxes = K.reshape(y_pred[...,categories:], (-1, grid_h * grid_w, nboxes, box_vector))
    
        pred_box_xy = pred_nboxes[...,0:2] + K.variable(cell_grid)
    
        pred_box_wh = pred_nboxes[...,2:4]
        
        if yolo_v0:
            
                pred_box_conf = 1
         
        else: 
        
                pred_box_conf = pred_nboxes[...,4]
        
        return pred_box_class, pred_box_xy, pred_box_wh, pred_box_conf

def extract_ground_cell_truth(y_truth, categories, grid_h, grid_w, nboxes, box_vector, yolo_v0):

        true_box_class = y_truth[...,0:categories]
        
        true_nboxes = K.reshape(y_truth[...,categories:], (-1, grid_h * grid_w, nboxes, box_vector))
    
        true_box_xy = true_nboxes[...,0:2]
    
        true_box_wh = true_nboxes[...,2:4]
        
        if yolo_v0:
             
               true_box_conf = 1
         
        else: 
        
               true_box_conf = true_nboxes[...,4]
        
        return true_box_class, true_box_xy, true_box_wh, true_box_conf


def extract_ground_cell_pred_segfree(y_pred, categories, grid_h, grid_w, cell_grid, nboxes, box_vector):

        pred_box_class = y_pred[...,0:categories]
        
        pred_nboxes = K.reshape(y_pred[...,categories:], (-1, grid_h * grid_w, nboxes, box_vector))
    
        pred_box_xy = pred_nboxes[...,0:2] + cell_grid
    
        
        return pred_box_class, pred_box_xy

def extract_ground_cell_truth_foc(y_truth, categories):

        true_box_class = y_truth[...,0:categories]
        
        return true_box_class

def extract_ground_cell_pred_foc(y_pred, categories):

        pred_box_class = y_pred[...,0:categories]
       
    
        
        return pred_box_class

def extract_ground_cell_truth_segfree(y_truth, categories, grid_h, grid_w, nboxes, box_vector):

        true_box_class = y_truth[...,0:categories]
        
        true_nboxes = K.reshape(y_truth[...,categories:], (-1, grid_h * grid_w, nboxes, box_vector))
    
        true_box_xy = true_nboxes[...,0:2]
    
        
        return true_box_class, true_box_xy


        
def compute_conf_loss(pred_box_wh, true_box_wh, pred_box_xy,true_box_xy,true_box_conf,pred_box_conf):
    
# compute the intersection of all boxes at once (the IOU)
        intersect_wh = K.maximum(K.zeros_like(pred_box_wh), (pred_box_wh + true_box_wh)/2 - K.square(pred_box_xy[...,0:1]- true_box_xy[...,0:1]) )
        intersect_area = intersect_wh[...,0] * intersect_wh[...,1] 
        true_area = true_box_wh[...,0] * true_box_wh[...,1] 
        pred_area = pred_box_wh[...,0] * pred_box_wh[...,1] 
        union_area = pred_area + true_area - intersect_area
        iou = tf.truediv(intersect_area , union_area)
  
        best_ious = K.max(iou, axis= -1)
        loss_conf = K.sum(K.square(true_box_conf*best_ious - pred_box_conf), axis=-1)

        loss_conf = loss_conf * lambdaobject

        return loss_conf 

def compute_conf_loss_static(pred_box_wh, true_box_wh, pred_box_xy,true_box_xy,true_box_conf,pred_box_conf):
    
# compute the intersection of all boxes at once (the IOU)
        intersect_wh = K.maximum(K.zeros_like(pred_box_wh), (pred_box_wh + true_box_wh)/2 - K.square(pred_box_xy- true_box_xy) )
        intersect_area = intersect_wh[...,0] * intersect_wh[...,1] 
        true_area = true_box_wh[...,0] * true_box_wh[...,1] 
        pred_area = pred_box_wh[...,0] * pred_box_wh[...,1] 
        union_area = pred_area + true_area - intersect_area
        iou = intersect_area / union_area
  
        loss_conf = K.sum(K.square(true_box_conf*iou - pred_box_conf), axis=-1)

        loss_conf = loss_conf * lambdaobject

        return loss_conf        

def calc_loss_xywh(true_box_conf, true_box_xy, pred_box_xy, true_box_wh, pred_box_wh):

    
    loss_xy      = K.sum(K.sum(K.square(true_box_xy - pred_box_xy), axis = -1), axis = -1)
    loss_wh      = K.sum(K.sum(K.square(K.sqrt(true_box_wh) - K.sqrt(pred_box_wh)), axis=-1), axis=-1)
    loss_xywh = (loss_xy + loss_wh)
    loss_xywh = lambdacoord * loss_xywh
    return loss_xywh

def calc_loss_xy(true_box_xy, pred_box_xy):

    
    loss_xy      = K.sum(K.sum(K.square(true_box_xy - pred_box_xy), axis = -1), axis = -1)
    loss_xy = lambdacoord * loss_xy
    return loss_xy

def calc_loss_angle(true_box_angle, pred_box_angle):

    
    loss_angle      = K.sum(K.sum(K.square(true_box_angle - pred_box_angle), axis = -1), axis = -1)
    loss_angle = lambdaangle * loss_angle
    return loss_angle
def calc_loss_class(true_box_class, pred_box_class, entropy):

    
        if entropy == 'binary':
            loss_class = K.mean(K.binary_crossentropy(true_box_class, pred_box_class), axis=-1)
        if entropy == 'notbinary':
            loss_class   = K.mean(K.categorical_crossentropy(true_box_class, pred_box_class), axis=-1)

        loss_class   = loss_class * lambdaclass 

        return loss_class



def dynamic_yolo_loss(categories, grid_h, grid_w, grid_t, nboxes, box_vector, entropy, yolo_v0, yolo_v1, yolo_v2):
    def loss(y_true, y_pred):    
        event_grid = get_event_grid(grid_h, grid_w, grid_t, nboxes)
        true_box_class, true_box_xyt, true_box_wh, true_box_conf, true_box_angle = extract_ground_event_truth(y_true, categories, grid_h, grid_w,grid_t, nboxes, box_vector, yolo_v0, yolo_v1, yolo_v2)
        pred_box_class, pred_box_xyt, pred_box_wh, pred_box_conf, pred_box_angle = extract_ground_event_pred(y_pred, categories, grid_h, grid_w,grid_t, event_grid, nboxes, box_vector, yolo_v0, yolo_v1, yolo_v2)

        loss_xywht = calc_loss_xywh(true_box_conf, true_box_xyt, pred_box_xyt, true_box_wh, pred_box_wh)

        loss_class   = calc_loss_class(true_box_class, pred_box_class, entropy)
        
        if yolo_v0:
                    combinedloss = (loss_xywht + loss_class)

        if yolo_v1:
                    loss_conf = compute_conf_loss(pred_box_wh, true_box_wh, pred_box_xyt,true_box_xyt,true_box_conf,pred_box_conf)
                    # Adding it all up   
                    combinedloss = (loss_xywht + loss_conf + loss_class)
        if yolo_v2:
                     
                    loss_conf = compute_conf_loss(pred_box_wh, true_box_wh, pred_box_xyt,true_box_xyt,true_box_conf,pred_box_conf) 
                    loss_angle = calc_loss_angle( true_box_angle, pred_box_angle)                             
                    combinedloss = (loss_xywht + loss_conf + loss_class + loss_angle) 


        return combinedloss 
        
    return loss 


def static_yolo_loss(categories, grid_h, grid_w, nboxes, box_vector, entropy, yolo_v0):
    
    def loss(y_true, y_pred):    

        cell_grid = get_cell_grid(grid_h, grid_w, nboxes)
        true_box_class, true_box_xy, true_box_wh, true_box_conf = extract_ground_cell_truth(y_true, categories, grid_h, grid_w, nboxes, box_vector, yolo_v0)
        pred_box_class, pred_box_xy, pred_box_wh, pred_box_conf = extract_ground_cell_pred(y_pred, categories, grid_h, grid_w, cell_grid, nboxes, box_vector, yolo_v0)

        loss_xywh = calc_loss_xywh(true_box_conf, true_box_xy, pred_box_xy, true_box_wh, pred_box_wh)

        loss_class   = calc_loss_class(true_box_class, pred_box_class, entropy)
        
        if yolo_v0:
            
                    combinedloss = (loss_xywh + loss_class)

        else:
                    loss_conf = compute_conf_loss_static(pred_box_wh, true_box_wh, pred_box_xy,true_box_xy,true_box_conf,pred_box_conf)
                    # Adding it all up   
                    combinedloss = (loss_xywh + loss_conf + loss_class)



        return combinedloss 
        
    return loss    

def static_yolo_loss_segfree(categories, grid_h, grid_w, nboxes, box_vector, entropy, yolo_v0):
    
    def loss(y_true, y_pred):    

        cell_grid = get_cell_grid(grid_h, grid_w, nboxes)
        true_box_class, true_box_xy = extract_ground_cell_truth_segfree(y_true, categories, grid_h, grid_w, nboxes, box_vector, yolo_v0)
        pred_box_class, pred_box_xy = extract_ground_cell_pred_segfree(y_pred, categories, grid_h, grid_w, cell_grid, nboxes, box_vector, yolo_v0)

        loss_xy = calc_loss_xy(true_box_xy, pred_box_xy)

        loss_class   = calc_loss_class(true_box_class, pred_box_class, entropy)
        
                    # Adding it all up   
        combinedloss = (loss_xy + loss_class)



        return combinedloss 
        
    return loss  
