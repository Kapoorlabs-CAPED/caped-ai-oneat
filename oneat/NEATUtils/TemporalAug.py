#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed August 17 12:57:26 2022

@author: kapoorlab
"""

import numpy as np

from albumentations import transforms    
from scipy import ndimage
import pandas as pd
from photutils.datasets import make_noise_image
class TemporalAug(object):



    """
    Augmentation creator for a TYX shape input image and labelimages
    Note:
        Only one type of augmentation can be applied for one creator.
    """
    def __init__(self,
                 rotate_angle=None, 
                 mean = None,
                 sigma = None,
                 distribution = None,
                 brightness_limit=None,
                 contrast_limit=None,
                 brightness_by_max=True,
                 always_apply=False,
                 prob_bright_contrast=0.5,
                 multiplier=None,
                 ):
        """
        Arguments:
         
       
        rotate_angle: int or 'random'
                Angle by which image is rotated using the affine transformation matrix.
        mean: float
                Mean of the distribution used for adding noise to the image
        sigma  : float
                Standard Deviation of the distribution used for adding noise to the image       
        distribution : "Gaussian", "Poisson", "Both"
                The tuple or a single string name for the distribution to add noise          
        brightness_limit ((float, float) or float): factor range for changing brightness.
            If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).
        contrast_limit ((float, float) or float): factor range for changing contrast.
            If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).
        brightness_by_max (Boolean): If True adjust contrast by image dtype maximum,
            else adjust contrast by image mean.
        prob_bright_contrast (float): probability of applying the transform. Default: 0.5.   
        multiplier (float): multiplier tuple for applying multiplicative image noise    
        """
       
        self.rotate_angle = rotate_angle
        self.mean = mean 
        self.sigma = sigma
        self.distribution = distribution
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.brightness_by_max = brightness_by_max 
        self.always_apply = always_apply
        self.prob_bright_contrast = prob_bright_contrast
        self.multiplier = multiplier
    
    def build(self,
              image=None,
              labelimage=None,
              labelcsv = None
              ):
        """
        Arguments:
        build augmentor to augment input image according to initialization
        image : array
            Input image to be augmented.
            The shape of input image should have 3 dimension( t, y, x).
        labelimage : Integer labelimage images
            The shape of this labelimages should match the shape of image (t, y, x).
        labelcsv : oneat compatiable csv file of class and event lcoations.
            The oneat training datamaker writes T,Y,X of the selected event locations    
  
        Return:
            augmentor
        """
        if image.ndim != 3:
            raise ValueError('Input image should have 3 dimensions.')
       
        if image.ndim != labelimage.ndim:
                raise ValueError('Input image and labelimage size do not much.')

        self.image = image
        self.labelimage = labelimage
        self.labelcsv = labelcsv
      
        self.image_dim = image.ndim
        self.image_shape = image.shape
       

        parse_dict = {}
        callback_geometric = None
        callback_intensity = None
        # rotate
        if  (self.rotate_angle is not None):
            callback_geometric = self._rotate_image

            if self.rotate_angle == 'random':
                parse_dict['rotate_angle'] = int(np.random.uniform(-180, 180))
            elif type(self.rotate_angle) == int:
                parse_dict['rotate_angle'] = np.radians(self.rotate_angle)
            else:
                raise ValueError('Rotate angle should be int or random')

        # add additive noise
        if (self.distribution is not None):

            callback_intensity = self._noise_image  
            if self.distribution == 'Gaussian':
                parse_dict['distribution'] = 'Gaussian'
            if self.distribution == 'Poisson':
                parse_dict['distribution'] = 'Poisson'
            if self.distribution == 'Both':
                parse_dict['distribution'] = 'Both' 

            parse_dict['mean'] = self.mean
            parse_dict['sigma'] = self.sigma
                         

        # add multiplicative noise
        if (self.multiplier is not None):
                callback_intensity = self._multiplicative_noise
                parse_dict['multiplier'] = self.multiplier
                 
        # random brightness and contrast
        if (self.brightness_limit is not None) or (self.contrast_limit is not None):

            callback_intensity = self._random_bright_contrast

            parse_dict['brightness_limit'] = self.brightness_limit
            parse_dict['contrast_limit'] = self.contrast_limit
            parse_dict['brightness_by_max'] = self.brightness_by_max
            parse_dict['always_apply'] = self.always_apply
            parse_dict['prob_bright_contrast'] = self.prob_bright_contrast



        # build and return augmentor with specified callback function,  the calbacks are eitehr geometic affectging the co ordinates of the 
        # clicked locations or they are purely intensity based not affecting the csv clicked locations
        if callback_geometric is not None:
            return self._return_augmentor(callback_geometric, parse_dict)

        if callback_intensity is not None:
            return self._return_augmentor_intensity(callback_intensity, parse_dict)
        else:
            raise ValueError('No augmentor returned. Arguments are not set properly.')

    def _return_augmentor(self, callback, parse_dict):
        """return augmented image, label and csv"""

        target_image = self.image
        target_labelimage = self.labelimage
        target_labelcsv = self.labelcsv
        
        # image and label augmentation by callback function
        ret_image, ret_labelcsv = callback(target_image,  parse_dict, csv = target_labelcsv) 
        ret_labelimage =  callback(target_labelimage, parse_dict) 

        return ret_image, ret_labelimage, ret_labelcsv

    def _return_augmentor_intensity(self, callback, parse_dict):
        """return augmented image with same label and csv"""

        target_image = self.image
        target_labelimage = self.labelimage
        target_labelcsv = pd.read_csv(self.labelcsv)

        # image and label augmentation by callback function
        ret_image = callback(target_image,  parse_dict) 
        ret_labelimage =  target_labelimage

        return ret_image, ret_labelimage, target_labelcsv


    def _multiplicative_noise(self, image, parse_dict):

        """ Add multiplicative noise using the albumentations library function"""
        multiplier = parse_dict['multiplier']
        intensity_transform = transforms.MultiplicativeNoise(multiplier=multiplier)
        aug_image = image
        time_points = image.shape[0]

        for i in range(time_points):

           aug_image[i,:,:] =  intensity_transform.apply(image[i,:,:])    


        return aug_image    
    

    def _random_bright_contrast(self, image, parse_dict):

        """ Add random brightness and contrast using the albumentations library function"""
         
        brightness_limit = parse_dict['brightness_limit']
        contrast_limit = parse_dict['contrast_limit']
        brightness_by_max = parse_dict['brightness_by_max']
        always_apply = parse_dict['always_apply']
        prob_bright_contrast = parse_dict['prob_bright_contrast']
        intensity_transform = transforms.RandomBrightnessContrast(brightness_limit= brightness_limit, 
            contrast_limit= contrast_limit, brightness_by_max= brightness_by_max, always_apply=always_apply, p= prob_bright_contrast) 
        aug_image = image
        time_points = image.shape[0]
        for i in range(time_points):

           aug_image[i,:,:] =  intensity_transform.apply(image[i,:,:])    


        return aug_image   
      

    def _noise_image(self, image, parse_dict):
          """ Add noise of the chosen distribution or a combination of distributions to all the timepoint of the input image"""
          mean = parse_dict['mean']
          sigma = parse_dict['sigma']
          distribution = parse_dict['distribution']   
          shape = (image.shape[1], image.shape[2])

          if distribution == 'Gaussian':
                
                addednoise = make_noise_image(shape, distribution='gaussian', mean=mean,
                          stddev=sigma)
              
          if distribution == 'Poisson':
  
                addednoise = make_noise_image(shape, distribution='poisson', mean=sigma)

          if distribution == 'Both':

                gaussiannoise = make_noise_image(shape, distribution='gaussian', mean=mean,
                          stddev=sigma)
                poissonnoise = make_noise_image(shape, distribution='poisson', mean=sigma)
            
                addednoise = gaussiannoise + poissonnoise

          else:

            raise ValueError('The distribution is not supported, has to be Gausssian, Poisson or Both (case sensitive names)')      
          
          
          aug_image = image
          time_points = image.shape[0]
          for i in range(time_points):
               aug_image[i,:,:] =  image[i,:,:] + addednoise     

          return aug_image

    def _rotate_image(self, image, parse_dict, csv = None):
        """rotate array usiong affine transformation and also if the csv file of coordinates is supplied"""
        rotate_angle = parse_dict['rotate_angle']
        rotate_matrix =  np.array([[np.cos(rotate_angle), -np.sin(rotate_angle)], [np.sin(rotate_angle), np.cos(rotate_angle)]])
        time_points = image.shape[0]
        aug_image = image
        for i in range(time_points):
             aug_image[i,:,:] =  ndimage.affine_transform(image[i,:,:],rotate_matrix)
        if csv is not None:
            dataset = pd.read_csv(csv)
            time = dataset[dataset.keys()[0]][1:]
            y = dataset[dataset.keys()[1]][1:]
            x = dataset[dataset.keys()[2]][1:]
            data = []
            for (key, t) in time.items():
                     coord_x = x[key]
                     coord_y = y[key]
                     coord_t = t
                     rotated_coord_x =  coord_x * np.cos(rotate_angle) - coord_y * np.sin(rotate_angle)
                     rotated_coord_y =  coord_x * np.sin(rotate_angle) + coord_y * np.cos(rotate_angle)
                     data.append([coord_t,rotated_coord_y, rotated_coord_x ])
            augmented_csv = pd.DataFrame(data, columns=['T', 'Y', 'X'])         
            return aug_image, augmented_csv

        if csv is None:
            return aug_image   


    