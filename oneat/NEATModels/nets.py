from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import keras as K
from keras import regularizers
from keras.layers import BatchNormalization, Activation
from keras.layers import Conv1D, Conv2D, Conv3D 
from keras.layers import ConvLSTM1D, ConvLSTM2D
from keras import layers
from keras import models
from keras.layers.core import Lambda
from keras.layers import  TimeDistributed
import tensorflow as tf


reg_weight = 1.e-4

"""
Using RESNET and stacked layer style architechtures to define NEAT architecture
"""

class Concat(layers.Layer):

     def __init__(self, axis = -1, name = 'Concat', **kwargs):

          self.axis = axis
          super(Concat, self).__init__(name = name)

     def get_config(self):
        config = super().get_config()
        config.update({
            "axis": self.axis,
            "name": self.name,
        })
        return config
 
     def call(self, x):

        y = Lambda(lambda x:layers.concatenate([x[0], x[1]], self.axis))(x)

        return y 



def LRNet(input_shape, categories, box_vector,nboxes = 1, stage_number = 3,  depth = 38, start_kernel = 3, mid_kernel = 3,  startfilter = 32,  input_weights = None, last_activation = 'softmax'):
    """ResNet Version 2 Model builder [b]
    depth of 29 == max pooling of 28 for image patch of 55
    depth of 56 == max pooling of 14 for image patch of 55
    """
    img_input = layers.Input(shape = (input_shape[0], None, None, input_shape[3]))
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = startfilter
    num_res_blocks = int((depth - 2) / 9)
    last_conv_factor =  2 ** (stage_number - 1)
    return_sequences = True
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_3d_lstm_layer(inputs=img_input,
                     num_filters=num_filters_in,
                     return_sequences = return_sequences,
                     kernel_size = start_kernel,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(stage_number):
        for res_block in range(num_res_blocks):
          
                    
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # not first layer and not first stage
                    strides = 2   # downsample

            # bottleneck residual unit
            y = resnet_3d_lstm_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             return_sequences = return_sequences,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_3d_lstm_layer(inputs=y,
                             num_filters=num_filters_in,
                               kernel_size= mid_kernel,
                               return_sequences = return_sequences,
                             conv_first=False)
            y = resnet_3d_lstm_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             return_sequences = return_sequences,
                             conv_first=False)
            
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                
                x = resnet_3d_lstm_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 return_sequences = return_sequences,
                                 batch_normalization=False)
                
                
                
            x = K.layers.add([x, y])
        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = resnet_3d_lstm_layer(inputs=x,
                             num_filters=num_filters_out,
                             kernel_size = mid_kernel,
                             return_sequences = False,
                             conv_first=False)
 
    x = (Conv2D(categories + nboxes * box_vector, kernel_size= mid_kernel,kernel_regularizer=regularizers.l2(reg_weight), padding = 'same'))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    input_cat = Lambda(lambda x:x[:,:,:,0:categories])(x)
    input_box = Lambda(lambda x:x[:,:,:,categories:])(x)

        
    output_cat = (Conv2D(categories, (round(input_shape[1]/last_conv_factor),round(input_shape[2]/last_conv_factor)),activation= last_activation,kernel_regularizer=regularizers.l2(reg_weight), padding = 'valid', name = 'yolo'))(input_cat)
    output_box = (Conv2D(nboxes*(box_vector), (round(input_shape[1]/last_conv_factor),round(input_shape[2]/last_conv_factor)),activation= 'sigmoid' ,kernel_regularizer=regularizers.l2(reg_weight), padding = 'valid', name = 'secyolo'))(input_box)

    block = Concat(-1)
    outputs = block([output_cat,output_box]) 
  
    inputs = img_input
   
    # Create model.
    model = models.Model(inputs, outputs)


    if input_weights is not None:

       model.load_weights(input_weights, by_name =True)
    
    return model


def VollNet(input_shape, categories, box_vector,nboxes = 1, stage_number = 3,  depth = 38, start_kernel = 3, mid_kernel = 3, startfilter = 32,  input_weights = None, last_activation = 'softmax'):
  
    img_input = layers.Input(shape = (None, None, None, input_shape[3]))
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = startfilter
    num_res_blocks = int((depth - 2) / 9)
    last_conv_factor =  2 ** (stage_number - 1)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_3D_layer(inputs=img_input,
                     num_filters=num_filters_in,
                     kernel_size = start_kernel,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(stage_number):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # not first layer and not first stage
                    strides = 2   # downsample

            # bottleneck residual unit
            y = resnet_3D_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_3D_layer(inputs=y,
                             num_filters=num_filters_in,
                               kernel_size= mid_kernel,
                             conv_first=False)
            y = resnet_3D_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_3D_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
              
            x = K.layers.add([x, y])
        num_filters_in = num_filters_out
    x = (Conv3D(categories + nboxes * box_vector, kernel_size= mid_kernel,kernel_regularizer=regularizers.l2(reg_weight), padding = 'same'))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    input_cat = Lambda(lambda x:x[:,:,:,:,0:categories])(x)
    input_box = Lambda(lambda x:x[:,:,:,:,categories:])(x)
    
        
    output_cat = (Conv3D(categories, (round(input_shape[0]/last_conv_factor),round(input_shape[1]/last_conv_factor),round(input_shape[2]/last_conv_factor)),activation= last_activation,kernel_regularizer=regularizers.l2(reg_weight), padding = 'valid', name = 'yolo'))(input_cat)
    output_box = (Conv3D(nboxes*(box_vector), (round(input_shape[0]/last_conv_factor),round(input_shape[1]/last_conv_factor),round(input_shape[2]/last_conv_factor)),activation= 'sigmoid' ,kernel_regularizer=regularizers.l2(reg_weight), padding = 'valid', name = 'secyolo'))(input_box)

    block = Concat(-1)
    outputs = block([output_cat,output_box]) 
  
    inputs = img_input
   
    # Create model.
    model = models.Model(inputs, outputs)


    if input_weights is not None:

       model.load_weights(input_weights, by_name = True)
    
    return model


def resnet_lstm_v2(input_shape, categories, box_vector,nboxes = 1, stage_number = 3,  depth = 38,  start_kernel = 3, mid_kernel = 3, startfilter = 48,  input_weights = None, last_activation = 'softmax'):
    """ResNet Version 2 Model builder [b]
    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """

    last_conv_factor =  2 ** (stage_number - 1)
    img_input = layers.Input(shape = (input_shape[0], input_shape[1], input_shape[2]))
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = startfilter
    num_res_blocks = int((depth - 2) / 9)
    return_sequences = True
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_lstm_layer(inputs=img_input,
                     num_filters=num_filters_in,
                     kernel_size = start_kernel,
                     return_sequences = return_sequences,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(stage_number):
        for res_block in range(num_res_blocks):
            
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # not first layer and not first stage
                    strides = 2  # downsample

            # bottleneck residual unit
            y = resnet_lstm_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             return_sequences = return_sequences,
                             conv_first=False)
            y = resnet_lstm_layer(inputs=y,
                             num_filters=num_filters_in,
                             return_sequences = return_sequences,
                             kernel_size= mid_kernel,
                             conv_first=False)
            y = resnet_lstm_layer(inputs=y,
                             num_filters=num_filters_out,
                             return_sequences = return_sequences,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                
                    
                x = resnet_lstm_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 return_sequences = return_sequences,
                                 batch_normalization=False)
              
            x = K.layers.add([x, y])
        
        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = resnet_lstm_layer(inputs=x,
                             num_filters=num_filters_out,
                             kernel_size = mid_kernel,
                             return_sequences = False,
                             conv_first=False)
    x = (Conv2D(categories + nboxes * box_vector, kernel_size= mid_kernel,kernel_regularizer=regularizers.l2(reg_weight), padding = 'same'))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    input_cat = Lambda(lambda x:x[:,:,:,0:categories])(x)
    input_box = Lambda(lambda x:x[:,:,:,categories:])(x)

    output_cat = (Conv2D(categories, (round(input_shape[0]/last_conv_factor),round(input_shape[1]/last_conv_factor)),activation= last_activation ,kernel_regularizer=regularizers.l2(reg_weight), padding = 'valid'))(input_cat)
    output_box = (Conv2D(nboxes * (box_vector), (round(input_shape[0]/last_conv_factor),round(input_shape[1]/last_conv_factor)),activation= 'sigmoid' ,kernel_regularizer=regularizers.l2(reg_weight), padding = 'valid'))(input_box)
    

    block = Concat(-1)
    outputs = block([output_cat,output_box])
    
    inputs = img_input
   
     
    # Create model.
    model = models.Model(inputs, outputs)
    
    
    if input_weights is not None:

        model.load_weights(input_weights, by_name =True)
        
    return model



def resnet_v2(input_shape, categories, box_vector,nboxes = 1, stage_number = 3,  depth = 38,  start_kernel = 3, mid_kernel = 3, startfilter = 48,  input_weights = None, last_activation = 'softmax'):
    """ResNet Version 2 Model builder [b]
    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """

    last_conv_factor =  2 ** (stage_number - 1)
    img_input = layers.Input(shape = (None, None, input_shape[2]))
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = startfilter
    num_res_blocks = int((depth - 2) / 9)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=img_input,
                     num_filters=num_filters_in,
                     kernel_size = start_kernel,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(stage_number):
        for res_block in range(num_res_blocks):
           
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # not first layer and not first stage
                    strides = 2  # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             kernel_size= mid_kernel,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                   
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
              
            x = K.layers.add([x, y])
        
        num_filters_in = num_filters_out

    x = (Conv2D(categories + nboxes * box_vector, kernel_size= mid_kernel,kernel_regularizer=regularizers.l2(reg_weight), padding = 'same'))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    input_cat = Lambda(lambda x:x[:,:,:,0:categories])(x)
    input_box = Lambda(lambda x:x[:,:,:,categories:])(x)

    output_cat = (Conv2D(categories, (round(input_shape[0]/last_conv_factor),round(input_shape[1]/last_conv_factor)),activation= last_activation ,kernel_regularizer=regularizers.l2(reg_weight), padding = 'valid'))(input_cat)
    output_box = (Conv2D(nboxes * (box_vector), (round(input_shape[0]/last_conv_factor),round(input_shape[1]/last_conv_factor)),activation= 'sigmoid' ,kernel_regularizer=regularizers.l2(reg_weight), padding = 'valid'))(input_box)
    

    block = Concat(-1)
    outputs = block([output_cat,output_box])
    
    inputs = img_input
   
     
    # Create model.
    model = models.Model(inputs, outputs)
    
    
    if input_weights is not None:

        model.load_weights(input_weights, by_name =True)
        
    return model

def resnet_1D_regression(input_shape,  stage_number = 3,  depth = 38,  start_kernel = 3, mid_kernel = 3, startfilter = 48,  input_weights = None, last_activation = 'linear'):
    """
    # Returns
        model (Model): Keras model instance
    """

    last_conv_factor =  2 ** (stage_number - 1)
    img_input = layers.Input(shape = input_shape)
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = startfilter
    num_res_blocks = int((depth - 2) / 9)

    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer_1D(inputs=img_input,
                     num_filters=num_filters_in,
                     kernel_size = start_kernel,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(stage_number):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # not first layer and not first stage
                    strides = 2  # downsample

            # bottleneck residual unit
            y = resnet_layer_1D(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer_1D(inputs=y,
                             num_filters=num_filters_in,
                               kernel_size= mid_kernel,
                             conv_first=False)
            y = resnet_layer_1D(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer_1D(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
              
            x = K.layers.add([x, y])
        
        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = (Conv2D(1, kernel_size= mid_kernel,kernel_regularizer=regularizers.l2(reg_weight), padding = 'same'))(x)
    
    outputs = x
    
    inputs = img_input
   
     
    # Create model.
    model = models.Model(inputs, outputs)
    
    
    if input_weights is not None:

        model.load_weights(input_weights, by_name =True)
        
    return model

def resnet_v2_class(input_shape, categories, box_vector,nboxes = 1, stage_number = 3,  depth = 38,  start_kernel = 3, mid_kernel = 3, startfilter = 48,  input_weights = None, last_activation = 'softmax'):
    """ResNet Version 2 Model builder [b]
    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """

    last_conv_factor =  2 ** (stage_number - 1)
    img_input = layers.Input(shape = (None, None, input_shape[2]))
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = startfilter
    num_res_blocks = int((depth - 2) / 9)

    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=img_input,
                     num_filters=num_filters_in,
                     kernel_size = start_kernel,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(stage_number):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # not first layer and not first stage
                    strides = 2  # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                               kernel_size= mid_kernel,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
              
            x = K.layers.add([x, y])
        
        num_filters_in = num_filters_out
    x = (Conv2D(categories + nboxes * box_vector, kernel_size= mid_kernel,kernel_regularizer=regularizers.l2(reg_weight), padding = 'same'))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    input_cat = Lambda(lambda x:x[:,:,:,0:categories])(x)
    outputs = (Conv2D(categories, (round(input_shape[0]/last_conv_factor),round(input_shape[1]/last_conv_factor)),activation= last_activation ,kernel_regularizer=regularizers.l2(reg_weight), padding = 'valid'))(input_cat)
   

    inputs = img_input
   
     
    # Create model.
    model = models.Model(inputs, outputs)
    
    
    if input_weights is not None:

        model.load_weights(input_weights, by_name =True)
        
    return model    
  

def resnet_lstm_v2_class(input_shape, categories, box_vector,nboxes = 1, stage_number = 3,  depth = 38,  start_kernel = 3, mid_kernel = 3, startfilter = 48,  input_weights = None, last_activation = 'softmax'):
    """ResNet Version 2 Model builder [b]
    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """

    last_conv_factor =  2 ** (stage_number - 1)
    img_input = layers.Input(shape = (None, None, input_shape[2]))
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = startfilter
    num_res_blocks = int((depth - 2) / 9)

    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_lstm_layer(inputs=img_input,
                     num_filters=num_filters_in,
                     kernel_size = start_kernel,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(stage_number):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # not first layer and not first stage
                    strides = 2  # downsample

            # bottleneck residual unit
            y = resnet_lstm_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_lstm_layer(inputs=y,
                             num_filters=num_filters_in,
                               kernel_size= mid_kernel,
                             conv_first=False)
            y = resnet_lstm_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_lstm_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
              
            x = K.layers.add([x, y])
        
        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    
    x = resnet_lstm_layer(inputs=x,
                             num_filters=num_filters_out,
                             kernel_size = mid_kernel,
                             return_sequences = False,
                             conv_first=False)
    x = (Conv2D(categories + nboxes * box_vector, kernel_size= mid_kernel,kernel_regularizer=regularizers.l2(reg_weight), padding = 'same'))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    input_cat = Lambda(lambda x:x[:,:,:,0:categories])(x)
    outputs = (Conv2D(categories, (round(input_shape[0]/last_conv_factor),round(input_shape[1]/last_conv_factor)),activation= last_activation ,kernel_regularizer=regularizers.l2(reg_weight), padding = 'valid'))(input_cat)
   

    inputs = img_input
   
     
    # Create model.
    model = models.Model(inputs, outputs)
    
    
    if input_weights is not None:

        model.load_weights(input_weights, by_name =True)
        
    return model





def resnet_3d_lstm_layer(inputs,
                         num_filters = 64,
                         kernel_size = 3,
                         strides = 1,
                         return_sequences = True,
                         activation='relu',
                         batch_normalization=True,
                         conv_first = True,
                         ):
    
     conv_lstm_3d = ConvLSTM2D(filters = num_filters, 
                kernel_size = (kernel_size, kernel_size),  
                activation=activation, 
                strides = strides,
                data_format = 'channels_last',
                return_sequences = return_sequences, 
                padding = "same")
     x = inputs
     if conv_first:
        x = conv_lstm_3d(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
     else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv_lstm_3d(x)
     return x 

def resnet_lstm_layer(inputs,
                 num_filters=64,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 return_sequences=True,
                 batch_normalization=True,
                 conv_first=True):
   
    conv_lstm = ConvLSTM1D(filters = num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  return_sequences= return_sequences,
                  kernel_regularizer=regularizers.l2(1e-4))

    x = inputs
    if conv_first:
        x = conv_lstm(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv_lstm(x)
    return x
    
   
def resnet_layer(inputs,
                 num_filters=64,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def resnet_layer_1D(inputs,
                 num_filters=64,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """1D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv1D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x
    
   
    
def resnet_3D_layer(inputs,
                 num_filters=64,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """3D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv3D(num_filters,
                  kernel_size = kernel_size,
                  strides = strides,
                  padding ='same',
                  kernel_initializer ='he_normal',
                  kernel_regularizer = regularizers.l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x
    


