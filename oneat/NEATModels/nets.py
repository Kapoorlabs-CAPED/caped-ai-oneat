from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import keras as K
from keras import regularizers
from keras.layers import BatchNormalization, Activation
from keras.layers import Conv2D, Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras import layers
from keras import models
from keras.layers.core import Lambda
from keras.layers import  TimeDistributed, Reshape



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






def TDresnet_layer(inputs,
                 num_filters=64,
                 kernel_size= 3,
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
    
    conv = TimeDistributed(Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(1e-4)))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = TimeDistributed(BatchNormalization())(x)
        if activation is not None:
            x = TimeDistributed(Activation(activation))(x)
    else:
        if batch_normalization:
            x = TimeDistributed(BatchNormalization())(x)
        if activation is not None:
            x = TimeDistributed(Activation(activation))(x)
        x = conv(x)
    return x








def OSNET(input_shape, categories,unit, box_vector, nboxes = 1, stage_number = 3, last_conv_factor = 4, depth = 38, start_kernel = 3, mid_kernel = 3, lstm_kernel = 3, startfilter = 48,  input_weights = None,last_activation = 'softmax'):
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

    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = ThreeDresnet_layer(inputs=img_input,
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
  
            x = ThreeDresnet_layer(inputs=x,
                             num_filters=num_filters_in,
                               kernel_size= mid_kernel,
                             conv_first=False,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization)


              
        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    num_filters_in_TD = startfilter
    num_res_blocks = int((depth - 2) / 9)
    
    z = TDresnet_layer(inputs=img_input,
                     num_filters=num_filters_in_TD,
                     kernel_size = start_kernel,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(stage_number):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in_TD * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in_TD * 2
                if res_block == 0:  # not first layer and not first stage
                    strides = 2   # downsample

            # bottleneck residual unit
      
            z = TDresnet_layer(inputs=z,
                             num_filters=num_filters_in_TD,
                               kernel_size= mid_kernel,
                             conv_first=False,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization)


              
        num_filters_in_TD = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    z = BatchNormalization()(z)
    z = Activation('relu')(z)
    
    
    
    branchAdd = layers.add([z, x])
    

    
    x = ConvLSTM2D(filters = unit, kernel_size = (lstm_kernel, lstm_kernel),  activation='relu', data_format = 'channels_last', return_sequences = False, padding = "same", name = "newlstmdeep")(branchAdd)


    x = (Conv2D(categories + nboxes * box_vector, kernel_size= mid_kernel,kernel_regularizer=regularizers.l2(reg_weight), padding = 'same'))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    input_cat = Lambda(lambda x:x[:,:,:,0:categories])(x)
    input_box = Lambda(lambda x:x[:,:,:,categories:])(x)

        
    output_cat = (Conv2D(categories, (round(input_shape[1]/last_conv_factor),round(input_shape[2]/last_conv_factor)),activation= last_activation ,kernel_regularizer=regularizers.l2(reg_weight), padding = 'valid', name = 'yolo'))(input_cat)
    output_box = (Conv2D(nboxes * (box_vector), (round(input_shape[1]/last_conv_factor),round(input_shape[2]/last_conv_factor)),activation= 'sigmoid' ,kernel_regularizer=regularizers.l2(reg_weight), padding = 'valid', name = 'secyolo'))(input_box)




    block = Concat(-1)
    outputs = block([output_cat,output_box]) 
    
    
    
    inputs = img_input
   
    # Create model.
    model = models.Model(inputs, outputs)


    if input_weights is not None:

       model.load_weights(input_weights, by_name =True)
    
    return model


def ORNET(input_shape, categories,unit, box_vector,nboxes = 1, stage_number = 3, last_conv_factor = 4, depth = 38, start_kernel = 3, mid_kernel = 3, lstm_kernel = 3, startfilter = 32,  input_weights = None, last_activation = 'softmax'):
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

    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = ThreeDresnet_layer(inputs=img_input,
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
            y = ThreeDresnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = ThreeDresnet_layer(inputs=y,
                             num_filters=num_filters_in,
                               kernel_size= mid_kernel,
                             conv_first=False)
            y = ThreeDresnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = ThreeDresnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
              
            x = K.layers.add([x, y])
        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    num_filters_in_TD = startfilter
    num_res_blocks = int((depth - 2) / 9)
    
    z = TDresnet_layer(inputs=img_input,
                     num_filters=num_filters_in_TD,
                     kernel_size = start_kernel,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(stage_number):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in_TD * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in_TD * 2
                if res_block == 0:  # not first layer and not first stage
                    strides = 2   # downsample

            # bottleneck residual unit
            yz = TDresnet_layer(inputs=z,
                             num_filters=num_filters_in_TD,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            yz = TDresnet_layer(inputs=yz,
                             num_filters=num_filters_in_TD,
                               kernel_size= mid_kernel,
                             conv_first=False)
            yz = TDresnet_layer(inputs=yz,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                z = TDresnet_layer(inputs=z,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
              
            z = K.layers.add([z, yz])
        num_filters_in_TD = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    z = BatchNormalization()(z)
    z = Activation('relu')(z)
    
    
    
    branchAdd = layers.add([z, x])
    

    
    x = ConvLSTM2D(filters = unit, kernel_size = (lstm_kernel, lstm_kernel),  activation='relu', data_format = 'channels_last', return_sequences = False, padding = "same", name = "newlstmdeep")(branchAdd)


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









    
def ThreeDresnet_layer(inputs,
                 num_filters=64,
                 kernel_size= 3,
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
    conv = Conv3D(num_filters,
                  kernel_size=kernel_size,
                  strides=(1,strides,strides),
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = (BatchNormalization())(x)
        if activation is not None:
            x = (Activation(activation))(x)
    else:
        if batch_normalization:
            x = (BatchNormalization())(x)
        if activation is not None:
            x = (Activation(activation))(x)
        x = conv(x)
    return x
    





def resnet_v2(input_shape, categories, box_vector,nboxes = 1, stage_number = 3, last_conv_factor = 4,  depth = 38,  start_kernel = 3, mid_kernel = 3, startfilter = 48,  input_weights = None, last_activation = 'softmax'):
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

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
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
  
def seqnet_v2(input_shape, categories, box_vector,nboxes = 1,stage_number = 3, last_conv_factor = 4, depth = 38, start_kernel = 3, mid_kernel = 3, startfilter = 48,  input_weights = None, last_activation = 'softmax'):
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

            x = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                               kernel_size= mid_kernel,
                             conv_first=False,
                             strides = strides,
                             activation=activation,
                             batch_normalization=batch_normalization)
           
 
              
        
        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = (Conv2D(categories + nboxes * box_vector, kernel_size= mid_kernel,kernel_regularizer=regularizers.l2(reg_weight), padding = 'same'))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    

    
    input_cat = Lambda(lambda x:x[:,:,:,0:categories])(x)
    input_box = Lambda(lambda x:x[:,:,:,categories:])(x)
    
      
    

    output_cat = (Conv2D(categories, (round(input_shape[0]/last_conv_factor),round(input_shape[1]/last_conv_factor)),activation= last_activation ,kernel_regularizer=regularizers.l2(reg_weight), padding = 'valid'))(input_cat)
    output_box = (Conv2D(nboxes*(box_vector), (round(input_shape[0]/last_conv_factor),round(input_shape[1]/last_conv_factor)),activation= 'sigmoid' ,kernel_regularizer=regularizers.l2(reg_weight), padding = 'valid'))(input_box)
    

    block = Concat(-1)
    outputs = block([output_cat,output_box])

    
    inputs = img_input
   
     
    # Create model.
    model = models.Model(inputs, outputs)
    
    
    if input_weights is not None:

        model.load_weights(input_weights, by_name =True)
        
    return model
      

def resnet_3D_v2(input_shape, categories,box_vector, stage_number = 3, last_conv_factor = 4,  depth = 38,  start_kernel = 3, mid_kernel = 3, startfilter = 48,  input_weights = None, last_activation = 'softmax'):
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
    img_input = layers.Input(shape = (input_shape[0], None, None, input_shape[3]))
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = startfilter
    num_res_blocks = int((depth - 2) / 9)

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
                    strides = 2  # downsample

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

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = (Conv3D(categories+  box_vector, kernel_size= mid_kernel,kernel_regularizer=regularizers.l2(reg_weight), padding = 'same'))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    input_cat = Lambda(lambda x:x[:,:,:,:,0:categories])(x)
    input_box = Lambda(lambda x:x[:,:,:,:,categories:])(x)
    
      
    

    output_cat = (Conv3D(categories, (round(input_shape[0]),round(input_shape[1]/last_conv_factor), round(input_shape[2]/last_conv_factor)),activation= last_activation ,kernel_regularizer=regularizers.l2(reg_weight), padding = 'valid'))(input_cat)
    output_box = (Conv3D((box_vector), (round(input_shape[0]),round(input_shape[1]/last_conv_factor), round(input_shape[2]/last_conv_factor)),activation= 'sigmoid' ,kernel_regularizer=regularizers.l2(reg_weight), padding = 'valid'))(input_box)
    

    block = Concat(-1)
    outputs = block([output_cat,output_box])


    
    inputs = img_input
   
    # Create model.
    model = models.Model(inputs, outputs)
    
    
    if input_weights is not None:

        model.load_weights(input_weights, by_name =True)
        
    return model
  
def seqnet_3D_v2(input_shape, categories, box_vector, stage_number = 3, last_conv_factor = 4, depth = 38, start_kernel = 3, mid_kernel = 3, startfilter = 48,  input_weights = None, last_activation = 'softmax'):
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
    img_input = layers.Input(shape = (input_shape[0], None, None, input_shape[3]))
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = startfilter
    num_res_blocks = int((depth - 2) / 9)

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
                    strides = 2  # downsample

            # bottleneck residual unit

            x = resnet_3D_layer(inputs=x,
                             num_filters=num_filters_in,
                               kernel_size= mid_kernel,
                             conv_first=False,
                             strides = strides,
                             activation=activation,
                             batch_normalization=batch_normalization)
           
 
              
        
        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = (Conv3D(categories+ box_vector, kernel_size= mid_kernel,kernel_regularizer=regularizers.l2(reg_weight), padding = 'same'))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    

    

    input_cat = Lambda(lambda x:x[:,:,:,:,0:categories])(x)
    input_box = Lambda(lambda x:x[:,:,:,:,categories:])(x)
    
      
    

    output_cat = (Conv3D(categories, (round(input_shape[0]),round(input_shape[1]/last_conv_factor), round(input_shape[2]/last_conv_factor)),activation= last_activation ,kernel_regularizer=regularizers.l2(reg_weight), padding = 'valid'))(input_cat)
    output_box = (Conv3D((box_vector), (round(input_shape[0]),round(input_shape[1]/last_conv_factor), round(input_shape[2]/last_conv_factor)),activation= 'sigmoid' ,kernel_regularizer=regularizers.l2(reg_weight), padding = 'valid'))(input_box)
    

    block = Concat(-1)
    outputs = block([output_cat,output_box])

    
    inputs = img_input
   
     
    # Create model.
    model = models.Model(inputs, outputs)
    
    
    if input_weights is not None:

        model.load_weights(input_weights, by_name =True)
        
    return model
      


    
   
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
                  kernel_size=kernel_size,
                  strides=(1,strides,strides),
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
    


