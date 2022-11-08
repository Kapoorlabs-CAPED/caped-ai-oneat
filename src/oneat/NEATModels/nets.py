from tensorflow import keras as K
from keras import layers, models, regularizers, backend
from keras.layers import (
    Activation,
    BatchNormalization,
    Conv1D,
    Conv2D,
    Conv3D,
    ConvLSTM1D,
    ConvLSTM2D,
    concatenate
)
from keras.layers.core import Lambda

reg_weight = 1.0e-4


class Concat(layers.Layer):
    def __init__(self, axis=-1):

        self.axis = axis
        super().__init__()

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "axis": self.axis
            }
        )
        return config

    def call(self, x):

        y = Lambda(lambda x: layers.concatenate([x[0], x[1]], self.axis))(x)

        return y


def LRNet(
    input_shape,
    categories,
    box_vector,
    nboxes=1,
    stage_number=3,
    depth=38,
    start_kernel=3,
    mid_kernel=3,
    startfilter=32,
    input_model=None,
    last_activation="softmax",
):
    """ResNet Version 2 Model builder [b]
    depth of 29 == max pooling of 28 for image patch of 55
    depth of 56 == max pooling of 14 for image patch of 55
    """
    img_input = layers.Input(
        shape=(input_shape[0], None, None, input_shape[3])
    )
    if (depth - 2) % 9 != 0:
        raise ValueError("depth should be 9n+2 (eg 56 or 110 in [b])")
    # Start model definition.
    num_filters_in = startfilter
    num_res_blocks = int((depth - 2) / 9)
    last_conv_factor = 2 ** (stage_number - 1)
    return_sequences = True
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = _resnet_3d_lstm_layer(
        inputs=img_input,
        num_filters=num_filters_in,
        return_sequences=return_sequences,
        kernel_size=start_kernel,
        conv_first=True,
    )

    # Instantiate the stack of residual units
    for stage in range(stage_number):
        for res_block in range(num_res_blocks):

            activation = "relu"
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

            y = _resnet_3d_lstm_layer(
                inputs=x,
                num_filters=num_filters_in,
                kernel_size=1,
                strides=strides,
                activation=activation,
                return_sequences=return_sequences,
                batch_normalization=batch_normalization,
                conv_first=False,
            )
            y = _resnet_3d_lstm_layer(
                inputs=y,
                num_filters=num_filters_in,
                kernel_size=mid_kernel,
                return_sequences=return_sequences,
                conv_first=False,
            )
            y = _resnet_3d_lstm_layer(
                inputs=y,
                num_filters=num_filters_out,
                kernel_size=1,
                return_sequences=return_sequences,
                conv_first=False,
            )

            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims

                x = _resnet_3d_lstm_layer(
                    inputs=x,
                    num_filters=num_filters_out,
                    kernel_size=1,
                    strides=strides,
                    activation=None,
                    return_sequences=return_sequences,
                    batch_normalization=False,
                )

            x = K.layers.add([x, y])
        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = _resnet_3d_lstm_layer(
        inputs=x,
        num_filters=num_filters_out,
        kernel_size=mid_kernel,
        return_sequences=False,
        conv_first=False,
    )

    x = (
        Conv2D(
            categories + nboxes * box_vector,
            kernel_size=mid_kernel,
            kernel_regularizer=regularizers.l2(reg_weight),
            padding="same",
        )
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    input_cat = Lambda(lambda x: x[:, :, :, 0:categories])(x)
    input_box = Lambda(lambda x: x[:, :, :, categories:])(x)

    output_cat = (
        Conv2D(
            categories,
            (
                round(input_shape[1] / last_conv_factor),
                round(input_shape[2] / last_conv_factor),
            ),
            activation=last_activation,
            kernel_regularizer=regularizers.l2(reg_weight),
            padding="valid"
        )
    )(input_cat)
    output_box = (
        Conv2D(
            nboxes * (box_vector),
            (
                round(input_shape[1] / last_conv_factor),
                round(input_shape[2] / last_conv_factor),
            ),
            activation="sigmoid",
            kernel_regularizer=regularizers.l2(reg_weight),
            padding="valid"
        )
    )(input_box)

    block = Concat(-1)
    outputs = block([output_cat, output_box])

    inputs = img_input

    # Create model.
    model = models.Model(inputs, outputs)

    if input_model is not None:

        model =  models.load_model(input_model)

    return model



    
    
    
    
def _voll_dense_block(x, nb_layers,
    num_filters,
    kernel_size,
    activation):
    
    for _ in range(nb_layers):
        x = _voll_dense_conv(x, num_filters, kernel_size,activation)
    return x    
    
def _voll_transition_block(x,
                           reduction,
                           activation="relu"):
    
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv3D(
        int(backend.int_shape(x)[-1] * reduction),
        1,
        use_bias=False
    )(x)
    
    x = layers.MaxPooling3D((2, 2, 2))(x)
    
    return x
    
def _voll_dense_conv(
    x,
    num_filters,
    kernel_size=3,
    activation="relu"
):
    
    
    y = BatchNormalization()(x)
    y = Activation(activation)(y)
    y = Conv3D(
        4 * num_filters,
        kernel_size= 1,
        use_bias = False,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(reg_weight),
    ) (y)
    y = BatchNormalization()(y)
    y = Activation(activation)(y)
    
    y = Conv3D(
        num_filters,
        kernel_size= kernel_size,
        use_bias = False,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(reg_weight),
    ) (y)
    
    block = Concat(-1)
    x = block([y, x])
    
    return x    
    
def _voll_conv(
    inputs,
    num_filters=64,
    kernel_size=3,
    strides=1,
    activation="relu",
    batch_normalization=True,
    conv_first=True,
):
    
    conv = Conv3D(
        num_filters,
        kernel_size=kernel_size,
        strides= strides,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(1e-4),
    )

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


def _voll_top(input_shape, stage_number):
    
        last_conv_factor = 2 ** (stage_number - 1) 
        print(input_shape, input_shape[0], input_shape[1], input_shape[2], input_shape[3])
        img_input = layers.Input(shape=(None, None, None, input_shape[3]))
        
        return last_conv_factor, img_input

def _voll_bottom(x, img_input, input_shape, categories, mid_kernel, last_conv_factor, last_activation, nboxes, box_vector, input_model):
        
        x = (Conv3D(categories + nboxes * box_vector, kernel_size= mid_kernel, kernel_regularizer=regularizers.l2(reg_weight), padding = 'same'))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        input_cat = Lambda(lambda x: x[:, :, :, :, 0:categories])(x)
        input_box = Lambda(lambda x: x[:, :, :, :, categories:])(x)
        output_cat = (
            Conv3D(
                categories,
                (
                    round(input_shape[0]/ last_conv_factor ),
                    round(input_shape[1] / last_conv_factor),
                    round(input_shape[2] / last_conv_factor),
                ),
                activation=last_activation,
                kernel_regularizer=regularizers.l2(reg_weight),
                padding="valid"
            )
        )(input_cat)
        output_box = (
            Conv3D(
                nboxes * (box_vector),
                (
                    round(input_shape[0] / last_conv_factor),
                    round(input_shape[1] / last_conv_factor),
                    round(input_shape[2] / last_conv_factor),
                ),
                activation="sigmoid",
                kernel_regularizer=regularizers.l2(reg_weight),
                padding="valid"
            )
        )(input_box)

        block = Concat(-1)
        outputs = block([output_cat, output_box])
        inputs = img_input
        # Create model.
        model = models.Model(inputs, outputs)

        if input_model is not None:

            model =  models.load_model(input_model)

        return model
    
class DenseNet:
    def __init__(self, depth, startfilter, stage_number, start_kernel, mid_kernel,reduction, activation='relu', **kwargs):
        self.stage_number = stage_number
        self.start_kernel = start_kernel
        self.mid_kernel = mid_kernel
        self.reduction = reduction
        self.depth = depth
        self.startfilter = startfilter
        self.activation = activation
        self.kwargs = kwargs

    def __call__(self, x):
        self.nb_layers = []
        if type(self.depth) is dict:
            for (k,v) in self.depth.items():
                self.nb_layers.append(v) # get the list
                
        if len(self.nb_layers) != self.stage_number:
                raise ValueError('If `stage_number` is a list, its length must match '
                                 'the number of layers provided by `depth`.')
        for stage in range(self.stage_number):
            
            x = _voll_dense_block(x, self.nb_layers[stage],
                self.startfilter//2,
                self.mid_kernel,
                self.activation)
            if stage < self.stage_number -1:
               x =_voll_transition_block(x,
                           self.reduction,
                           self.activation)
               
            x = BatchNormalization()(x)
            x = Activation(self.activation)(x)   
        
        return x
class ResNet:
    def __init__(self, depth, startfilter, stage_number, start_kernel, mid_kernel, activation='relu', **kwarg):
        
        self.depth = depth
        self.startfilter = startfilter
        self.stage_number = stage_number
        self.start_kernel = start_kernel
        self.mid_kernel = mid_kernel
        self.activation = activation  
        
    def __call__(self, x):
        
        self.num_filters_in = self.startfilter
        if type(self.depth) is dict:
            for (k,v) in self.depth.items():
                self.nb_layers = v
        else:
                
              self.nb_layers = self.depth      
                
        num_res_blocks = int((self.nb_layers - 2) / 9)
        x = _voll_conv(
            x,
            num_filters=self.startfilter,
            kernel_size=self.start_kernel,
            conv_first=True
        )
              
        for stage in range(self.stage_number):
                for res_block in range(num_res_blocks):
                    activation = self.activation
                    batch_normalization = True
                    strides = 1
                    if stage == 0:
                        self.num_filters_out = self.num_filters_in * 4
                        if res_block == 0:  # first layer and first stage
                            activation = None
                            batch_normalization = False
                    else:
                        self.num_filters_out = self.num_filters_in * 2
                        if res_block == 0:  # not first layer and not first stage
                            strides = 2  # downsample

                    y = _voll_conv(
                        inputs=x,
                        num_filters=self.num_filters_in,
                        kernel_size=1,
                        strides=strides,
                        activation=activation,
                        batch_normalization=batch_normalization,
                        conv_first=False,
                    )
                    y = _voll_conv(
                        inputs=y,
                        num_filters=self.num_filters_in,
                        kernel_size=self.mid_kernel,
                        conv_first=False,
                    )
                    y = _voll_conv(
                        inputs=y,
                        num_filters=self.num_filters_out,
                        kernel_size=1,
                        conv_first=False,
                    )
                    if res_block == 0:
                        # linear projection residual shortcut connection to match
                        # changed dims
                        x = _voll_conv(
                            inputs=x,
                            num_filters=self.num_filters_out,
                            kernel_size=1,
                            strides=strides,
                            activation=None,
                            batch_normalization=False,
                        )

                    x = K.layers.add([x, y])
                self.num_filters_in = self.num_filters_out
                
        return x        
                
def DenseVollNet(
                input_shape,
                categories: dict,
                box_vector,
                nboxes: int=1,
                start_kernel: int=7,
                mid_kernel: int=3,
                startfilter: int=64,
                stage_number: int = 3,
                input_model=None,
                last_activation: str="softmax",
                depth: dict={'depth_0': 6, 'depth_1': 12, 'depth_2':24, 'depth_3':16},
                reduction: float = 0.5,
                **kwargs
):
    
        last_conv_factor, img_input = _voll_top(input_shape = input_shape, stage_number = stage_number)
        densenet = DenseNet(depth, startfilter, stage_number, start_kernel, mid_kernel, reduction, **kwargs)
        x = densenet(img_input)
        model = _voll_bottom(x, img_input, input_shape, categories, mid_kernel, last_conv_factor, last_activation, nboxes, box_vector, input_model)
        return model
        


def VollNet(
            input_shape,
            categories: dict,
            box_vector,
            nboxes: int=1,
            start_kernel: int=7,
            mid_kernel: int=3,
            startfilter: int=64,
            stage_number: int = 3,
            input_model=None,
            last_activation: str="softmax",
            depth: dict={'depth_0': 29}
):
    
        last_conv_factor, img_input = _voll_top(input_shape = input_shape, stage_number = stage_number)
        resnet = ResNet(depth, startfilter, stage_number, start_kernel, mid_kernel)
        x = resnet(img_input)
        model = _voll_bottom(x, img_input, input_shape, categories, mid_kernel, last_conv_factor, last_activation, nboxes, box_vector, input_model)
        return model


def resnet_lstm_v2(
    input_shape,
    categories,
    box_vector,
    nboxes=1,
    stage_number=3,
    depth=38,
    start_kernel=3,
    mid_kernel=3,
    startfilter=48,
    input_model=None,
    last_activation="softmax",
):
   

    last_conv_factor = 2 ** (stage_number - 1)
    img_input = layers.Input(
        shape=(input_shape[0], input_shape[1], input_shape[2])
    )
    if (depth - 2) % 9 != 0:
        raise ValueError("depth should be 9n+2 (eg 56 or 110 in [b])")
    # Start model definition.
    num_filters_in = startfilter
    num_res_blocks = int((depth - 2) / 9)
    return_sequences = True
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = _resnet_lstm_layer(
        inputs=img_input,
        num_filters=num_filters_in,
        kernel_size=start_kernel,
        return_sequences=return_sequences,
        conv_first=True,
    )

    # Instantiate the stack of residual units
    for stage in range(stage_number):
        for res_block in range(num_res_blocks):

            activation = "relu"
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

            y = _resnet_lstm_layer(
                inputs=x,
                num_filters=num_filters_in,
                kernel_size=1,
                strides=strides,
                activation=activation,
                batch_normalization=batch_normalization,
                return_sequences=return_sequences,
                conv_first=False,
            )
            y = _resnet_lstm_layer(
                inputs=y,
                num_filters=num_filters_in,
                return_sequences=return_sequences,
                kernel_size=mid_kernel,
                conv_first=False,
            )
            y = _resnet_lstm_layer(
                inputs=y,
                num_filters=num_filters_out,
                return_sequences=return_sequences,
                kernel_size=1,
                conv_first=False,
            )
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims

                x = _resnet_lstm_layer(
                    inputs=x,
                    num_filters=num_filters_out,
                    kernel_size=1,
                    strides=strides,
                    activation=None,
                    return_sequences=return_sequences,
                    batch_normalization=False,
                )

            x = K.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = _resnet_lstm_layer(
        inputs=x,
        num_filters=num_filters_out,
        kernel_size=mid_kernel,
        return_sequences=False,
        conv_first=False,
    )
    x = (
        Conv2D(
            categories + nboxes * box_vector,
            kernel_size=mid_kernel,
            kernel_regularizer=regularizers.l2(reg_weight),
            padding="same",
        )
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    input_cat = Lambda(lambda x: x[:, :, :, 0:categories])(x)
    input_box = Lambda(lambda x: x[:, :, :, categories:])(x)

    output_cat = (
        Conv2D(
            categories,
            (
                round(input_shape[0] / last_conv_factor),
                round(input_shape[1] / last_conv_factor),
            ),
            activation=last_activation,
            kernel_regularizer=regularizers.l2(reg_weight),
            padding="valid",
        )
    )(input_cat)
    output_box = (
        Conv2D(
            nboxes * (box_vector),
            (
                round(input_shape[0] / last_conv_factor),
                round(input_shape[1] / last_conv_factor),
            ),
            activation="sigmoid",
            kernel_regularizer=regularizers.l2(reg_weight),
            padding="valid",
        )
    )(input_box)

    block = Concat(-1)
    outputs = block([output_cat, output_box])

    inputs = img_input

    # Create model.
    model = models.Model(inputs, outputs)

    if input_model is not None:

        model =  models.load_model(input_model)

    return model


def resnet_v2(
    input_shape,
    categories,
    box_vector,
    nboxes=1,
    stage_number=3,
    depth=38,
    start_kernel=3,
    mid_kernel=3,
    startfilter=48,
    input_model=None,
    last_activation="softmax",
):
   

    last_conv_factor = 2 ** (stage_number - 1)
    img_input = layers.Input(shape=(None, None, input_shape[2]))
    if (depth - 2) % 9 != 0:
        raise ValueError("depth should be 9n+2 (eg 56 or 110 in [b])")
    # Start model definition.
    num_filters_in = startfilter
    num_res_blocks = int((depth - 2) / 9)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = _resnet_layer(
        inputs=img_input,
        num_filters=num_filters_in,
        kernel_size=start_kernel,
        conv_first=True,
    )

    # Instantiate the stack of residual units
    for stage in range(stage_number):
        for res_block in range(num_res_blocks):

            activation = "relu"
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

            y = _resnet_layer(
                inputs=x,
                num_filters=num_filters_in,
                kernel_size=1,
                strides=strides,
                activation=activation,
                batch_normalization=batch_normalization,
                conv_first=False,
            )
            y = _resnet_layer(
                inputs=y,
                num_filters=num_filters_in,
                kernel_size=mid_kernel,
                conv_first=False,
            )
            y = _resnet_layer(
                inputs=y,
                num_filters=num_filters_out,
                kernel_size=1,
                conv_first=False,
            )
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims

                x = _resnet_layer(
                    inputs=x,
                    num_filters=num_filters_out,
                    kernel_size=1,
                    strides=strides,
                    activation=None,
                    batch_normalization=False,
                )

            x = K.layers.add([x, y])

        num_filters_in = num_filters_out

    x = (
        Conv2D(
            categories + nboxes * box_vector,
            kernel_size=mid_kernel,
            kernel_regularizer=regularizers.l2(reg_weight),
            padding="same",
        )
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    input_cat = Lambda(lambda x: x[:, :, :, 0:categories])(x)
    input_box = Lambda(lambda x: x[:, :, :, categories:])(x)

    output_cat = (
        Conv2D(
            categories,
            (
                round(input_shape[0] / last_conv_factor),
                round(input_shape[1] / last_conv_factor),
            ),
            activation=last_activation,
            kernel_regularizer=regularizers.l2(reg_weight),
            padding="valid",
        )
    )(input_cat)
    output_box = (
        Conv2D(
            nboxes * (box_vector),
            (
                round(input_shape[0] / last_conv_factor),
                round(input_shape[1] / last_conv_factor),
            ),
            activation="sigmoid",
            kernel_regularizer=regularizers.l2(reg_weight),
            padding="valid",
        )
    )(input_box)

    block = Concat(-1)
    outputs = block([output_cat, output_box])

    inputs = img_input

    # Create model.
    model = models.Model(inputs, outputs)

    if input_model is not None:

        model =  models.load_model(input_model)

    return model


def resnet_1D_regression(
    input_shape,
    stage_number=3,
    depth=38,
    start_kernel=3,
    mid_kernel=3,
    startfilter=48,
    input_model=None,
):
    """
    # Returns
        model (Model): Keras model instance
    """

    img_input = layers.Input(shape=input_shape)
    if (depth - 2) % 9 != 0:
        raise ValueError("depth should be 9n+2 (eg 56 or 110 in [b])")
    # Start model definition.
    num_filters_in = startfilter
    num_res_blocks = int((depth - 2) / 9)

    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = _resnet_layer_1D(
        inputs=img_input,
        num_filters=num_filters_in,
        kernel_size=start_kernel,
        conv_first=True,
    )

    # Instantiate the stack of residual units
    for stage in range(stage_number):
        for res_block in range(num_res_blocks):
            activation = "relu"
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

            y = _resnet_layer_1D(
                inputs=x,
                num_filters=num_filters_in,
                kernel_size=1,
                strides=strides,
                activation=activation,
                batch_normalization=batch_normalization,
                conv_first=False,
            )
            y = _resnet_layer_1D(
                inputs=y,
                num_filters=num_filters_in,
                kernel_size=mid_kernel,
                conv_first=False,
            )
            y = _resnet_layer_1D(
                inputs=y,
                num_filters=num_filters_out,
                kernel_size=1,
                conv_first=False,
            )
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = _resnet_layer_1D(
                    inputs=x,
                    num_filters=num_filters_out,
                    kernel_size=1,
                    strides=strides,
                    activation=None,
                    batch_normalization=False,
                )

            x = K.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = (
        Conv2D(
            1,
            kernel_size=mid_kernel,
            kernel_regularizer=regularizers.l2(reg_weight),
            padding="same",
        )
    )(x)

    outputs = x

    inputs = img_input

    # Create model.
    model = models.Model(inputs, outputs)

    if input_model is not None:

        model =  models.load_model(input_model)

    return model


def resnet_v2_class(
    input_shape,
    categories,
    box_vector,
    nboxes=1,
    stage_number=3,
    depth=38,
    start_kernel=3,
    mid_kernel=3,
    startfilter=48,
    input_model=None,
    last_activation="softmax",
):
    

    last_conv_factor = 2 ** (stage_number - 1)
    img_input = layers.Input(shape=(None, None, input_shape[2]))
    if (depth - 2) % 9 != 0:
        raise ValueError("depth should be 9n+2 (eg 56 or 110 in [b])")
    # Start model definition.
    num_filters_in = startfilter
    num_res_blocks = int((depth - 2) / 9)

    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = _resnet_layer(
        inputs=img_input,
        num_filters=num_filters_in,
        kernel_size=start_kernel,
        conv_first=True,
    )

    # Instantiate the stack of residual units
    for stage in range(stage_number):
        for res_block in range(num_res_blocks):
            activation = "relu"
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

            y = _resnet_layer(
                inputs=x,
                num_filters=num_filters_in,
                kernel_size=1,
                strides=strides,
                activation=activation,
                batch_normalization=batch_normalization,
                conv_first=False,
            )
            y = _resnet_layer(
                inputs=y,
                num_filters=num_filters_in,
                kernel_size=mid_kernel,
                conv_first=False,
            )
            y = _resnet_layer(
                inputs=y,
                num_filters=num_filters_out,
                kernel_size=1,
                conv_first=False,
            )
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = _resnet_layer(
                    inputs=x,
                    num_filters=num_filters_out,
                    kernel_size=1,
                    strides=strides,
                    activation=None,
                    batch_normalization=False,
                )

            x = K.layers.add([x, y])

        num_filters_in = num_filters_out
    x = (
        Conv2D(
            categories + nboxes * box_vector,
            kernel_size=mid_kernel,
            kernel_regularizer=regularizers.l2(reg_weight),
            padding="same",
        )
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    input_cat = Lambda(lambda x: x[:, :, :, 0:categories])(x)
    outputs = (
        Conv2D(
            categories,
            (
                round(input_shape[0] / last_conv_factor),
                round(input_shape[1] / last_conv_factor),
            ),
            activation=last_activation,
            kernel_regularizer=regularizers.l2(reg_weight),
            padding="valid",
        )
    )(input_cat)

    inputs = img_input

    # Create model.
    model = models.Model(inputs, outputs)

    if input_model is not None:

        model =  models.load_model(input_model)

    return model


def resnet_lstm_v2_class(
    input_shape,
    categories,
    box_vector,
    nboxes=1,
    stage_number=3,
    depth=38,
    start_kernel=3,
    mid_kernel=3,
    startfilter=48,
    input_model=None,
    last_activation="softmax",
):
   

    last_conv_factor = 2 ** (stage_number - 1)
    img_input = layers.Input(shape=(None, None, input_shape[2]))
    if (depth - 2) % 9 != 0:
        raise ValueError("depth should be 9n+2 (eg 56 or 110 in [b])")
    # Start model definition.
    num_filters_in = startfilter
    num_res_blocks = int((depth - 2) / 9)

    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = _resnet_lstm_layer(
        inputs=img_input,
        num_filters=num_filters_in,
        kernel_size=start_kernel,
        conv_first=True,
    )

    # Instantiate the stack of residual units
    for stage in range(stage_number):
        for res_block in range(num_res_blocks):
            activation = "relu"
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

            y = _resnet_lstm_layer(
                inputs=x,
                num_filters=num_filters_in,
                kernel_size=1,
                strides=strides,
                activation=activation,
                batch_normalization=batch_normalization,
                conv_first=False,
            )
            y = _resnet_lstm_layer(
                inputs=y,
                num_filters=num_filters_in,
                kernel_size=mid_kernel,
                conv_first=False,
            )
            y = _resnet_lstm_layer(
                inputs=y,
                num_filters=num_filters_out,
                kernel_size=1,
                conv_first=False,
            )
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = _resnet_lstm_layer(
                    inputs=x,
                    num_filters=num_filters_out,
                    kernel_size=1,
                    strides=strides,
                    activation=None,
                    batch_normalization=False,
                )

            x = K.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling

    x = _resnet_lstm_layer(
        inputs=x,
        num_filters=num_filters_out,
        kernel_size=mid_kernel,
        return_sequences=False,
        conv_first=False,
    )
    x = (
        Conv2D(
            categories + nboxes * box_vector,
            kernel_size=mid_kernel,
            kernel_regularizer=regularizers.l2(reg_weight),
            padding="same",
        )
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    input_cat = Lambda(lambda x: x[:, :, :, 0:categories])(x)
    outputs = (
        Conv2D(
            categories,
            (
                round(input_shape[0] / last_conv_factor),
                round(input_shape[1] / last_conv_factor),
            ),
            activation=last_activation,
            kernel_regularizer=regularizers.l2(reg_weight),
            padding="valid",
        )
    )(input_cat)

    inputs = img_input

    # Create model.
    model = models.Model(inputs, outputs)

    if input_model is not None:

        model =  models.load_model(input_model)

    return model


def _resnet_3d_lstm_layer(
    inputs,
    num_filters=64,
    kernel_size=3,
    strides=1,
    return_sequences=True,
    activation="relu",
    batch_normalization=True,
    conv_first=True,
):

    conv_lstm_3d = ConvLSTM2D(
        filters=num_filters,
        kernel_size=(kernel_size, kernel_size),
        activation=activation,
        strides=strides,
        data_format="channels_last",
        return_sequences=return_sequences,
        padding="same",
    )
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


def _resnet_lstm_layer(
    inputs,
    num_filters=64,
    kernel_size=3,
    strides=1,
    activation="relu",
    return_sequences=True,
    batch_normalization=True,
    conv_first=True,
):

    conv_lstm = ConvLSTM1D(
        filters=num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        kernel_initializer="he_normal",
        return_sequences=return_sequences,
        kernel_regularizer=regularizers.l2(1e-4),
    )

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


def _resnet_layer(
    inputs,
    num_filters=64,
    kernel_size=3,
    strides=1,
    activation="relu",
    batch_normalization=True,
    conv_first=True,
):
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
    conv = Conv2D(
        num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(1e-4),
    )

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


def _resnet_layer_1D(
    inputs,
    num_filters=64,
    kernel_size=3,
    strides=1,
    activation="relu",
    batch_normalization=True,
    conv_first=True,
):
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
    conv = Conv1D(
        num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(1e-4),
    )

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
    return 