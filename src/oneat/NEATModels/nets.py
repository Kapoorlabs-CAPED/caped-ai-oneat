import keras as K
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
    input_weights=None,
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
            padding="valid",
            name="yolo",
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
            padding="valid",
            name="secyolo",
        )
    )(input_box)

    block = Concat(-1)
    outputs = block([output_cat, output_box])

    inputs = img_input

    # Create model.
    model = models.Model(inputs, outputs)

    if input_weights is not None:

        model.load_weights(input_weights, by_name=True)

    return model


def VollNet(
    input_shape,
    categories,
    box_vector,
    nboxes=1,
    stage_number=3,
    depth=38,
    start_kernel=7,
    mid_kernel=3,
    startfilter=32,
    input_weights=None,
    last_activation="softmax",
):

    img_input = layers.Input(shape=(None, None, None, input_shape[3]))
    if (depth - 2) % 9 != 0:
        raise ValueError("depth should be 9n+2 (eg 56 or 110 in [b])")
    # Start model definition.
    num_filters_in = startfilter
    num_res_blocks = int((depth - 2) / 9)
    last_conv_factor = 2 ** (stage_number - 1)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_3D_layer(
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

            y = resnet_3D_layer(
                inputs=x,
                num_filters=num_filters_in,
                kernel_size=1,
                strides=strides,
                activation=activation,
                batch_normalization=batch_normalization,
                conv_first=False,
            )
            y = resnet_3D_layer(
                inputs=y,
                num_filters=num_filters_in,
                kernel_size=mid_kernel,
                conv_first=False,
            )
            y = resnet_3D_layer(
                inputs=y,
                num_filters=num_filters_out,
                kernel_size=1,
                conv_first=False,
            )
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_3D_layer(
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
        Conv3D(
            categories + nboxes * box_vector,
            kernel_size=mid_kernel,
            kernel_regularizer=regularizers.l2(reg_weight),
            padding="same",
        )
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    input_cat = Lambda(lambda x: x[:, :, :, :, 0:categories])(x)
    input_box = Lambda(lambda x: x[:, :, :, :, categories:])(x)

    output_cat = (
        Conv3D(
            categories,
            (
                round(input_shape[0] ),
                round(input_shape[1] / last_conv_factor),
                round(input_shape[2] / last_conv_factor ),
            ),
            activation=last_activation,
            kernel_regularizer=regularizers.l2(reg_weight),
            padding="valid",
            name="yolo",
        )
    )(input_cat)
    output_box = (
        Conv3D(
            nboxes * (box_vector),
            (
                round(input_shape[0]),
                round(input_shape[1] / last_conv_factor),
                round(input_shape[2] / last_conv_factor),
            ),
            activation="sigmoid",
            kernel_regularizer=regularizers.l2(reg_weight),
            padding="valid",
            name="secyolo",
        )
    )(input_box)

    block = Concat(-1)
    outputs = block([output_cat, output_box])

    inputs = img_input

    # Create model.
    model = models.Model(inputs, outputs)

    if input_weights is not None:

        model.load_weights(input_weights, by_name=True)

    return model
class DenseNet:
    def __init__(self, stage_number, growth_rate, start_kernel = 7, mid_kernel = 3,reduction = 0.5):
        self.stage_number = stage_number
        self.growth_rate = growth_rate
        self.start_kernel = start_kernel
        self.mid_kernel = mid_kernel
        self.reduction = reduction

    def __call__(self, x):
        if (self.reduction != 1.0) :
            channels = self.growth_rate * 2
        else:
            channels = 16
        x = self.first_conv3d(x, channels)
        for i, n_blocks in enumerate(self.stage_number):
            if i != 0:
                x = self.transition_layer(x)
            x = self.dense_block(x, n_blocks)
       
        return x

    def first_conv3d(self, x, channels):
        kernel_size =  self.start_kernel
        x = self._conv3d(x, channels, kernel_size)
        
        return x

    def convolution_block(self, x):
        
        return self.bn_relu_conv3d(x, self.growth_rate, self.mid_kernel)

    def dense_block(self, x, n_blocks):
        for _ in range(n_blocks):
            x = self._dense_block(x)
        return x

    def _dense_block(self, x):
        bypass = self.convolution_block(x)
        return layers.Concatenate()([x, bypass])

    def transition_layer(self, x):
        output_channels = int(x.shape[-1] * self.reduction)
        x = self.bn_relu_conv3d(x, output_channels, 1)
        return layers.MaxPooling3D((2, 2, 2))(x)

    def bn_relu_conv3d(self, x, output_channels, kernel):
        
        return self._conv3d(x, output_channels, kernel,
                            dropout_rate=self.dropout_rate)

    def _conv3d(self, x, output_channels, kernel, padding="same"):
        x = layers.Conv3D(output_channels, kernel, padding=padding, kernel_regularizer=regularizers.l2(reg_weight))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("sigmoid")(x)
        
        return x
def DenseVollNet(
                input_shape,
                categories: dict,
                box_vector,
                nboxes: int=1,
                start_kernel: int=7,
                mid_kernel: int=3,
                startfilter: int=32,
                stage_number: int = 3,
                input_weights=None,
                last_activation: str="softmax",
                depth: int=40,
                growth_rate: int=32,
                nb_filter: int=-1,
                nb_layers_per_block: dict = {'depth_0': 6, 'depth_1': 12, 'depth_2':24},
                reduction: float = 0.5
):
    
    
        # layers in each dense block
        nb_layers = []
        if type(nb_layers_per_block) is dict:
            for (k,v) in nb_layers_per_block.items():
                nb_layers.append(v) # get the list

            if len(nb_layers) != stage_number:
                raise ValueError('If `stage_number` is a list, its length must match '
                                 'the number of layers provided by `nb_layers`.')

         
        last_conv_factor = 2 ** (stage_number - 1) 
        print(input_shape, input_shape[0], input_shape[1], input_shape[2], input_shape[3])
        img_input = layers.Input(shape=(None, None, None, input_shape[3]))
        densenet = DenseNet(nb_layers, growth_rate = growth_rate, start_kernel = start_kernel, mid_kernel = mid_kernel, reduction = reduction)
        x = densenet(img_input)
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
                padding="valid",
                name="yolo",
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
                padding="valid",
                name="secyolo",
            )
        )(input_box)

        block = Concat(-1)
        outputs = block([output_cat, output_box])
        inputs = img_input
        # Create model.
        model = models.Model(inputs, outputs)

        if input_weights is not None:

            model.load_weights(input_weights, by_name=True)

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
    input_weights=None,
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

    if input_weights is not None:

        model.load_weights(input_weights, by_name=True)

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
    input_weights=None,
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

    if input_weights is not None:

        model.load_weights(input_weights, by_name=True)

    return model


def resnet_1D_regression(
    input_shape,
    stage_number=3,
    depth=38,
    start_kernel=3,
    mid_kernel=3,
    startfilter=48,
    input_weights=None,
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

    if input_weights is not None:

        model.load_weights(input_weights, by_name=True)

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
    input_weights=None,
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

    if input_weights is not None:

        model.load_weights(input_weights, by_name=True)

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
    input_weights=None,
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

    if input_weights is not None:

        model.load_weights(input_weights, by_name=True)

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
    return x






def resnet_3D_layer(
    inputs,
    num_filters=64,
    kernel_size=3,
    strides=1,
    activation="relu",
    batch_normalization=True,
    conv_first=True,
):
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
