from keras.models import *
from keras.layers import *

import os

file_path = os.path.dirname(os.path.abspath(__file__))
th_weights = file_path + "/../data/vgg16_weights_th_dim_ordering_th_kernels.h5"
tf_weights = file_path + "/../data/vgg16_weights_tf_dim_ordering_tf_kernels.h5"


def VGG(input_height=416, input_width=608, data_format="channels_first"):
    if data_format == 'channels_first':
        img_input = Input(shape=(3, input_height, input_width))
    elif data_format == 'channels_last':
        img_input = Input(shape=(input_height, input_width, 3))
    else:
        assert data_format == 'channels_first' or data_format == 'channels_last'

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=data_format)(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=data_format)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=data_format)(x)
    f1 = x
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=data_format)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=data_format)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=data_format)(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=data_format)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=data_format)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=data_format)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=data_format)(x)
    f3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=data_format)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=data_format)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=data_format)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=data_format)(x)
    f4 = x

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=data_format)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=data_format)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=data_format)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=data_format)(x)
    f5 = x

    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(1000, activation='softmax', name='predictions')(x)

    return f1, f2, f3, f4, f5, img_input, Model(img_input, x)


def VGGSegnet(n_classes, input_height=416, input_width=608, vgg_level=3, data_format="channels_first"):
    f1, f2, f3, f4, f5, img_input, vgg = VGG(input_height=input_height, input_width=input_width, data_format=data_format)
    if data_format == 'channels_first':
        vgg.load_weights(th_weights)
    elif data_format == 'channels_last':
        vgg.load_weights(tf_weights)
    else:
        assert data_format == 'channels_first' or data_format == 'channels_last'

    levels = [f1, f2, f3, f4, f5]

    o = levels[vgg_level]

    o = (ZeroPadding2D((1, 1), data_format=data_format))(o)
    o = (Conv2D(512, (3, 3), padding='valid', data_format=data_format))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=data_format))(o)
    o = (ZeroPadding2D((1, 1), data_format=data_format))(o)
    o = (Conv2D(256, (3, 3), padding='valid', data_format=data_format))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=data_format))(o)
    o = (ZeroPadding2D((1, 1), data_format=data_format))(o)
    o = (Conv2D(128, (3, 3), padding='valid', data_format=data_format))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=data_format))(o)
    o = (ZeroPadding2D((1, 1), data_format=data_format))(o)
    o = (Conv2D(64, (3, 3), padding='valid', data_format=data_format))(o)
    o = (BatchNormalization())(o)

    o = Conv2D(n_classes, (3, 3), padding='same', data_format=data_format)(o)
    o_shape = Model(img_input, o).output_shape

    if data_format == 'channels_first':
        outputHeight = o_shape[2]
        outputWidth = o_shape[3]
    else:
        print(o_shape)
        outputHeight = o_shape[1]
        outputWidth = o_shape[2]

    o = (Reshape((-1, outputHeight * outputWidth)))(o)
    o = (Permute((2, 1)))(o)
    o = (Activation('softmax'))(o)
    model = Model(img_input, o)
    model.outputWidth = outputWidth
    model.outputHeight = outputHeight

    return model


if __name__ == '__main__':
    m = VGGSegnet(101)
    from keras.utils import plot_model

    plot_model(m, show_shapes=True, to_file='model.png')
