# todo upgrade to keras 2.0


from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.core import Layer, Activation, Reshape, Permute
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential


def segnet(nClasses, optimizer=None, input_height=360, input_width=480, data_format="channels_first"):
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2

    model = Sequential()
    if data_format == "channels_first":
        model.add(Layer(input_shape=(3, input_height, input_width)))
    else:
        model.add(Layer(input_shape=(input_height, input_width, 3)))

    # encoder
    model.add(ZeroPadding2D(padding=(pad, pad), data_format=data_format))
    model.add(Convolution2D(filter_size, kernel, kernel, border_mode='valid', data_format=data_format))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size), data_format=data_format))

    model.add(ZeroPadding2D(padding=(pad, pad), data_format=data_format))
    model.add(Convolution2D(128, kernel, kernel, border_mode='valid', data_format=data_format))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size), data_format=data_format))

    model.add(ZeroPadding2D(padding=(pad, pad)))
    model.add(Convolution2D(256, kernel, kernel, border_mode='valid', data_format=data_format))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size), data_format=data_format))

    model.add(ZeroPadding2D(padding=(pad, pad), data_format=data_format))
    model.add(Convolution2D(512, kernel, kernel, border_mode='valid', data_format=data_format))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # decoder
    model.add(ZeroPadding2D(padding=(pad, pad), data_format=data_format))
    model.add(Convolution2D(512, kernel, kernel, border_mode='valid', data_format=data_format))
    model.add(BatchNormalization())

    model.add(UpSampling2D(size=(pool_size, pool_size), data_format=data_format))
    model.add(ZeroPadding2D(padding=(pad, pad), data_format=data_format))
    model.add(Convolution2D(256, kernel, kernel, border_mode='valid', data_format=data_format))
    model.add(BatchNormalization())

    model.add(UpSampling2D(size=(pool_size, pool_size), data_format=data_format))
    model.add(ZeroPadding2D(padding=(pad, pad), data_format=data_format))
    model.add(Convolution2D(128, kernel, kernel, border_mode='valid', data_format=data_format))
    model.add(BatchNormalization())

    model.add(UpSampling2D(size=(pool_size, pool_size), data_format=data_format))
    model.add(ZeroPadding2D(padding=(pad, pad), data_format=data_format))
    model.add(Convolution2D(filter_size, kernel, kernel, border_mode='valid', data_format=data_format))
    model.add(BatchNormalization())

    model.add(Convolution2D(nClasses, 1, 1, border_mode='valid', data_format=data_format ))

    model.outputHeight = model.output_shape[-2]
    model.outputWidth = model.output_shape[-1]

    if data_format == "channels_first":
        input_shape = (nClasses, model.output_shape[-2], model.output_shape[-1])
    else:
        input_shape = (model.output_shape[-2], model.output_shape[-1], nClasses)
    model.add(Reshape((nClasses, model.output_shape[-2] * model.output_shape[-1]), input_shape=input_shape))

    model.add(Permute((2, 1)))
    model.add(Activation('softmax'))

    if not optimizer is None:
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

    return model
