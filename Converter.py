"""
https://github.com/keras-team/keras/wiki/Converting-convolution-kernels-from-Theano-to-TensorFlow-and-vice-versa
"""

from keras import backend as K
from keras.utils.conv_utils import convert_kernel
import tensorflow as tf

from Models import VGGSegnet


def convertTheano2Tensorflow(model, theano_weights_file, output_weights_file):
    """
    :param model:
    :param theano_weights_file:
    :param output_weights_file:
    :return:
    """
    model.load_weights(theano_weights_file)
    ops = []
    for layer in model.layers:
        if layer.__class__.__name__ in ['Convolution1D', 'Convolution2D', 'Convolution3D', 'AtrousConvolution2D']:
            original_w = K.get_value(layer.W)
            converted_w = convert_kernel(original_w)
            ops.append(tf.assign(layer.W, converted_w).op)
    K.get_session().run(ops)
    model.save_weights(output_weights_file)


def convertTensorflow2Theano(model, tensorflow_weights_file, output_weights_file):
    """
    TODO untested!!
    :param model:
    :param tensorflow_weights_file:
    :param output_weights_file:
    :return:
    """
    model.load_weights(tensorflow_weights_file)
    for layer in model.layers:
        if layer.__class__.__name__ in ['Convolution1D', 'Convolution2D']:
            original_w = K.get_value(layer.W)
            converted_w = convert_kernel(original_w)
            K.set_value(layer.W, converted_w)
    model.save_weights(output_weights_file)


if __name__ == '__main__':
    f1, f2, f3, f4, f5, img_input, vgg = VGGSegnet.VGG(
        input_height=224,
        input_width=224
    )
    convertTheano2Tensorflow(
        model=vgg,
        theano_weights_file='data/vgg16_weights_th_dim_ordering_th_kernels.h5',
        output_weights_file='data/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    )
