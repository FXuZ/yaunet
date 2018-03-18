#!/usr/bin/env python3

import tensorflow as tf

def conv2d(xin, W, padding='VALID'):
    '''
    Wrapper for thhe tf.nn.conv2d function with some defaults
    Returns:
        The tf variable of the convolution output.
    '''
    return tf.nn.conv2d(xin, W, strides=[1, 1, 1, 1], padding=padding)

def add_conv_layer(xin, in_channel=4, out_channel=4, filter_shape=(2, 2), padding='VALID'):
    '''
    Function that creates a conv weight variable and convolves it with the input
    Args:
        xin (tf tensor): the input tensorflow 3-D tensor.
        n_channel (int): number of the convolution channels / number of parallel filters
        filter_shape (tuple): shape of the convolution filter
    Returns:
        2-element tuple of the weight and the convolution output
    '''
    W = tf.get_variable(dtype=tf.float32, shape=(filter_shape + (in_channel, out_channel)))
    return W, conv2d(xin, W, padding=padding)

def unconv2d(xin, W, padding='VALID'):
    pass

def add_unconv_layer(xin, in_chhannel=4, out_channel=4, filter_shape=(2, 2), padding='VALID'):
    pass
