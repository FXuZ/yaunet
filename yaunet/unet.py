#!/usr/bin/env python3

import os
import csv
import tensorflow as tf
import numpy as np
import pylab

def build_net(depth=5):
    img = tf.placeholder(dtype=tf.float32, shape=(None, 256, 256, 1))
    inputs = {}
    outputs = {}
    # Down-convolve
    layers = []
    n_channel = 64
    layer1_0 = tf.layers.conv2d(img, filters=n_channel, kernel_size=3,
                                strides=(1, 1), padding='same', activation=tf.nn.relu)
    layer1_1 = tf.layers.conv2d(layer1_0, filters=n_channel, kernel_size=3,
                                strides=(1, 1), padding='same', activation=tf.nn.relu)

    layers.append([img, layer1_0, layer1_1])

    for _ in range(depth - 1):
        n_channel = n_channel << 1
        layer0 = tf.layers.max_pooling2d(layers[-1][-1], pool_size=2, padding='same', strides=2)
        layer1 = tf.layers.conv2d(layer0, filters=n_channel, kernel_size=3,
                                  strides=(1, 1), padding='same', activation=tf.nn.relu)
        layer2 = tf.layers.conv2d(layer1, filters=n_channel, kernel_size=3,
                                  strides=(1, 1), padding='same', activation=tf.nn.relu)
        layers.append([layer0, layer1, layer2])

    # Up-convolve

    for i in range(depth - 1):
        n_channel = n_channel >> 1
        up_layer0 = tf.layers.conv2d_transpose(layers[-1][-1], filters=n_channel,
                                               kernel_size=2, strides=(2, 2))
        up_layer1 = tf.concat([layers[depth - i - 2][-1], up_layer0], axis=3)
        up_layer2 = tf.layers.conv2d(up_layer1, filters=n_channel,
                                     kernel_size=3, strides=(1, 1), padding='same')
        up_layer3 = tf.layers.conv2d(up_layer2, filters=n_channel,
                                     kernel_size=3, strides=(1, 1), padding='same')
        layers.append([up_layer1, up_layer2, up_layer3])

    logits = tf.layers.conv2d(layers[-1][-1], filters=2, kernel_size=1, strides=(1, 1), padding='same')
    output = tf.nn.softmax(logits, axis=3)
    truth = tf.placeholder(dtype=tf.int16, shape=(None, 256, 256, 2))
    loss = tf.losses.softmax_cross_entropy(onehot_labels=truth, logits=output)
    inputs['img'] = img
    inputs['label'] = truth
    outputs['logits'] = logits
    outputs['loss'] = loss
    return inputs, outputs, layers

def predict(img, output, img_in, sess):
    # img_in should be the input data tensor
    seg = tf.argmax(output, axis=3)
    return sess.run(seg, feed_dict={img: img_in})

def train_step(img, label, loss, img_in, label_in, sess):
    optim = tf.train.AdamOptimizer(learning_rate=0.0004)
    tf.global_variables_initializer().run()
    sess.run(optim.minimize(loss), feed_dict={img: img_in, label: label_in})

def get_image_data(path, csv_file):
    img = {}
    for iid in os.listdir(path):
        img_path = os.path.join(os.path.join(path, iid), 'images')
        img[iid] = {'image': pylab.imread(os.path.join(img_path, iid) + '.png')}

    def run_length_decode(enc, shape):
        mask = np.zeros(shape).flatten()
        for i in range(len(enc) << 1):
            start = int(enc[i])
            length = int(enc[i+1])
            mask[start: start + length] = np.ones(length)
        return mask.reshape(shape)

    with open(csv_file) as cf:
        label_enc = csv.reader(cf, delimiter=',')
        next(label_enc)
        for row in label_enc:
            iid = row[0]
            inst = img[iid]
            print(len(row[1].split(' ')))
            inst['mask'] = run_length_decode(row[1].split(' '), inst['image'].shape)

    return img

def main():
    with tf.Session() as sess:
        inputs, outputs, layers = build_net()

if __name__ == '__main__':
    data = get_image_data('../../training', '../../stage1_train_labels.csv')
