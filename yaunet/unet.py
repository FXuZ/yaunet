#!/usr/bin/env python3

import os
import csv
import tensorflow as tf
import numpy as np

from skimage import io
from skimage.transform import resize

IMG_SIZE = (256, 256)
IMG_KEY = 'image'
MSK_KEY = 'mask'

def build_net(depth=5):
    inputs = {}
    outputs = {}
    # Down-convolve
    layers = []
    n_channel = 64
    with tf.variable_scope('down0') as scope:
        img = tf.placeholder(dtype=tf.float32, shape=(None, 256, 256, 1), name='feature')
        layer1_0 = tf.layers.conv2d(img, filters=n_channel, kernel_size=3,
                                    strides=(1, 1), padding='same', activation=tf.nn.relu)
        layer1_1 = tf.layers.conv2d(layer1_0, filters=n_channel, kernel_size=3,
                                    strides=(1, 1), padding='same', activation=tf.nn.relu)

    layers.append([img, layer1_0, layer1_1])

    for i in range(depth - 1):
        n_channel = n_channel << 1
        with tf.variable_scope('down{}'.format(i+1))as scope:
            layer0 = tf.layers.max_pooling2d(layers[-1][-1], pool_size=2, padding='same', strides=2)
            layer1 = tf.layers.conv2d(layer0, filters=n_channel, kernel_size=3,
                                      strides=(1, 1), padding='same', activation=tf.nn.relu)
            layer2 = tf.layers.conv2d(layer1, filters=n_channel, kernel_size=3,
                                      strides=(1, 1), padding='same', activation=tf.nn.relu)
        layers.append([layer0, layer1, layer2])

    # Up-convolve

    for i in range(depth - 1):
        n_channel = n_channel >> 1
        with tf.variable_scope('up{}'.format(i+1)) as scope:
            up_layer0 = tf.layers.conv2d_transpose(layers[-1][-1], filters=n_channel,
                                                kernel_size=2, strides=(2, 2))
            up_layer1 = tf.concat([layers[depth - i - 2][-1], up_layer0], axis=3)
            up_layer2 = tf.layers.conv2d(up_layer1, filters=n_channel,
                                         kernel_size=3, strides=(1, 1), padding='same', activation=tf.nn.relu)
            up_layer3 = tf.layers.conv2d(up_layer2, filters=n_channel,
                                         kernel_size=3, strides=(1, 1), padding='same', activation=tf.nn.relu)
        layers.append([up_layer1, up_layer2, up_layer3])

    with tf.variable_scope('output') as scope:
        logits = tf.layers.conv2d(layers[-1][-1], filters=2, kernel_size=1, strides=(1, 1), padding='same')
        output = tf.nn.softmax(logits, axis=3)
    truth = tf.placeholder(dtype=tf.int64, shape=(None, 256, 256), name='label')
    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(truth, 2),
                                                          logits=output))
    inputs['feature'] = img
    inputs['label'] = truth
    outputs['logits'] = logits
    outputs['loss'] = loss
    outputs['prediction'] = tf.argmax(output, axis=3)
    outputs['accuracy'] = tf.metrics.mean_iou(truth, outputs['prediction'], 2)
    return inputs, outputs, layers

def predict(img, output, img_in, sess):
    # img_in should be the input data tensor
    seg = tf.argmax(output, axis=3)
    return sess.run(seg, feed_dict={img: img_in})

def evaluate(predict, truth):
    return tf.metrics.mean_iou(predict, truth)

# def train_step(img, label, loss, img_in, label_in, sess):
#     optim = tf.train.AdamOptimizer(learning_rate=0.0004)
#     tf.global_variables_initializer().run()
#     sess.run(optim.minimize(loss), feed_dict={img: img_in, label: label_in})

def get_image_data(path, csv_file):
    img = {}
    for iid in os.listdir(path):
        img_path = os.path.join(os.path.join(path, iid), 'images')
        img[iid] = {IMG_KEY: io.imread(os.path.join(img_path, iid) + '.png', as_grey=True)}
        img[iid]['shape'] = img[iid][IMG_KEY].shape

    def run_length_decode(base, enc):
        shape = base.shape
        base = base.flatten()
        for i in range(len(enc) >> 1):
            try:
                start = int(enc[i*2])
                length = int(enc[i*2 + 1])
                base[start - 1: start + length - 1] = np.ones(length, dtype=np.int32)
            except ValueError:
                print(i, len(enc), enc, start, length, base.shape)
                raise ValueError
        return base.reshape(shape)

    with open(csv_file) as cf:
        label_enc = csv.reader(cf, delimiter=',')
        next(label_enc)
        for row in label_enc:
            iid = row[0]
            if iid in img:
                inst = img[iid]
                if 'mask' not in inst:
                    inst['mask'] = np.zeros(inst['image'].shape, dtype=np.int32)
                inst['mask'] = run_length_decode(inst['mask'], row[1].split(' '))

    return img

def import_dataset(data_dict):
    key, img_data = zip(*data_dict.items())
    feature = np.stack([img['image'] for img in img_data])
    label = np.stack([img['mask'] for img in img_data])
    while True:
        for i in range(feature.shape[0] // 10):
            yield feature[i: i+10], label[i: i+10]
    # return tf.data.Dataset.from_tensor_slices((feature, label))

def get_all_data(data_dict):
    key, img_data = zip(*data_dict.items())
    feature = np.stack([img['image'] for img in img_data])
    label = np.stack([img['mask'] for img in img_data])
    return feature, label

def resize_image(data_dict, size=IMG_SIZE):
    for key, img_data in data_dict.items():
        img_data[IMG_KEY] = resize(img_data[IMG_KEY], size).reshape(IMG_SIZE + (1,))
        mask_resize = resize(img_data[MSK_KEY], size)
        img_data[MSK_KEY] = (mask_resize > mask_resize.mean()).astype(np.int32)
    return data_dict

def main():
    data = get_image_data('../../small', '../../stage1_train_labels.csv')
    print('=== data loaded')
    train_iter = import_dataset(resize_image(data))# .prefetch(10).batch(10)
    all_feature, all_label = get_all_data(data)
    test_feature, test_label = get_all_data(resize_image(get_image_data('../../small_test/', '../../stage1_train_labels.csv')))
    inputs, outputs, layers = build_net()
    optim = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train_op = optim.minimize(outputs['loss'])
    print('=== start training')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        i = 0
        for _ in range(200):
            i += 1
            try:
                feature, label = next(train_iter)
                sess.run(train_op,
                         feed_dict={
                             inputs['feature']: feature,
                             inputs['label']: label
                         })
                if not (i % 10):
                    print("=== step {}".format(i))
                    print("loss: {}".format(sess.run(outputs['loss'], feed_dict={
                        inputs['feature']: all_feature,
                        inputs['label']: all_label
                    })))
                    print("accuracy: {}".format(sess.run(outputs['accuracy'], feed_dict={
                        inputs['feature']: test_feature,
                        inputs['label']: test_label
                    })))
            except tf.errors.OutOfRangeError:
                break

if __name__ == '__main__':
    main()
