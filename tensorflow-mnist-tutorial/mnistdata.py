# encoding: UTF-8
# Copyright 2018 Google.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import numpy as np

import sys
sys.path.append("mlengine")
from trainer.task import load_mnist_data  # import from mlengine/trainer/task.py
from trainer.task import load_dataset     # import from mlengine/trainer/task.py

# This loads entire dataset to an in-memory numpy array.
# This uses tf.data.Dataset to avoid duplicating code.
# Normally, if you already have a tf.data.Dataset, loading
# it to memory is not useful. The goal here is educational:
# teach about neural network basics without having to
# explain tf.data.Dataset now. The concept will be introduced
# later.
# The proper way of using tf.data.Dataset is to call
# features, labels = tf_dataset.make_one_shot_iterator().get_next()
# and then to use "features" and "labels" in your Tensorflow
# model directly. These tensorflow nodes, when executed, will
# automatically trigger the loading of the next batch of data.
# The sample that uses tf.data.Dataset correctly is in mlengine/trainer.

class MnistData(object):

    def __init__(self, tf_dataset, one_hot, reshape):
        self.pos = 0
        self.images = None
        self.labels = None
        # load entire Dataset into memory by chunks of 10000
        tf_dataset = tf_dataset.batch(10000)
        tf_dataset = tf_dataset.repeat(1)
        features, labels = tf_dataset.make_one_shot_iterator().get_next()
        if not reshape:
            features = tf.reshape(features, [-1, 28, 28, 1])
        if one_hot:
            labels = tf.one_hot(labels, 10)
        with tf.Session() as sess:
            while True:
                try:
                    feats, labs = sess.run([features, labels])
                    self.images = feats if self.images is None else np.concatenate([self.images, feats])
                    self.labels = labs if self.labels is None else np.concatenate([self.labels, labs])
                except tf.errors.OutOfRangeError:
                    break


    def next_batch(self, batch_size):
        if self.pos+batch_size > len(self.images) or self.pos+batch_size > len(self.labels):
            self.pos = 0
        res = (self.images[self.pos:self.pos+batch_size], self.labels[self.pos:self.pos+batch_size])
        self.pos += batch_size
        return res


class Mnist(object):
    def __init__(self, train_dataset, test_dataset, one_hot, reshape):
        self.train = MnistData(train_dataset, one_hot, reshape)
        self.test = MnistData(test_dataset, one_hot, reshape)


def read_data_sets(data_dir, one_hot, reshape):
    train_images_file, train_labels_file, test_images_file, test_labels_file = load_mnist_data(data_dir)
    train_dataset = load_dataset(train_images_file, train_labels_file)
    train_dataset = train_dataset.shuffle(60000)
    test_dataset = load_dataset(test_images_file, test_labels_file)
    mnist = Mnist(train_dataset, test_dataset, one_hot, reshape)
    return mnist
