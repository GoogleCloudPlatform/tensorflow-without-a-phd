# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import tensorflow as tf


# Model
def model_fn(features, labels, mode, params):

    # layer configurations
    filter_sizes = {'S': [4, 3, 2], 'M': [5, 4, 3], 'L': [6, 5, 4]}
    filter_size = filter_sizes[params['filter_sizes']]

    def learn_rate(lr, step):
        return params['lr1']  + tf.train.exponential_decay(lr, step, params['lr2'], 1/math.e)

    def batch_normalization(x):  # axis=-1 will work for both dense and convolutional layers
        return tf.layers.batch_normalization(x, axis=-1, momentum=params['bnexp'], epsilon=1e-5, center=True, scale=False, training=(mode == tf.estimator.ModeKeys.TRAIN))

    def layer_conv2d_relu(x, filters, kernel_size, strides=1):
        return tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size, strides=strides, padding="same", activation=tf.nn.relu, use_bias=True)

    def layer_conv2d_batch_norm_relu(x, filters, kernel_size, strides=1):
        y = tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size, strides=strides, padding="same", activation=None, use_bias=False)
        return tf.nn.relu(batch_normalization(y))

    def layer_dense_relu(x, size, dropout_rate):
        return tf.layers.dense(x, size, activation=tf.nn.relu, use_bias=True)

    def layer_dense_batch_norm_relu_dropout(x, size, dropout_rate):
        y = tf.layers.dense(x, size, activation=None, use_bias=False)
        z = tf.nn.relu(batch_normalization(y))
        return tf.layers.dropout(z, rate=dropout_rate, training=(mode == tf.estimator.ModeKeys.TRAIN))

    # model
    X = tf.reshape(features["image"], [-1, 20, 20, 3])  # reshape not necessary here
    # image format uint8
    X = tf.to_float(X) / 255.0
    Y_ = labels

    # 4 layer conv
    #Y1 = layer_conv2d_batch_norm_relu(X,  filters=params['conv1'],   kernel_size=filter_size[0], strides=1)
    #Y2 = layer_conv2d_batch_norm_relu(Y1, filters=params['conv1']*2, kernel_size=filter_size[1], strides=2)
    #Y2bis = layer_conv2d_batch_norm_relu(Y2, filters=params['conv1'], kernel_size=filter_size[2], strides=1)
    #Y3 = layer_conv2d_batch_norm_relu(Y2bis, filters=params['conv1']*2, kernel_size=filter_size[3], strides=2)
    #Y4 = tf.reshape(Y3, [-1, 2*params['conv1']*5*5])

    # 3 layer conv
    Y1 = layer_conv2d_batch_norm_relu(X,  filters=params['conv1'],   kernel_size=filter_size[0], strides=1)
    #Y1 = layer_conv2d_relu(X,  filters=params['conv1'],   kernel_size=filter_size[0], strides=1)
    Y2 = layer_conv2d_batch_norm_relu(Y1, filters=params['conv1']*2, kernel_size=filter_size[1], strides=2)
    #Y2 = layer_conv2d_relu(Y1, filters=params['conv1']*2, kernel_size=filter_size[1], strides=2)
    Y3 = layer_conv2d_batch_norm_relu(Y2, filters=params['conv1']*4, kernel_size=filter_size[2], strides=2)
    #Y3 = layer_conv2d_relu(Y2, filters=params['conv1']*4, kernel_size=filter_size[2], strides=2)
    Y4 = tf.reshape(Y3, [-1, 4*params['conv1']*5*5])

    Y5 = layer_dense_batch_norm_relu_dropout(Y4, params['dense'], params['dropout'])
    #Y5 = layer_dense_relu(Y4, params['dense'], params['dropout'])
    Ylogits = tf.layers.dense(Y5, 2)
    predict = tf.nn.softmax(Ylogits)
    classes = tf.cast(tf.argmax(predict, 1), tf.uint8)

    # model outputs
    loss = train_op = eval_metrics = None
    if mode != tf.estimator.ModeKeys.PREDICT:
        loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(tf.one_hot(Y_,2), Ylogits)) * 100
        train_op = tf.contrib.layers.optimize_loss(loss, tf.train.get_global_step(), learning_rate=params['lr0'], optimizer="Adam", learning_rate_decay_fn=learn_rate)
        eval_metrics = {'accuracy': tf.metrics.accuracy(classes, Y_)}

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={"predictions": predict, "classes": classes},  # name these fields as you like
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metrics,
        export_outputs={'classes': tf.estimator.export.PredictOutput({"predictions": predict, "classes": classes, "boxes": features["boxes"]})}
    )