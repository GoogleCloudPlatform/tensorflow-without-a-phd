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
import trainer_count.digits as dd

# Model
def model_fn(features, labels, mode, params):

    def learn_rate(lr, step):
        return params['lr1'] + tf.train.exponential_decay(lr, step, params['lr2'], 1/math.e)

    def batch_normalization(x):  # axis=-1 will work for both dense and convolutional layers
        return tf.layers.batch_normalization(x, axis=-1, momentum=params['bnexp'], epsilon=1e-5, center=True, scale=False, training=(mode == tf.estimator.ModeKeys.TRAIN))

    def layer_conv2d_batch_norm_relu(x, filters, kernel_size, strides=1):
        y = tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size, strides=strides, padding="same", activation=None, use_bias=False)
        return tf.nn.relu(batch_normalization(y))

    def layer_dense_batch_norm_relu_dropout(x, size, dropout_rate):
        y = tf.layers.dense(x, size, activation=None, use_bias=False)
        z = tf.nn.relu(batch_normalization(y))
        return tf.layers.dropout(z, rate=dropout_rate, training=(mode == tf.estimator.ModeKeys.TRAIN))

    # model
    X = features["image"]
    # image format uint8
    X = tf.to_float(X) / 255.0
    Y_ = labels

    N_CLASSES = 2

    # 9 conv layers + 2 dense layers
    Y1 = layer_conv2d_batch_norm_relu(X,  filters=8, kernel_size=6, strides=1)  # 256
    Y2 = layer_conv2d_batch_norm_relu(Y1, filters=16, kernel_size=5, strides=2) # 128
    Y3 = layer_conv2d_batch_norm_relu(Y2, filters=8, kernel_size=5, strides=1)  # 128
    Y4 = layer_conv2d_batch_norm_relu(Y3, filters=16, kernel_size=4, strides=2) # 64
    Y5 = layer_conv2d_batch_norm_relu(Y4, filters=8, kernel_size=4, strides=1)  # 64
    Y6 = layer_conv2d_batch_norm_relu(Y5, filters=16, kernel_size=4, strides=2) # 32
    Y7 = layer_conv2d_batch_norm_relu(Y6, filters=8, kernel_size=4, strides=1)  # 32
    Y8 = layer_conv2d_batch_norm_relu(Y7, filters=16, kernel_size=4, strides=2) # 16
    Y9 = layer_conv2d_batch_norm_relu(Y8, filters=4, kernel_size=1, strides=1)  # 16
    Y10 = tf.layers.average_pooling2d(Y9, pool_size=4, strides=4, padding="valid") # 4x4x4
    Y11 = tf.reshape(Y10, [-1, 4*4*4])
    Ylogits = tf.layers.dense(Y11, N_CLASSES)
    predict = tf.nn.softmax(Ylogits)
    classes = tf.cast(tf.argmax(predict, 1), tf.int32)

    # debug
    digits = tf.image.grayscale_to_rgb(dd.digits_left())
    digits = tf.image.resize_bilinear(digits, [256, 256])
    expected_digit_tags = tf.gather(digits, tf.minimum(labels,9)) # correct digits to be printed on the images
    debug_img = tf.maximum(X, expected_digit_tags*0.8)
    digits = tf.image.grayscale_to_rgb(dd.digits_right())
    digits = tf.image.resize_bilinear(digits, [256, 256])
    computed_digit_tags = tf.gather(digits, classes) # computed digits to be printed on the images
    debug_img = tf.maximum(debug_img, computed_digit_tags*0.8)
    tf.summary.image("input_image", debug_img, max_outputs=10)

    # model outputs
    loss = train_op = eval_metrics = None
    if mode != tf.estimator.ModeKeys.PREDICT:
        loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(tf.one_hot(Y_,N_CLASSES), Ylogits)) * 100
        train_op = tf.contrib.layers.optimize_loss(loss, tf.train.get_global_step(), learning_rate=params['lr0'], optimizer="Adam", learning_rate_decay_fn=learn_rate)
        eval_metrics = {'accuracy': tf.metrics.accuracy(classes, Y_)}

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={"predictions": predict, "classes": classes},  # name these fields as you like
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metrics,
        export_outputs={'classes': tf.estimator.export.PredictOutput({"predictions": predict, "classes": classes})}
    )

def model_fn2(features, labels, mode, params):

    def learn_rate(lr, step):
        return params['lr1'] + tf.train.exponential_decay(lr, step, params['lr2'], 1/math.e)

    def batch_normalization(x):  # axis=-1 will work for both dense and convolutional layers
        return tf.layers.batch_normalization(x, axis=-1, momentum=params['bnexp'], epsilon=1e-5, center=True, scale=False, training=(mode == tf.estimator.ModeKeys.TRAIN))

    def layer_conv2d_batch_norm_relu(x, filters, kernel_size, strides=1):
        y = tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size, strides=strides, padding="same", activation=None, use_bias=False)
        return tf.nn.relu(batch_normalization(y))

    def layer_dense_batch_norm_relu_dropout(x, size, dropout_rate):
        y = tf.layers.dense(x, size, activation=None, use_bias=False)
        z = tf.nn.relu(batch_normalization(y))
        return tf.layers.dropout(z, rate=dropout_rate, training=(mode == tf.estimator.ModeKeys.TRAIN))

    # model
    X = features["image"]
    # image format uint8
    X = tf.to_float(X) / 255.0
    Y_ = labels

    N_CLASSES = 2

    # 9 conv layers + 2 dense layers
    Y1 = layer_conv2d_batch_norm_relu(X, filters=32, kernel_size=3, strides=1) # output 256*256*32
    Y2 = layer_conv2d_batch_norm_relu(Y1, filters=64, kernel_size=4, strides=2) # output 128*128*64
    Y3 = layer_conv2d_batch_norm_relu(Y2, filters=64, kernel_size=3, strides=1) # output 128*128*64
    Y4 = layer_conv2d_batch_norm_relu(Y3, filters=32, kernel_size=1, strides=1) # output 128*128*32
    Y5 = layer_conv2d_batch_norm_relu(Y4, filters=64, kernel_size=4, strides=2) # output 64*64*64
    Y6 = layer_conv2d_batch_norm_relu(Y5, filters=64, kernel_size=3, strides=1) # output 64*64*64
    Y7 = layer_conv2d_batch_norm_relu(Y6, filters=32, kernel_size=1, strides=1) # output 64*64*32
    Y8 = layer_conv2d_batch_norm_relu(Y7, filters=64, kernel_size=4, strides=2) # output 32*32*64
    Y9 = layer_conv2d_batch_norm_relu(Y8, filters=64, kernel_size=3, strides=1) # output 32*32*64
    Y10 = layer_conv2d_batch_norm_relu(Y9, filters=32, kernel_size=1, strides=1) # output 32*32*32
    Y11 = layer_conv2d_batch_norm_relu(Y10, filters=64, kernel_size=4, strides=2) # output 16*16*64
    Y12 = layer_conv2d_batch_norm_relu(Y11, filters=64, kernel_size=3, strides=1) # output 16*16*64
    Y13 = layer_conv2d_batch_norm_relu(Y12, filters=32, kernel_size=1, strides=1) # output 16*16*32
    Y14 = layer_conv2d_batch_norm_relu(Y13, filters=64, kernel_size=4, strides=2) # output 8*8*64
    Y15 = layer_conv2d_batch_norm_relu(Y14, filters=4, kernel_size=1, strides=1) # output 8*8*4
    Y16 = tf.layers.average_pooling2d(Y15, pool_size=4, strides=4, padding="valid") # 2x2x4
    Y17 = tf.reshape(Y16, [-1, 2*2*4])
    Ylogits = tf.layers.dense(Y17, N_CLASSES)
    predict = tf.nn.softmax(Ylogits)
    classes = tf.cast(tf.argmax(predict, 1), tf.int32)

    # debug
    digits = tf.image.grayscale_to_rgb(dd.digits_left())
    digits = tf.image.resize_bilinear(digits, [256, 256])
    expected_digit_tags = tf.gather(digits, tf.minimum(labels,9)) # correct digits to be printed on the images
    debug_img = tf.maximum(X, expected_digit_tags*0.8)
    digits = tf.image.grayscale_to_rgb(dd.digits_right())
    digits = tf.image.resize_bilinear(digits, [256, 256])
    computed_digit_tags = tf.gather(digits, classes) # computed digits to be printed on the images
    debug_img = tf.maximum(debug_img, computed_digit_tags*0.8)
    tf.summary.image("input_image", debug_img, max_outputs=10)

    # model outputs
    loss = train_op = eval_metrics = None
    if mode != tf.estimator.ModeKeys.PREDICT:
        loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(tf.one_hot(Y_,N_CLASSES), Ylogits)) * 100
        train_op = tf.contrib.layers.optimize_loss(loss, tf.train.get_global_step(), learning_rate=params['lr0'], optimizer="Adam", learning_rate_decay_fn=learn_rate)
        eval_metrics = {'accuracy': tf.metrics.accuracy(classes, Y_)}

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={"predictions": predict, "classes": classes},  # name these fields as you like
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metrics,
        export_outputs={'classes': tf.estimator.export.PredictOutput({"predictions": predict, "classes": classes})}
    )

def model_fn_squeeze(features, labels, mode, params):

    def learn_rate(lr, step):
        return params['lr1'] + tf.train.exponential_decay(lr, step, params['lr2'], 1/math.e)

    def batch_normalization(x):  # axis=-1 will work for both dense and convolutional layers
        return tf.layers.batch_normalization(x, axis=-1, momentum=params['bnexp'], epsilon=1e-5, center=True, scale=False, training=(mode == tf.estimator.ModeKeys.TRAIN))

    def layer_conv2d_batch_norm_relu(x, filters, kernel_size, strides=1):
        y = tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size, strides=strides, padding="same", activation=None, use_bias=False)
        return tf.nn.relu(batch_normalization(y))

    def layer_dense_batch_norm_relu_dropout(x, size, dropout_rate):
        y = tf.layers.dense(x, size, activation=None, use_bias=False)
        z = tf.nn.relu(batch_normalization(y))
        return tf.layers.dropout(z, rate=dropout_rate, training=(mode == tf.estimator.ModeKeys.TRAIN))

    # model
    X = features["image"]
    # image format uint8
    X = tf.to_float(X) / 255.0
    Y_ = labels

    N_CLASSES = 2

    # simplified small squeezenet architecture
    Y1 = layer_conv2d_batch_norm_relu(X, filters=32, kernel_size=6, strides=2)  # output 128x128x32
    # maxpool
    Y2 = tf.layers.max_pooling2d(Y1, pool_size=3, strides=2, padding="same")    # output 64x64x32
    Y3 = layer_conv2d_batch_norm_relu(Y2, filters=16, kernel_size=1, strides=1) #squeeze
    Y4l = layer_conv2d_batch_norm_relu(Y3, filters=32, kernel_size=1, strides=1) #expand 1x1
    Y4t = layer_conv2d_batch_norm_relu(Y3, filters=32, kernel_size=3, strides=1) #expand 3x3
    Y4 = tf.concat([Y4l, Y4t], 3)                                                # output 64x64x64
    Y5 = layer_conv2d_batch_norm_relu(Y4, filters=32, kernel_size=1, strides=1) #squeeze
    Y6l = layer_conv2d_batch_norm_relu(Y5, filters=64, kernel_size=1, strides=1) #expand 1x1
    Y6t = layer_conv2d_batch_norm_relu(Y5, filters=64, kernel_size=3, strides=1) #expand 3x3
    Y6 = tf.concat([Y6l, Y6t], 3)                                                # output 64x64x128
    # maxpool
    Y9 = tf.layers.max_pooling2d(Y6, pool_size=3, strides=2, padding="same")   # output 32x32x128
    Y10 = layer_conv2d_batch_norm_relu(Y9, filters=32, kernel_size=1, strides=1) #squeeze
    Y11l = layer_conv2d_batch_norm_relu(Y10, filters=32, kernel_size=1, strides=1) #expand 1x1
    Y11t = layer_conv2d_batch_norm_relu(Y10, filters=32, kernel_size=3, strides=1) #expand 3x3
    Y11 = tf.concat([Y11l, Y11t], 3)                                              # output 32x32x64
    Y12 = layer_conv2d_batch_norm_relu(Y11, filters=16, kernel_size=1, strides=1) #squeeze
    Y12l = layer_conv2d_batch_norm_relu(Y12, filters=16, kernel_size=1, strides=1) #expand 1x1
    Y12t = layer_conv2d_batch_norm_relu(Y12, filters=16, kernel_size=3, strides=1) #expand 3x3
    Y13 = tf.concat([Y12l, Y12t], 3)                                              # output 32x32x32
    #maxpool
    Y16 = tf.layers.max_pooling2d(Y13, pool_size=3, strides=2, padding="same")    # output 16x16x32
    Y17 = layer_conv2d_batch_norm_relu(Y16, filters=8, kernel_size=1, strides=1) #squeeze
    Y17l = layer_conv2d_batch_norm_relu(Y17, filters=8, kernel_size=1, strides=1) #expand 1x1
    Y17t = layer_conv2d_batch_norm_relu(Y17, filters=6, kernel_size=3, strides=1) #expand 3x3
    Y18 = tf.concat([Y17l, Y17t], 3)                                              # output 16x16x16
    Y19 = layer_conv2d_batch_norm_relu(Y18, filters=4, kernel_size=1, strides=1) # output 16*16*4
    Y20 = tf.layers.average_pooling2d(Y19, pool_size=4, strides=4, padding="valid") # 4x4x4
    Y21 = tf.reshape(Y20, [-1, 4*4*4])
    Ylogits = tf.layers.dense(Y21, N_CLASSES)
    predict = tf.nn.softmax(Ylogits)
    classes = tf.cast(tf.argmax(predict, 1), tf.int32)

    # debug
    digits = tf.image.grayscale_to_rgb(dd.digits_left())
    digits = tf.image.resize_bilinear(digits, [256, 256])
    expected_digit_tags = tf.gather(digits, tf.minimum(labels,9)) # correct digits to be printed on the images
    debug_img = tf.maximum(X, expected_digit_tags*0.8)
    digits = tf.image.grayscale_to_rgb(dd.digits_right())
    digits = tf.image.resize_bilinear(digits, [256, 256])
    computed_digit_tags = tf.gather(digits, classes) # computed digits to be printed on the images
    debug_img = tf.maximum(debug_img, computed_digit_tags*0.8)
    tf.summary.image("input_image", debug_img, max_outputs=10)

    # model outputs
    loss = train_op = eval_metrics = None
    if mode != tf.estimator.ModeKeys.PREDICT:
        loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(tf.one_hot(Y_,N_CLASSES), Ylogits)) * 100
        train_op = tf.contrib.layers.optimize_loss(loss, tf.train.get_global_step(), learning_rate=params['lr0'], optimizer="Adam", learning_rate_decay_fn=learn_rate)
        eval_metrics = {'accuracy': tf.metrics.accuracy(classes, Y_)}

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={"predictions": predict, "classes": classes},  # name these fields as you like
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metrics,
        export_outputs={'classes': tf.estimator.export.PredictOutput({"predictions": predict, "classes": classes})}
    )

def model_fn_squeeze2(features, labels, mode, params):

    def learn_rate(lr, step):
        return params['lr1'] + tf.train.exponential_decay(lr, step, params['lr2'], 1/math.e)

    def batch_normalization(x):  # axis=-1 will work for both dense and convolutional layers
        return tf.layers.batch_normalization(x, axis=-1, momentum=params['bnexp'], epsilon=1e-5, center=True, scale=False, training=(mode == tf.estimator.ModeKeys.TRAIN))

    def layer_conv2d_batch_norm_relu(x, filters, kernel_size, strides=1):
        y = tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size, strides=strides, padding="same", activation=None, use_bias=False)
        return tf.nn.relu(batch_normalization(y))

    def layer_dense_batch_norm_relu_dropout(x, size, dropout_rate):
        y = tf.layers.dense(x, size, activation=None, use_bias=False)
        z = tf.nn.relu(batch_normalization(y))
        return tf.layers.dropout(z, rate=dropout_rate, training=(mode == tf.estimator.ModeKeys.TRAIN))

    # model
    X = features["image"]
    # image format uint8
    X = tf.to_float(X) / 255.0
    Y_ = labels

    N_CLASSES = 2

    # simplified small squeezenet architecture
    # downsample
    Y1 = layer_conv2d_batch_norm_relu(X, filters=16, kernel_size=6, strides=2)  # output 128x128x32
    # downsample
    Y2 = layer_conv2d_batch_norm_relu(Y1, filters=32, kernel_size=3, strides=2)  # output 128x128x32

    Y3 = layer_conv2d_batch_norm_relu(Y2, filters=16, kernel_size=1, strides=1) #squeeze
    Y4l = layer_conv2d_batch_norm_relu(Y3, filters=16, kernel_size=1, strides=1) #expand 1x1
    Y4t = layer_conv2d_batch_norm_relu(Y3, filters=16, kernel_size=3, strides=1) #expand 3x3
    Y4 = tf.concat([Y4l, Y4t], 3)                                                # output 64x64x32
    Y5 = layer_conv2d_batch_norm_relu(Y4, filters=16, kernel_size=1, strides=1) #squeeze
    Y6l = layer_conv2d_batch_norm_relu(Y5, filters=16, kernel_size=1, strides=1) #expand 1x1
    Y6t = layer_conv2d_batch_norm_relu(Y5, filters=16, kernel_size=3, strides=1) #expand 3x3
    Y6 = tf.concat([Y6l, Y6t], 3)                                                # output 64x64x32

    # downsample
    Y9 = layer_conv2d_batch_norm_relu(Y6, filters=64, kernel_size=3, strides=2)  # output 32x32x64

    Y10 = layer_conv2d_batch_norm_relu(Y9, filters=32, kernel_size=1, strides=1) #squeeze
    Y11l = layer_conv2d_batch_norm_relu(Y10, filters=32, kernel_size=1, strides=1) #expand 1x1
    Y11t = layer_conv2d_batch_norm_relu(Y10, filters=32, kernel_size=3, strides=1) #expand 3x3
    Y11 = tf.concat([Y11l, Y11t], 3)                                              # output 32x32x64
    Y12 = layer_conv2d_batch_norm_relu(Y11, filters=32, kernel_size=1, strides=1) #squeeze
    Y12l = layer_conv2d_batch_norm_relu(Y12, filters=32, kernel_size=1, strides=1) #expand 1x1
    Y12t = layer_conv2d_batch_norm_relu(Y12, filters=32, kernel_size=3, strides=1) #expand 3x3
    Y13 = tf.concat([Y12l, Y12t], 3)                                              # output 32x32x64

    # downsample
    Y16 = layer_conv2d_batch_norm_relu(Y13, filters=32, kernel_size=3, strides=2)  # output 16x16x32

    Y17 = layer_conv2d_batch_norm_relu(Y16, filters=16, kernel_size=1, strides=1) #squeeze
    Y17l = layer_conv2d_batch_norm_relu(Y17, filters=16, kernel_size=1, strides=1) #expand 1x1
    Y17t = layer_conv2d_batch_norm_relu(Y17, filters=16, kernel_size=3, strides=1) #expand 3x3
    Y18 = tf.concat([Y17l, Y17t], 3)                                              # output 16x16x32
    Y19 = layer_conv2d_batch_norm_relu(Y18, filters=4, kernel_size=1, strides=1) # output 16*16*4
    Y20 = tf.layers.average_pooling2d(Y19, pool_size=4, strides=4, padding="valid") # 4x4x4
    Y21 = tf.reshape(Y20, [-1, 4*4*4])
    Ylogits = tf.layers.dense(Y21, N_CLASSES)
    predict = tf.nn.softmax(Ylogits)
    classes = tf.cast(tf.argmax(predict, 1), tf.int32)

    # debug
    digits = tf.image.grayscale_to_rgb(dd.digits_left())
    digits = tf.image.resize_bilinear(digits, [256, 256])
    expected_digit_tags = tf.gather(digits, tf.minimum(labels,9)) # correct digits to be printed on the images
    debug_img = tf.maximum(X, expected_digit_tags*0.8)
    digits = tf.image.grayscale_to_rgb(dd.digits_right())
    digits = tf.image.resize_bilinear(digits, [256, 256])
    computed_digit_tags = tf.gather(digits, classes) # computed digits to be printed on the images
    debug_img = tf.maximum(debug_img, computed_digit_tags*0.8)
    tf.summary.image("input_image", debug_img, max_outputs=10)

    # model outputs
    loss = train_op = eval_metrics = None
    if mode != tf.estimator.ModeKeys.PREDICT:
        loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(tf.one_hot(Y_,N_CLASSES), Ylogits)) * 100
        train_op = tf.contrib.layers.optimize_loss(loss, tf.train.get_global_step(), learning_rate=params['lr0'], optimizer="Adam", learning_rate_decay_fn=learn_rate)
        eval_metrics = {'accuracy': tf.metrics.accuracy(classes, Y_)}

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={"predictions": predict, "classes": classes},  # name these fields as you like
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metrics,
        export_outputs={'classes': tf.estimator.export.PredictOutput({"predictions": predict, "classes": classes})}
    )