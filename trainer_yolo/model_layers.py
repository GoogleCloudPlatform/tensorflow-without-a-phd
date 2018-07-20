"""
Copyright 2018 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
_______________________________________________________________________

Utility functions defining the specific neural network model layers used
in this model."""

import tensorflow as tf

def dropout(x, mode, params):
    """Dropout configured from command-line parameters "dropout" and "spatial_dropout".
    In "spatial_dropout" mode, the dropout mask stays constant when scanning the image
    in X and Y directions. This gives better results in convolutional layers."""

    noiseshape = None
    if params["spatial_dropout"]:
        noiseshape = tf.shape(x)  # shape [batch, x, y, filter]
        # in the noise_shape parameter, 1 means "keep the dropout mask the same when this dimension changes"
        noiseshape = noiseshape * tf.constant([1, 0, 0, 1]) + tf.constant([0, 1, 1, 0])
    return tf.layers.dropout(x, params["dropout"],
                             noise_shape=noiseshape,
                             training=(mode == tf.estimator.ModeKeys.TRAIN))


def batch_normalization(x, mode, params, scale=False):
    """Batch normalization layer parametrized from command-line parameter "bnexp". Batch
    normalization includes scaling and centering. Centering is always used and replaces
    biases. Do not use biases in your neural network layer. Scaling should be off (False)
    for activation functions that are invariant to scale (relu) and on (True) for those
    that are not (sigmoid, tanh, ...). Batch normalization is typically applied between
    a neural network layer and the activation function."""

    return tf.layers.batch_normalization(x, axis=-1, # axis=-1 will work for both dense and convolutional layers
                                         momentum=params['bnexp'],  # decay for exponential moving averages
                                         epsilon=1e-5,
                                         center=True,  # batch norm centering replaces biases in the layer
                                         scale=scale,
                                         training=(mode == tf.estimator.ModeKeys.TRAIN))


def maxpool(x):
    """Max pooling layer. Pools over 2x2 regions every 2 pixels."""
    # TODO: test both varieties of maxpool: pool_size=3, strides=2 | pool_size=2, strides=2

    return tf.layers.max_pooling2d(x, pool_size=2, strides=2, padding="same")


def maxpool_l(x, info):
    """Logged version of maxpool."""

    y = maxpool(x)
    info = _layer_stats(info, "maxpool 2x2", y, 0, 0)
    return y, info


def conv2d_batch_norm_relu_dropout(x, mode, params, filters, kernel_size, strides):
    """Convolutional layer with batch normalization and dropout."""

    y = tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size, strides=strides, padding="same",
                         activation=None,  # will be activated with RELU after batch norm
                         use_bias=False)  # no biases if using batch norm (batch norm centering has same effect)
    y = batch_normalization(y, mode, params, scale=False)  # no scaling needed with RELU
    y = tf.nn.relu(y)
    y = dropout(y, mode, params)
    return y


def conv2d_batch_norm_relu_dropout_l(x, mode, params, info, filters, kernel_size, strides):
    """Logged version of conv2d_batch_norm_relu_dropout"""

    y = conv2d_batch_norm_relu_dropout(x, mode, params, filters, kernel_size, strides)
    info = _layer_stats(info, "conv {0}x{0}x{1} stride {2}".format(kernel_size, filters, strides),
                        y, 1, _count_conv_weights(x, y, kernel_size))
    return y, info


def conv1x1_batch_norm(x, mode, params, depth):
    """1x1 convolutional layer with batch normalization. To be activated with sigmoid
    or tanh since it uses batch norm scaling"""

    y = tf.layers.conv2d(x, filters=depth, kernel_size=1, strides=1, padding="same",
                         activation=None,  # will be activated after batch norm
                         use_bias=False)  # no biases if using batch norm (batch norm centering has same effect)
    y = batch_normalization(y, mode, params, scale=True) # scaling needed for tanh or sigmoid
    return y


def conv1x1(x, depth):
    """1x1 convolutional layer. No activation or regularization."""

    return tf.layers.conv2d(x, filters=depth, kernel_size=1, strides=1, padding="same",
                            activation=None, # activated later
                            use_bias=True)  # use biases since this layer does not have batch normalization

def sqnet_squeeze(x, mode, params, info, depth):
    """Squeezenet "squeeze" layer, i.e. a 1x1 convolutional layer. """

    y = conv2d_batch_norm_relu_dropout(x, mode, params, filters=depth, kernel_size=1, strides=1)
    info = _layer_stats(info, "squeeze", y, 1, _count_conv_weights(x, y, 1))
    return y, info


def sqnet_squeeze_pool(x, mode, params, info, depth):
    """Squeezenet "squeeze" layer with stride 2, i.e. a 2x2 convolutional layer applied
    every 2 pixels."""

    y = conv2d_batch_norm_relu_dropout(x, mode, params, filters=depth, kernel_size=2, strides=2)
    info = _layer_stats(info, "squeeze stride 2", y, 1, _count_conv_weights(x, y, 2))
    return y, info


def sqnet_expand(x, mode, params, info, depth, last=False):
    """Squeezenet "expand" layer, i.e. a 1x1 convolutional layer in parallel with a 3x3
    convolutional layer. Their results are concatenated. If the 'last' parameter is set
    output depth is set to be divisible by 5 so that a YOLO head can be added right after."""

    d1 = d2 = depth//2
    if last:
        d1, d2 = _ensure_sum_divisible_by_5(d1, d2)

    y1x1 = conv2d_batch_norm_relu_dropout(x, mode, params, filters=d1, kernel_size=1, strides=1)
    y3x3 = conv2d_batch_norm_relu_dropout(x, mode, params, filters=d2, kernel_size=3, strides=1)
    y = tf.concat([y1x1, y3x3], 3)
    info = _layer_stats(info, "expand", y, 1, _count_conv_weights(x, y1x1, 1) + _count_conv_weights(x, y3x3, 3))
    return y, info


def sqnet_spread_layers(n):
    """Spreads n layers evenly between the 4 maxpool layers. Used in configurable squeezenet model."""

    first_expand_layer = n % 2  # if the number of layers is odd, add an "expand" layer after the first one
    inner_layers_n = (n - 2 - first_expand_layer) // 2
    depth_incr_doubler = [(inner_layers_n + 0) % 4 // 3 + 1,
                          (inner_layers_n + 2) % 4 // 3 + 1,
                          1,
                          1]
    layers_n = [1 + first_expand_layer,
                (inner_layers_n + 0) // 4 * 2,
                (inner_layers_n + 2) // 4 * 2,
                (inner_layers_n + 3) // 4 * 2,
                (inner_layers_n + 1) // 4 * 2,
                1]
    if layers_n[1] == 0: depth_incr_doubler[3] = 0 # adjustment needed for small number of layers
    if layers_n[2] == 0: depth_incr_doubler[2] = 0
    return layers_n, first_expand_layer, depth_incr_doubler


def _count_conv_weights(input, output, filters_size):
    """Counts convolutional layer weights for logging purposes."""

    return int(input.get_shape()[3] * output.get_shape()[3] * filters_size * filters_size)


def YOLO_head(x, mode, params, info, grid_nn, cell_n):
    """YOLO (You Look Only Once) bounding box head. Divides each image into a gid_nn x grid_nn
    grid and predicts cell_n bounding boxes per grid cell."""

    pool_size = 16//grid_nn
    # Average pooling down to the grid size.
    # for GRID_N=16, need pool_size=1, strides=1 (no pooling)
    # for GRID_N=8, need pool_size=2, strides=2
    # for GRID_N=4, need pool_size=4, strides=4
    y = tf.layers.average_pooling2d(x, pool_size=pool_size, strides=pool_size, padding="valid") # [batch, grid_nn, grid_nn, cell_n*32]

    info = _layer_stats(info, "YOLO head, avg pool", y, 0, 0)

    # for each cell, this has CELL_B predictions of bounding box (x,y,w,c)
    # apply tanh for x, y, sigmoid for w, softmax for c
    # TODO: idea: batch norm may be bad on this layer
    # TODO: try with a deeper layer as well
    # TODO: try a filtered convolution instead of pooling2d, maybe info from cell sides should be weighted differently
    box_xr, box_yr, box_wr, box_c0, box_c1 = tf.split(y, 5, axis=-1)  # shape 4 x [batch, grid_nn, grid_nn, 16]
    box_x = tf.nn.tanh(conv1x1_batch_norm(box_xr, mode, params, depth=cell_n))  # shape [batch, grid_nn, grid_nn, cell_n]
    box_y = tf.nn.tanh(conv1x1_batch_norm(box_yr, mode, params, depth=cell_n))  # shape [batch, grid_nn, grid_nn, cell_n]
    box_w = tf.nn.sigmoid(conv1x1_batch_norm(box_wr, mode, params, depth=cell_n))  # shape [batch, grid_nn, grid_nn, cell_n]
    box_c = tf.concat([box_c0, box_c1], axis=-1)
    # no batch norm before softmax
    # TODO: really no batch norm here ? What kind of batch norm could work ?
    box_c_logits = conv1x1(box_c, depth=cell_n*2)  # shape [batch, grid_nn, grid_nn, cell_n*2], 2 = number of classes, plane or not plane
    box_all = tf.concat([box_x, box_y, box_w, box_c_logits], axis=-1)

    info = _layer_stats(info, "YOLO head, box XYWC", box_all, 1,
                        3 * _count_conv_weights(box_xr, box_x, 1) + _count_conv_weights(box_c, box_c_logits, 1))

    box_c_logits = tf.reshape(box_c_logits, [-1, grid_nn, grid_nn, cell_n, 2])
    box_c = tf.nn.softmax(box_c_logits)  # shape [batch, GRID_N,GRID_N,CELL_B,2]
    #box_c_noplane, box_c_plane = tf.unstack(box_c, axis=-1)

    # Leave some breathing room to the roi sizes so that rois from adjacent cells can reach into this one.
    # This prevents training from punishing cells that do see an airplane but are not assigned any because
    # the plane is centered in an adjacent cell very close to the limit. A ground truth box that is slightly
    # off could change cell ownership of a plane while not changing anyhting about the underlying pixels.
    box_x = box_x * 1.0 * params["cell_grow"]
    box_y = box_y * 1.0 * params["cell_grow"]

    return box_x, box_y, box_w, box_c, box_c_logits, info


def _layer_stats(info, layer_name, output, layers_incr, weights_incr):
    """Internal utility function. Counts weights and layers. Produces a message. Pass in
    info=None to start counting. The return valus is an info dictionary with the following
    data:
        info["layers"]: total number of layers in the neural network (maxpool layers do not count)
        info["weights"]: total number of trainable parameters
        info['description']: description of the structure of the neural network"""

    # Initialization
    if (info == None):
        info = {'layers':0, 'weights':0, 'description':""}

    info["layers"] += layers_incr
    info["weights"] += weights_incr
    depth = output.get_shape()[3]
    message_weights = "({:,d} weights)".format(weights_incr).rjust(20)
    message_shape = "{}x{}x{}".format(output.get_shape()[1], output.get_shape()[2], depth).ljust(14)
    message1 = "NN layer {:>2}: {:>20} -> {} {}".format(info["layers"], layer_name, message_shape, message_weights)
    message2 = "NN layer {:>24} -> {} {}".format(layer_name, message_shape, message_weights)
    info['description'] += "\n"
    if layers_incr > 0:
        info['description'] += message1
    else:
        info['description'] += message2
    return info

def _ensure_sum_divisible_by_5(a, b):
    """Adjust two numbers by incrementing them until their sum is divisible by 5.
    Used in adjusting convolutional layer sizes in Squeezenet's 'expand' modules."""
    rev = False
    while (a + b) % 5 != 0:
        b, a = a+1, b
        rev = not rev
    return (b, a) if rev else (a, b)
