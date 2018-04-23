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

from __future__ import print_function
import re
import math
import tensorflow as tf
import trainer_yolo.digits as dd
from trainer_yolo import boxutils
from trainer_yolo.boxutils import IOUCalculator
from tensorflow.python.platform import tf_logging as logging

TILE_SIZE = 256  # must be the same as in train.py
MAX_DETECTED_ROIS_PER_TILE = 60  # can be different from MAX_TARGET_ROIS_PER_TILE in train.py. The max possible is GRID_N * GRID_N * CELL_B.

def get_bottom_left_digits(classes):
    digits = tf.image.grayscale_to_rgb(dd.digits_bottom_left(TILE_SIZE//8, TILE_SIZE//8))
    digits = tf.image.resize_bilinear(digits, [TILE_SIZE, TILE_SIZE])
    return tf.gather(digits, tf.minimum(classes,9))  # correct digits to be printed on the images

def get_bottom_right_digits(classes):
    digits = tf.image.grayscale_to_rgb(dd.digits_bottom_right(TILE_SIZE//8, TILE_SIZE//8))
    digits = tf.image.resize_bilinear(digits, [TILE_SIZE, TILE_SIZE])
    return tf.gather(digits, tf.minimum(classes,9))  # correct digits to be printed on the images

def get_top_right_red_white_digits(classes):
    digits = dd.digits_top_right(TILE_SIZE//8, TILE_SIZE//8)
    zeros = tf.zeros(tf.shape(digits))
    digits_red = tf.concat([digits, zeros, zeros], -1)
    digits_white = tf.concat([digits, digits, digits], -1)
    d0,d1,d2,d3,d4,d5,d6,d7,d8,d9 = tf.split(digits_red, 10)
    b0,b1,b2,b3,b4,b5,b6,b7,b8,b9 = tf.split(digits_white, 10)
    # zero is white, other digits are red
    digits = tf.concat([b0,d1,d2,d3,d4,d5,d6,d7,d8,d9], axis=0)
    digits = tf.image.resize_bilinear(digits, [TILE_SIZE, TILE_SIZE])
    return tf.gather(digits, tf.minimum(classes,9))  # correct digits to be printed on the images

def image_compose(img1, img2):
    # img1 has the real image
    # img2 has markings on a black background
    pix_r, pix_g, pix_b = tf.split(img2, 3, axis=3)
    alpha = tf.maximum(pix_r, pix_g)
    alpha = tf.maximum(pix_b, alpha)
    alpha = tf.concat([alpha, alpha, alpha], axis=3)
    return (img1*(1-alpha)+img2*alpha)

def draw_color_boxes(img, boxes, r, g, b):
    pix_r, _, _ = tf.split(img, 3, axis=3)
    black = tf.zeros_like(pix_r)
    # the Tensorflow function draw_bounding_boxes expects coordinates in the y1, x1, y2, x2 format
    white_boxes = tf.image.draw_bounding_boxes(black, boxutils.swap_xy(boxes))
    box_img = tf.concat([white_boxes * r, white_boxes * g, white_boxes * b], axis=3)
    white_boxes = tf.concat([white_boxes, white_boxes, white_boxes], axis=3)
    return tf.where(tf.greater(white_boxes, 0.0), box_img, img)

def learn_rate_decay(step, params):
    return params['lr1'] + tf.train.exponential_decay(params['lr0'], step, params['lr2'], 1/math.e)

def dropout(x, mode, params):
    # dropout mask stays constant when scanning the image in X and Y with a filter
    # in the noise_shape parameter, 1 means "keep the dropout mask the same when this dimension changes"
    noiseshape = None
    if params["spatial_dropout"]:
        noiseshape = tf.shape(x)
        noiseshape = noiseshape * tf.constant([1,0,0,1]) + tf.constant([0,1,1,0])
    y = tf.layers.dropout(x, params["dropout"], noise_shape=noiseshape, training=(mode==tf.estimator.ModeKeys.TRAIN))
    return y

def layer_batch_normalization(x, mode, params, scale=False):  # axis=-1 will work for both dense and convolutional layers
    return tf.layers.batch_normalization(x, axis=-1, momentum=params['bnexp'], epsilon=1e-5, center=True, scale=scale,
                                         training=(mode == tf.estimator.ModeKeys.TRAIN))

def layer_conv2d_batch_norm_relu_dropout(x, mode, params, filters, kernel_size, strides):
    # batch norm for relu, no need of scale adjustment in batch norm, no need for biases since batch norm centering does the same
    y = tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size, strides=strides, padding="same", activation=None, use_bias=False)
    y = layer_batch_normalization(y, mode, params)
    return dropout(tf.nn.relu(y), mode, params)

def layer_conv1x1_batch_norm(x, mode, params, depth):
    # batch norm for tanh or sigmoid, we need scale adjustment in batch norm but no need for biases since batch norm centering does the same
    y = tf.layers.conv2d(x, filters=depth, kernel_size=1, strides=1, padding="same", activation=None, use_bias=False)
    #y = layer_batch_normalization(y, mode, params, scale=True)
    return y

def layer_conv1x1(x, depth):
    # no batch norm so using biases
    y = tf.layers.conv2d(x, filters=depth, kernel_size=1, strides=1, padding="same", activation=None, use_bias=True)
    return y

def model_core_squeezenet(X, mode, params):
    # simplified small squeezenet architecture
    Y = layer_conv2d_batch_norm_relu_dropout(X, mode, params, filters=32, kernel_size=3, strides=1)
    Yl = layer_conv2d_batch_norm_relu_dropout(Y, mode, params, filters=16, kernel_size=1, strides=1)
    Yt = layer_conv2d_batch_norm_relu_dropout(Y, mode, params, filters=16, kernel_size=3, strides=1)
    Y = tf.concat([Yl, Yt], 3)
    Y = tf.layers.max_pooling2d(Y, pool_size=2, strides=2, padding="same")
    Y = layer_conv2d_batch_norm_relu_dropout(Y, mode, params, filters=28, kernel_size=1, strides=1)
    Yl = layer_conv2d_batch_norm_relu_dropout(Y, mode, params, filters=40, kernel_size=1, strides=1)
    Yt = layer_conv2d_batch_norm_relu_dropout(Y, mode, params, filters=40, kernel_size=3, strides=1)
    Y = tf.concat([Yl, Yt], 3)
    Y = tf.layers.max_pooling2d(Y, pool_size=2, strides=2, padding="same")
    Y = layer_conv2d_batch_norm_relu_dropout(Y, mode, params, filters=52, kernel_size=1, strides=1)
    Yl = layer_conv2d_batch_norm_relu_dropout(Y, mode, params, filters=64, kernel_size=1, strides=1)
    Yt = layer_conv2d_batch_norm_relu_dropout(Y, mode, params, filters=64, kernel_size=3, strides=1)
    Y = tf.concat([Yl, Yt], 3)
    Y = tf.layers.max_pooling2d(Y, pool_size=2, strides=2, padding="same")
    Y = layer_conv2d_batch_norm_relu_dropout(Y, mode, params, filters=52, kernel_size=1, strides=1)
    Yl = layer_conv2d_batch_norm_relu_dropout(Y, mode, params, filters=40, kernel_size=1, strides=1)
    Yt = layer_conv2d_batch_norm_relu_dropout(Y, mode, params, filters=40, kernel_size=3, strides=1)
    Y = tf.concat([Yl, Yt], 3)
    Y = tf.layers.max_pooling2d(Y, pool_size=2, strides=2, padding="same")
    Y = layer_conv2d_batch_norm_relu_dropout(Y, mode, params, filters=28, kernel_size=1, strides=1)
    Yl = layer_conv2d_batch_norm_relu_dropout(Y, mode, params, filters=17, kernel_size=1, strides=1)
    Yt = layer_conv2d_batch_norm_relu_dropout(Y, mode, params, filters=18, kernel_size=3, strides=1)
    Y = tf.concat([Yl, Yt], 3)
    return Y

def model_core_darknet(X, mode, params):
    Y = layer_conv2d_batch_norm_relu_dropout(X, mode, params, filters=64, kernel_size=3, strides=1)
    Y = layer_conv2d_batch_norm_relu_dropout(Y, mode, params, filters=50, kernel_size=1, strides=1)
    Y = tf.layers.max_pooling2d(Y, pool_size=2, strides=2, padding="same") # output 128x128
    Y = layer_conv2d_batch_norm_relu_dropout(Y, mode, params, filters=52, kernel_size=3, strides=1)
    Y = layer_conv2d_batch_norm_relu_dropout(Y, mode, params, filters=54, kernel_size=1, strides=1)
    Y = tf.layers.max_pooling2d(Y, pool_size=3, strides=2, padding="same") # output 64x64
    Y = layer_conv2d_batch_norm_relu_dropout(Y, mode, params, filters=56, kernel_size=3, strides=1)
    Y = layer_conv2d_batch_norm_relu_dropout(Y, mode, params, filters=58, kernel_size=1, strides=1)
    Y = tf.layers.max_pooling2d(Y, pool_size=3, strides=2, padding="same") # output 32x32
    Y = layer_conv2d_batch_norm_relu_dropout(Y, mode, params, filters=60, kernel_size=3, strides=1)
    Y = layer_conv2d_batch_norm_relu_dropout(Y, mode, params, filters=62, kernel_size=1, strides=1)
    Y = tf.layers.max_pooling2d(Y, pool_size=3, strides=2, padding="same") # output 16x16
    Y = layer_conv2d_batch_norm_relu_dropout(Y, mode, params, filters=64, kernel_size=3, strides=1)
    Y = layer_conv2d_batch_norm_relu_dropout(Y, mode, params, filters=65, kernel_size=1, strides=1)
    return Y

# Model
def model_fn_squeeze(features, labels, mode, params):

    # YOLO parameters: each tile is divided into a grid_nn x grid_nn grid,
    # each grid cell predicts cell_n ROIs.
    grid_nn = params["grid_nn"]
    cell_n = params["cell_n"]

    # model inputs
    X = features["image"]
    X = tf.to_float(X) / 255.0 # input image format is uint8

    Y = model_core_darknet(X, mode, params)

    # not needed at GRID_N=16
    # for GRID_N=16, need pool_size=1, strides=1
    # for GRID_N=8, need pool_size=2, strides=2
    # for GRID_N=4, need pool_size=4, strides=4
    pool_size = 16//grid_nn
    T = tf.layers.average_pooling2d(Y, pool_size=pool_size, strides=pool_size, padding="valid") # [batch, grid_nn, grid_nn, cell_n*32]
    # for each cell, this has CELL_B predictions of bounding box (x,y,w,c)
    # apply tanh for x, y and sigmoid for w, c
    TX0, TY0, TW0, TC00, TC01 = tf.split(T, 5, axis=-1)  # shape 4 x [batch, grid_nn, grid_nn, 16]
    TC0 = tf.concat([TC00, TC01], axis=-1)
    # TODO: idea: batch norm may be bad on this layer
    # TODO: try with a deeper layer as well
    # TODO: try a filtered convolution instead of pooling2d, maybe info from cell sides should be weighted differently
    # TODO: try softmax for predicting confidence instead of C
    TX = tf.nn.tanh(layer_conv1x1_batch_norm(TX0, mode, params, depth=cell_n))  # shape [batch, grid_nn, grid_nn, cell_n]
    TY = tf.nn.tanh(layer_conv1x1_batch_norm(TY0, mode, params, depth=cell_n))  # shape [batch, grid_nn, grid_nn, cell_n]
    TW = tf.nn.sigmoid(layer_conv1x1_batch_norm(TW0, mode, params, depth=cell_n))  # shape [batch, grid_nn, grid_nn, cell_n]
    #TC = tf.nn.sigmoid(layer_conv1x1_batch_norm(TC0, depth=CELL_B))  # shape [batch, GRID_N,GRID_N,CELL_B]
    #  2 is the number of classes: planes, or no planes
    TClogits = layer_conv1x1(TC0, depth=cell_n*2)   # shape [batch, grid_nn, grid_nn, cell_n*2]
    TClogits = tf.reshape(TClogits, [-1, grid_nn, grid_nn, cell_n, 2])
    TC = tf.nn.softmax(TClogits)                               # shape [batch, GRID_N,GRID_N,CELL_B,2]
    TC_noplane, TC_plane = tf.unstack(TC, axis=-1)
    TCsim = tf.cast(tf.argmax(TC, axis=-1), dtype=tf.float32)  # shape [batch, GRID_N,GRID_N,CELL_B]

    # leave some breathing room to the roi sizes so that rois from adjacent cells can reach into this one
    # only do this at training time to account for slightly misplaced ground truth ROIs. No need at prediction time.
    TX = TX * 1.0 * params["cell_grow"]
    TY = TY * 1.0 * params["cell_grow"]
    if mode == tf.estimator.ModeKeys.PREDICT:
        TX = tf.clip_by_value(TX, -1.0, 1.0)
        TY = tf.clip_by_value(TY, -1.0, 1.0)

    # testing different options for W
    # Woption0
    #real_TW = TW
    # Woption1
    #real_TW = TW*TW
    # Woption2
    real_TW = TW
    # Woption3
    #real_TW = tf.sqrt(TW)
    # Woption4
    #real_TW = tf.sqrt(TW)
    # Woption5
    #real_TW = TW*TW

    DETECTION_TRESHOLD = 0.5  # plane "detected" if predicted C>0.5
    detected_TW = tf.where(tf.greater(TCsim, DETECTION_TRESHOLD), real_TW, tf.zeros_like(real_TW))
    # all rois with confidence factors
    predicted_rois = tf.stack([TX,TY,real_TW], axis=-1)  # shape [batch, GRID_N, GRID_N, CELL_B, 3]
    predicted_rois = boxutils.grid_cell_to_tile_coords(predicted_rois, grid_nn, TILE_SIZE)/TILE_SIZE
    predicted_rois = tf.reshape(predicted_rois, [-1, grid_nn*grid_nn*cell_n, 4])
    predicted_C = tf.reshape(TCsim, [-1, grid_nn*grid_nn*cell_n])
    # only the rois with confidence > DETECTION_TRESHOLD
    detected_rois = tf.stack([TX,TY,detected_TW], axis=-1)  # shape [batch, GRID_N, GRID_N, CELL_B, 3]
    detected_rois = boxutils.grid_cell_to_tile_coords(detected_rois, grid_nn, TILE_SIZE)/TILE_SIZE
    detected_rois = tf.reshape(detected_rois, [-1, grid_nn*grid_nn*cell_n, 4])
    detected_rois, detected_rois_overflow = boxutils.remove_empty_rois(detected_rois, MAX_DETECTED_ROIS_PER_TILE)

    loss = train_op = eval_metrics = None
    if mode != tf.estimator.ModeKeys.PREDICT:
        ZERO_W = 0.0001
        C_ = labels["count"]
        T_ = labels["yolo_target_rois"]  # shape [4,4,3,3] = [batch, GRID_N, GRID_N, CEL_B, xyw]
        target_rois = labels["target_rois"] # shape [batch, MAX_TARGET_ROIS_PER_TILE, x1y1x2y2]
        TX_, TY_, TW_ = tf.unstack(T_, 3, axis=-1) # shape 3 x [batch, 4,4,3] = [batch, GRID_N, GRID_N, CELL_B]
        # target probability is 1 if there is a corresponding target box, 0 otherwise
        bTC_ = tf.greater(TW_, ZERO_W)
        onehotTC_ = tf.one_hot(tf.cast(bTC_, tf.int32), 2, dtype=tf.float32)
        fTC_ = tf.cast(bTC_, tf.float32) # shape [batch, 4,4,3] = [batch, GRID_N, GRID_N, CELL_B]
        #yolo_target_rois = boxutils.grid_cell_to_tile_coords(T_, grid_nn, TILE_SIZE)/TILE_SIZE
        #yolo_target_rois = tf.reshape(target_rois, [-1, grid_nn*grid_nn*cell_n, 4])

        # accuracy
        ERROR_TRESHOLD = 0.3  # plane correctly localized if predicted x,y,w within % of ground truth
        detect_correct = tf.logical_not(tf.logical_xor(tf.greater(TCsim, DETECTION_TRESHOLD), bTC_))
        ones = tf.ones(tf.shape(TW_))
        nonzero_TW_ = tf.where(bTC_, TW_, ones)
        # true if correct size where there is a plane, nonsense value where there is no plane
        size_correct = tf.less(tf.abs(real_TW - TW_) / nonzero_TW_, ERROR_TRESHOLD)
        # true if correct position where there is a plane, nonsense value where there is no plane
        position_correct = tf.less(tf.sqrt(tf.square(TX-TX_) + tf.square(TY-TY_)) / nonzero_TW_ / grid_nn, ERROR_TRESHOLD)
        truth_no_plane = tf.logical_not(bTC_)
        size_correct = tf.logical_or(size_correct, truth_no_plane)
        position_correct = tf.logical_or(position_correct, truth_no_plane)
        size_correct = tf.logical_and(detect_correct, size_correct)
        position_correct = tf.logical_and(detect_correct, position_correct)
        all_correct = tf.logical_and(size_correct, position_correct)
        mistakes = tf.reduce_sum(tf.cast(tf.logical_not(all_correct), tf.int32), axis=[1,2,3])  # shape [batch]

        # IOU (Intersection Over Union) accuracy
        iou_accuracy = IOUCalculator.batch_intersection_over_union(detected_rois*TILE_SIZE, target_rois*TILE_SIZE, SIZE=TILE_SIZE)
        iou_accuracy_overflow = tf.greater(tf.reduce_sum(detected_rois_overflow), 0)
        # check that we are not overflowing the tensor size. Issue a warning if we are. This should only happen at
        # the begining of the training with a completely uninitialized network.
        iou_accuracy = tf.cond(iou_accuracy_overflow, lambda: tf.Print(iou_accuracy, [detected_rois_overflow],
            summarize=250, message="ROI tensor overflow in IOU computation. The computed IOU is not correct and will"
            "be reported as 0. Increase MAX_DETECTED_ROIS_PER_TILE to avoid."), lambda: tf.identity(iou_accuracy))
        iou_accuracy = IOUCalculator.batch_mean(iou_accuracy)
        # set iou_accuracy to 0 if there has been any overflow in its computation
        iou_accuracy = tf.where(iou_accuracy_overflow, tf.zeros_like(iou_accuracy), iou_accuracy)

        # debug: expected and predicted counts
        debug_img = X
        #debug_img = image_compose(debug_img, get_bottom_left_digits(C_))
        #debug_img = image_compose(debug_img, get_bottom_right_digits(classes))
        debug_img = image_compose(debug_img, get_top_right_red_white_digits(mistakes))
        # debug: ground truth boxes in grey
        debug_img = draw_color_boxes(debug_img, target_rois, 0.7, 0.7, 0.7)
        # debug: computed ROIs boxes in shades of yellow
        no_box = tf.zeros(tf.shape(predicted_rois))
        select = tf.stack([predicted_C, predicted_C, predicted_C, predicted_C], axis=-1)
        select_correct = tf.reshape(all_correct, [-1, grid_nn*grid_nn*cell_n])
        select_size_correct = tf.reshape(size_correct, [-1, grid_nn*grid_nn*cell_n])
        select_position_correct = tf.reshape(position_correct, [-1, grid_nn*grid_nn*cell_n])

        select_correct = tf.stack([select_correct,select_correct,select_correct,select_correct], axis=2)
        select_size_correct = tf.stack([select_size_correct,select_size_correct,select_size_correct,select_size_correct], axis=2)
        select_position_correct = tf.stack([select_position_correct,select_position_correct,select_position_correct,select_position_correct], axis=2)

        correct_rois = tf.where(select_correct, predicted_rois, no_box)
        other_rois = tf.where(tf.logical_not(select_correct), predicted_rois, no_box)
        correct_size_rois = tf.where(select_size_correct, other_rois, no_box)
        other_rois = tf.where(tf.logical_not(select_size_correct), other_rois, no_box)
        correct_pos_rois = tf.where(select_position_correct, other_rois, no_box)
        other_rois = tf.where(tf.logical_not(select_position_correct), other_rois, no_box)
        # correct rois in yellow
        for i in range(9):
            debug_rois = tf.where(tf.greater(select, 0.1*(i+1)), correct_rois, no_box)
            debug_img = draw_color_boxes(debug_img, debug_rois, 0.1*(i+2), 0.1*(i+2), 0)
        # size only correct rois in orange
        for i in range(9):
            debug_rois = tf.where(tf.greater(select, 0.1*(i+1)), correct_size_rois, no_box)
            debug_img = draw_color_boxes(debug_img, debug_rois, 0.1*(i+2), 0.05*(i+2), 0)
        # position only correct rois in purple
        for i in range(9):
            debug_rois = tf.where(tf.greater(select, 0.1*(i+1)), correct_pos_rois, no_box)
            debug_img = draw_color_boxes(debug_img, debug_rois, 0.05*(i+2), 0, 0.1*(i+2))
        # incorrect rois in red
        for i in range(9):
            debug_rois = tf.where(tf.greater(select, 0.1*(i+1)), other_rois, no_box)
            debug_img = draw_color_boxes(debug_img, debug_rois, 0.1*(i+2), 0, 0)
        # this is apparently not useful
        #debug_img = tf.cast(debug_img*255, dtype=tf.uint8)
        tf.summary.image("input_image", debug_img, max_outputs=20)

        # model outputs
        position_loss = tf.reduce_mean(fTC_ * (tf.square(TX-TX_)+tf.square(TY-TY_)))
        #position_loss = tf.reduce_mean(TC_plane * (tf.square(TX-TX_)+tf.square(TY-TY_)))
        # YOLO trick: take square root of predicted size for loss so as not to drown errors on small boxes

        # testing different options for W
        # Woption0
        #size_loss = tf.reduce_mean(fTC_ * tf.square(tf.sqrt(TW)-tf.sqrt(TW_)) * 2)
        # Woption1
        #size_loss = tf.reduce_mean(fTC_ * tf.square(TW-tf.sqrt(TW_)) * 2)
        # Woption2

        size_loss = tf.reduce_mean(fTC_ * tf.square(TW-TW_) * 2)
        #size_loss = tf.reduce_mean(TC_plane * tf.square(TW-TW_) * 2)

        # Woption3
        #size_loss = tf.reduce_mean(fTC_ * tf.square(tf.sqrt(TW)-TW_) * 2)
        # Woption4
        #size_loss = tf.reduce_mean(fTC_ * tf.square(TW-TW_*TW_) * 2)
        # Woption5
        #size_loss = tf.reduce_mean(fTC_ * tf.square(TW*TW-TW_) * 2)

        # detection losses with regressed TC
        #obj_loss = tf.reduce_mean(fTC_ * tf.square(TC - 1))
        #noobj_loss = tf.reduce_mean((1-fTC_) * tf.square(TC - 0))

        # detection loss with softmax TC
        obj_loss = tf.losses.softmax_cross_entropy(onehotTC_, TClogits)

        # TODO: idea, add a per-cell plane/no plane detection head. Maybe it can force better gradients (?)
        # because current split of detections "per responsible bounding box" might be hard for a neural network
        # TODO: similar idea: if only one plane in cell, teach all CELL_B=cell_n detectors to detect it
        # if multiple planes, then each one its own detector. This could avoid detectors on areas with planes to be trained to detect nothing.
        # TODO: try proper softmax loss for plane/no plane prediction
        # TODO: try two or more grids, shifted by 1/2 cell size: This could make it easier to have cells detect planes in their center,
        # if that is an actual problem they have (no idea)
        # TODO: compute detection box loss agains all ROI, not just assigned ROIs: if neighboring cell detects something
        # that aligns well with ground truth, no reason to penalise
        # TODO: improve randomness in training tile selection. Currently, only one batch of random displacements, applied
        # during entire training
        # TODO: idea, try using TC instead of TC_ in position loss and size loss
        # TODO: dropout
        # TODO: one run without batch norm for comparison

        # YOLO trick: weights the different losses differently
        LW1 = params['lw1']
        LW2 = params['lw2']
        LW3 = params['lw3']
        LWT = (LW1 + LW2 + LW3)*1.0 # 1.0 needed here to convert to float
        # TODO: hyperparam tune the hell out of these loss weights
        w_obj_loss = obj_loss*(LW1/LWT)
        w_position_loss = position_loss*(LW2/LWT)
        w_size_loss = size_loss*(LW3/LWT)
        #w_noobj_loss = noobj_loss*(LW3/LWT)
        #loss = w_position_loss + w_size_loss + w_obj_loss + w_noobj_loss
        loss = w_position_loss + w_size_loss + w_obj_loss

        # average number of mistakes per image
        nb_mistakes = tf.reduce_sum(mistakes)

        lr = learn_rate_decay(tf.train.get_or_create_global_step(), params)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = tf.contrib.training.create_train_op(loss, optimizer)
        eval_metrics = {
                        "position_error": tf.metrics.mean(w_position_loss),
                        "size_error": tf.metrics.mean(w_size_loss),
                        "plane_cross_entropy_error": tf.metrics.mean(w_obj_loss),
                        #"plane_confidence_error": tf.metrics.mean(w_obj_loss),
                        #"no_plane_confidence_error": tf.metrics.mean(w_noobj_loss),
                        "mistakes": tf.metrics.mean(nb_mistakes),
                        'IOU': tf.metrics.mean(iou_accuracy)
                        }
        #debug
        tf.summary.scalar("position_error", w_position_loss)
        tf.summary.scalar("size_error", w_size_loss)
        tf.summary.scalar("plane_cross_entropy_error", w_obj_loss)
        #tf.summary.scalar("plane_confidence_error", w_obj_loss)
        #tf.summary.scalar("no_plane_confidence_error", w_noobj_loss)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("mistakes", nb_mistakes)
        tf.summary.scalar("learning_rate", lr)
        #tf.summary.histogram("detected_rois_overflow", detected_rois_overflow)
        #tf.summary.scalar("IOU", iou_accuracy) # This would run out of memory

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={"rois":predicted_rois, "rois_confidence": predicted_C},  # name these fields as you like
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metrics,
        # the visualisation GUI was coded for swapped coordinates y1x1y2x2
        export_outputs={'classes': tf.estimator.export.PredictOutput({"rois":boxutils.swap_xy(predicted_rois), "rois_confidence": predicted_C})}
    )

# Model
def model_fn_squeeze2(features, labels, mode, params):

    # YOLO parameters: each tile is divided into a grid_nn x grid_nn grid,
    # each grid cell predicts cell_n ROIs.
    grid_nn = params["grid_nn"]
    cell_n = params["cell_n"]
    base_depth = params["base_depth5"] * 5

    def dropout(x):
        # dropout mask stays constant when scanning the image in X and Y with a filter
        # in the noise_shape parameter, 1 means "keep the droput mask the same when this dimension changes"
        noiseshape = tf.shape(x)
        noiseshape = noiseshape * tf.constant([1,0,0,1]) + tf.constant([0,1,1,0])
        y = tf.layers.dropout(x, params["dropout"], noise_shape=noiseshape, training=(mode==tf.estimator.ModeKeys.TRAIN))
        return y

    def layer_conv2d_batch_norm_relu(x, filters, kernel_size, strides=1):
        y = tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size, strides=strides, padding="same", activation=None, use_bias=False)
        return dropout(tf.nn.relu(layer_batch_normalization(y, mode, params)))

    def layer_conv1x1_batch_norm(x, depth):
        y = tf.layers.conv2d(x, filters=depth, kernel_size=1, strides=1, padding="same", activation=None, use_bias=False)
        return layer_batch_normalization(y, mode, params)

    def layer_conv1x1(x, depth):
        return tf.layers.conv2d(x, filters=depth, kernel_size=1, strides=1, padding="same", activation=None, use_bias=True)

    def layer_squeeze(x, depth):
        return layer_conv2d_batch_norm_relu(x, filters=depth, kernel_size=1, strides=1)

    def layer_squeeze_pool(x, depth):
        return layer_conv2d_batch_norm_relu(x, filters=depth, kernel_size=2, strides=2)

    def layer_expand(x, depth):
        y1x1 = layer_conv2d_batch_norm_relu(x, filters=depth//2, kernel_size=1, strides=1)
        y3x3 = layer_conv2d_batch_norm_relu(x, filters=depth//2, kernel_size=3, strides=1)
        return tf.concat([y1x1, y3x3], 3)

    # TODO: test both varieties of maxpool
    # pool_size=3, strides=2 | pool_size=2, strides=2
    def layer_maxpool(x):
        return tf.layers.max_pooling2d(x, pool_size=2, strides=2, padding="same")

    # model inputs
    X = features["image"]
    X = tf.to_float(X) / 255.0 # input image format is uint8

    # simplified small squeezenet architecture
    Y = layer_conv2d_batch_norm_relu(X, filters=35, kernel_size=3, strides=1)
    Y = layer_expand(Y, 2*35)
    #   --- maxpool ---
    #Y = layer_maxpool(Y)
    Y = layer_squeeze_pool(Y, 40)
    Y = layer_expand(Y, 2*45)
    Y = layer_squeeze(Y, 50)
    Y = layer_expand(Y, 2*55)
    #   --- maxpool ---
    #Y = layer_maxpool(Y)
    Y = layer_squeeze_pool(Y, 60)
    Y = layer_expand(Y, 2*65)
    Y = layer_squeeze(Y, 70)
    Y = layer_expand(Y, 2*75)
    #   --- maxpool ---
    #Y = layer_maxpool(Y)
    Y = layer_squeeze_pool(Y, 70)
    Y = layer_expand(Y, 2*65)
    Y = layer_squeeze(Y, 60)
    Y = layer_expand(Y, 2*55)
    #   --- maxpool ---
    #Y = layer_maxpool(Y)
    Y = layer_squeeze_pool(Y, 50)
    Y = layer_expand(Y, 2*45)
    Y = layer_squeeze(Y, 40)
    Y = layer_expand(Y, 2*35)

    # YOLO bounding box head
    #
    # not needed at GRID_N=16
    # for GRID_N=16, need pool_size=1, strides=1
    # for GRID_N=8, need pool_size=2, strides=2
    # for GRID_N=4, need pool_size=4, strides=4
    pool_size = 16//grid_nn
    T = tf.layers.average_pooling2d(Y, pool_size=pool_size, strides=pool_size, padding="valid") # [batch, grid_nn, grid_nn, cell_n*32]
    # for each cell, this has CELL_B predictions of bounding box (x,y,w,c)
    # apply tanh for x, y and sigmoid for w, c
    TX0, TY0, TW0, TC00, TC01 = tf.split(T, 5, axis=-1)  # shape 4 x [batch, grid_nn, grid_nn, 16]
    TC0 = tf.concat([TC00, TC01], axis=-1)
    TX = tf.nn.tanh(layer_conv1x1(TX0, depth=cell_n))  # shape [batch, grid_nn, grid_nn, cell_n]
    TY = tf.nn.tanh(layer_conv1x1(TY0, depth=cell_n))  # shape [batch, grid_nn, grid_nn, cell_n]
    TW = tf.nn.sigmoid(layer_conv1x1(TW0, depth=cell_n))  # shape [batch, grid_nn, grid_nn, cell_n]
    #  2 is the number of classes: planes, or no planes
    TClogits = layer_conv1x1(TC0, depth=cell_n*2)   # shape [batch, grid_nn, grid_nn, cell_n*2]
    TClogits = tf.reshape(TClogits, [-1, grid_nn, grid_nn, cell_n, 2])
    TC = tf.nn.softmax(TClogits)                               # shape [batch, GRID_N,GRID_N,CELL_B,2]
    TC_noplane, TC_plane = tf.unstack(TC, axis=-1)
    TCsim = tf.cast(tf.argmax(TC, axis=-1), dtype=tf.float32)  # shape [batch, GRID_N,GRID_N,CELL_B]

    # leave some breathing room to the roi sizes so that rois from adjacent cells can reach into this one
    # only do this at training time to account for slightly misplaced ground truth ROIs. No need at prediction time.
    TX = TX * 1.0 * params["cell_grow"]
    TY = TY * 1.0 * params["cell_grow"]
    if mode == tf.estimator.ModeKeys.PREDICT:
        TX = tf.clip_by_value(TX, -1.0, 1.0)
        TY = tf.clip_by_value(TY, -1.0, 1.0)

    DETECTION_TRESHOLD = 0.5  # plane "detected" if predicted C>0.5
    detected_TW = tf.where(tf.greater(TCsim, DETECTION_TRESHOLD), TW, tf.zeros_like(TW))
    # all rois with confidence factors
    predicted_rois = tf.stack([TX,TY,TW], axis=-1)  # shape [batch, GRID_N, GRID_N, CELL_B, 3]
    predicted_rois = boxutils.grid_cell_to_tile_coords(predicted_rois, grid_nn, TILE_SIZE)/TILE_SIZE
    predicted_rois = tf.reshape(predicted_rois, [-1, grid_nn*grid_nn*cell_n, 4])
    predicted_C = tf.reshape(TCsim, [-1, grid_nn*grid_nn*cell_n])
    # only the rois with confidence > DETECTION_TRESHOLD
    detected_rois = tf.stack([TX,TY,detected_TW], axis=-1)  # shape [batch, GRID_N, GRID_N, CELL_B, 3]
    detected_rois = boxutils.grid_cell_to_tile_coords(detected_rois, grid_nn, TILE_SIZE)/TILE_SIZE
    detected_rois = tf.reshape(detected_rois, [-1, grid_nn*grid_nn*cell_n, 4])
    detected_rois, detected_rois_overflow = boxutils.remove_empty_rois(detected_rois, MAX_DETECTED_ROIS_PER_TILE)

    loss = train_op = eval_metrics = None
    if mode != tf.estimator.ModeKeys.PREDICT:
        ZERO_W = 0.0001
        C_ = labels["count"]
        T_ = labels["yolo_target_rois"]  # shape [4,4,3,3] = [batch, GRID_N, GRID_N, CEL_B, xyw]
        target_rois = labels["target_rois"] # shape [batch, MAX_TARGET_ROIS_PER_TILE, x1y1x2y2]
        TX_, TY_, TW_ = tf.unstack(T_, 3, axis=-1) # shape 3 x [batch, 4,4,3] = [batch, GRID_N, GRID_N, CELL_B]
        # target probability is 1 if there is a corresponding target box, 0 otherwise
        bTC_ = tf.greater(TW_, ZERO_W)
        onehotTC_ = tf.one_hot(tf.cast(bTC_, tf.int32), 2, dtype=tf.float32)
        fTC_ = tf.cast(bTC_, tf.float32) # shape [batch, 4,4,3] = [batch, GRID_N, GRID_N, CELL_B]

        # accuracy
        ERROR_TRESHOLD = 0.3  # plane correctly localized if predicted x,y,w within % of ground truth
        detect_correct = tf.logical_not(tf.logical_xor(tf.greater(TCsim, DETECTION_TRESHOLD), bTC_))
        ones = tf.ones(tf.shape(TW_))
        nonzero_TW_ = tf.where(bTC_, TW_, ones)
        # true if correct size where there is a plane, nonsense value where there is no plane
        size_correct = tf.less(tf.abs(TW - TW_) / nonzero_TW_, ERROR_TRESHOLD)
        # true if correct position where there is a plane, nonsense value where there is no plane
        position_correct = tf.less(tf.sqrt(tf.square(TX-TX_) + tf.square(TY-TY_)) / nonzero_TW_ / grid_nn, ERROR_TRESHOLD)
        truth_no_plane = tf.logical_not(bTC_)
        size_correct = tf.logical_or(size_correct, truth_no_plane)
        position_correct = tf.logical_or(position_correct, truth_no_plane)
        size_correct = tf.logical_and(detect_correct, size_correct)
        position_correct = tf.logical_and(detect_correct, position_correct)
        all_correct = tf.logical_and(size_correct, position_correct)
        mistakes = tf.reduce_sum(tf.cast(tf.logical_not(all_correct), tf.int32), axis=[1,2,3])  # shape [batch]

        # IOU (Intersection Over Union) accuracy
        iou_accuracy = IOUCalculator.batch_intersection_over_union(detected_rois*TILE_SIZE, target_rois*TILE_SIZE, SIZE=TILE_SIZE)
        iou_accuracy_overflow = tf.greater(tf.reduce_sum(detected_rois_overflow), 0)
        # check that we are not overflowing the tensor size. Issue a warning if we are. This should only happen at
        # the begining of the training with a completely uninitialized network.
        iou_accuracy = tf.cond(iou_accuracy_overflow, lambda: tf.Print(iou_accuracy, [detected_rois_overflow],
                                                                       summarize=250, message="ROI tensor overflow in IOU computation. The computed IOU is not correct and will"
                                                                                              "be reported as 0. Increase MAX_DETECTED_ROIS_PER_TILE to avoid."), lambda: tf.identity(iou_accuracy))
        #if mode == tf.estimator.ModeKeys.EVAL:
        #    iou_accuracy = tf.Print(iou_accuracy, [iou_accuracy], summarize=100, message="Eval IOU: ")

        iou_accuracy = IOUCalculator.batch_mean(iou_accuracy)

        #if mode == tf.estimator.ModeKeys.EVAL:
        #    iou_accuracy = tf.Print(iou_accuracy, [iou_accuracy], summarize=100, message="Eval batch average IOU: ")

        # set iou_accuracy to 0 if there has been any overflow in its computation
        iou_accuracy = tf.where(iou_accuracy_overflow, tf.zeros_like(iou_accuracy), iou_accuracy)

        # debug images
        debug_img = X
        debug_img = image_compose(debug_img, get_top_right_red_white_digits(mistakes))
        # debug: ground truth boxes in grey
        debug_img = draw_color_boxes(debug_img, target_rois, 0.7, 0.7, 0.7)
        # debug: computed ROIs boxes in shades of yellow
        no_box = tf.zeros(tf.shape(predicted_rois))
        select = tf.stack([predicted_C, predicted_C, predicted_C, predicted_C], axis=-1)
        select_correct = tf.reshape(all_correct, [-1, grid_nn*grid_nn*cell_n])
        select_size_correct = tf.reshape(size_correct, [-1, grid_nn*grid_nn*cell_n])
        select_position_correct = tf.reshape(position_correct, [-1, grid_nn*grid_nn*cell_n])

        select_correct = tf.stack([select_correct,select_correct,select_correct,select_correct], axis=2)
        select_size_correct = tf.stack([select_size_correct,select_size_correct,select_size_correct,select_size_correct], axis=2)
        select_position_correct = tf.stack([select_position_correct,select_position_correct,select_position_correct,select_position_correct], axis=2)

        correct_rois = tf.where(select_correct, predicted_rois, no_box)
        other_rois = tf.where(tf.logical_not(select_correct), predicted_rois, no_box)
        correct_size_rois = tf.where(select_size_correct, other_rois, no_box)
        other_rois = tf.where(tf.logical_not(select_size_correct), other_rois, no_box)
        correct_pos_rois = tf.where(select_position_correct, other_rois, no_box)
        other_rois = tf.where(tf.logical_not(select_position_correct), other_rois, no_box)
        # correct rois in yellow
        for i in range(9):
            debug_rois = tf.where(tf.greater(select, 0.1*(i+1)), correct_rois, no_box)
            debug_img = draw_color_boxes(debug_img, debug_rois, 0.1*(i+2), 0.1*(i+2), 0)
        # size only correct rois in orange
        for i in range(9):
            debug_rois = tf.where(tf.greater(select, 0.1*(i+1)), correct_size_rois, no_box)
            debug_img = draw_color_boxes(debug_img, debug_rois, 0.1*(i+2), 0.05*(i+2), 0)
        # position only correct rois in purple
        for i in range(9):
            debug_rois = tf.where(tf.greater(select, 0.1*(i+1)), correct_pos_rois, no_box)
            debug_img = draw_color_boxes(debug_img, debug_rois, 0.05*(i+2), 0, 0.1*(i+2))
        # incorrect rois in red
        for i in range(9):
            debug_rois = tf.where(tf.greater(select, 0.1*(i+1)), other_rois, no_box)
            debug_img = draw_color_boxes(debug_img, debug_rois, 0.1*(i+2), 0, 0)
        tf.summary.image("input_image", debug_img, max_outputs=20)

        # model outputs
        position_loss = tf.reduce_mean(fTC_ * (tf.square(TX-TX_)+tf.square(TY-TY_)))
        size_loss = tf.reduce_mean(fTC_ * tf.square(TW-TW_) * 2)
        obj_loss = tf.losses.softmax_cross_entropy(onehotTC_, TClogits)

        # YOLO trick: weights the different losses differently
        LW1 = params['lw1']
        LW2 = params['lw2']
        LW3 = params['lw3']
        LWT = (LW1 + LW2 + LW3)*1.0 # 1.0 needed here to convert to float
        w_obj_loss = obj_loss*(LW1/LWT)
        w_position_loss = position_loss*(LW2/LWT)
        w_size_loss = size_loss*(LW3/LWT)
        loss = w_position_loss + w_size_loss + w_obj_loss

        # average number of mistakes per image
        nb_mistakes = tf.reduce_sum(mistakes)

        lr = learn_rate_decay(tf.train.get_or_create_global_step(), params)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = tf.contrib.training.create_train_op(loss, optimizer)
        eval_metrics = {
            "position_error": tf.metrics.mean(w_position_loss),
            "size_error": tf.metrics.mean(w_size_loss),
            "plane_cross_entropy_error": tf.metrics.mean(w_obj_loss),
            "mistakes": tf.metrics.mean(nb_mistakes),
            'IOU': tf.metrics.mean(iou_accuracy)
        }
        #debug
        tf.summary.scalar("position_error", w_position_loss)
        tf.summary.scalar("size_error", w_size_loss)
        tf.summary.scalar("plane_cross_entropy_error", w_obj_loss)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("mistakes", nb_mistakes)
        tf.summary.scalar("learning_rate", lr)
        #tf.summary.scalar("IOU", iou_accuracy) # This would run out of memory

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={"rois":predicted_rois, "rois_confidence": predicted_C},  # name these fields as you like
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metrics,
        # the visualisation GUI was coded for swapped coordinates y1x1y2x2
        export_outputs={'classes': tf.estimator.export.PredictOutput({"rois":boxutils.swap_xy(predicted_rois), "rois_confidence": predicted_C})}
    )

# Model
def model_fn_squeeze3(features, labels, mode, params):

    # YOLO parameters: each tile is divided into a grid_nn x grid_nn grid,
    # each grid cell predicts cell_n ROIs.
    grid_nn = params["grid_nn"]
    cell_n = params["cell_n"]
    base_depth = params["first_layer_filter_depth"] # to make it continuous for hyperparam tuning
    layers_n = params["layers"]
    depth_increment = params["depth_increment"]
    first_layer_fstride = params["first_layer_filter_stride"]
    first_layer_fsize = params["first_layer_filter_size"]



    def dropout(x):
        # dropout mask stays constant when scanning the image in X and Y with a filter
        # in the noise_shape parameter, 1 means "keep the dropout mask the same when this dimension changes"
        noiseshape = None
        if params["spatial_dropout"]:
            noiseshape = tf.shape(x)
            noiseshape = noiseshape * tf.constant([1,0,0,1]) + tf.constant([0,1,1,0])
        y = tf.layers.dropout(x, params["dropout"], noise_shape=noiseshape, training=(mode==tf.estimator.ModeKeys.TRAIN))
        return y

    def layer_conv2d_batch_norm_relu(x, filters, kernel_size, strides=1):
        y = tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size, strides=strides, padding="same", activation=None, use_bias=False)
        return dropout(tf.nn.relu(layer_batch_normalization(y, mode, params)))

    def layer_conv1x1_batch_norm(x, depth):
        y = tf.layers.conv2d(x, filters=depth, kernel_size=1, strides=1, padding="same", activation=None, use_bias=False)
        return layer_batch_normalization(y, mode, params)

    def layer_conv1x1(x, depth):
        return tf.layers.conv2d(x, filters=depth, kernel_size=1, strides=1, padding="same", activation=None, use_bias=True)

    def log_and_update_layer_stats(info, layer_name, output, layers_incr, weights_incr, log_layer_number=True):
        info["layers"] += layers_incr
        info["weights"] += weights_incr
        depth = output.get_shape()[3]
        message_weights = "({:,d} weights)".format(weights_incr).rjust(20)
        message_shape = "{}x{}x{}".format(output.get_shape()[1], output.get_shape()[2], depth).ljust(14)
        message1 = "NN layer {:>2}: {:>20} -> {} {}".format(info["layers"], layer_name, message_shape, message_weights)
        message2 = "NN layer {:>24} -> {} {}".format(layer_name, message_shape, message_weights)
        logging.log(logging.INFO, message1 if log_layer_number else message2)
        return info

    def count_conv_weights(input, output, filters_size):
        return int(input.get_shape()[3] * output.get_shape()[3] * filters_size * filters_size)

    def layer_squeeze(x, depth, info):
        y = layer_conv2d_batch_norm_relu(x, filters=depth, kernel_size=1, strides=1)
        info = log_and_update_layer_stats(info, "squeeze", y, 1, count_conv_weights(x,y,1))
        return y, info

    def layer_expand(x, depth, info, last=False):
        d1 = d2 = depth//2
        if last:
            d1, d2 = boxutils.ensure_sum_divisible_by_5(d1, d2)
        y1x1 = layer_conv2d_batch_norm_relu(x, filters=d1, kernel_size=1, strides=1)
        y3x3 = layer_conv2d_batch_norm_relu(x, filters=d2, kernel_size=3, strides=1)
        y = tf.concat([y1x1, y3x3], 3)
        info = log_and_update_layer_stats(info, "expand", y, 1, count_conv_weights(x,y1x1,1) + count_conv_weights(x,y3x3,3))
        return y, info

    def layers_squeeze_expand(x, depth_increment, info, last=False):
        depth = int(x.get_shape()[3])//2
        depth += depth_increment
        x, info = layer_squeeze(x, depth, info)
        depth += depth_increment
        x, info = layer_expand(x, 2*depth, info, last=last)
        return x, info

    # TODO: test both varieties of maxpool: pool_size=3, strides=2 | pool_size=2, strides=2
    def layer_maxpool(x, info):
        y = tf.layers.max_pooling2d(x, pool_size=2, strides=2, padding="same")
        info = log_and_update_layer_stats(info, "maxpool 2x2", y, 0, 0, log_layer_number=False)
        return y, info

    # TODO: refactor this as a utility function
    # This tries to spread the layers evenly between the softmax layers
    # and also spread depth increases and decreases in a progressive way.
    # Depth at the end is the same as it was initially.
    first_expand_layer = layers_n % 2  # if the number of layers is odd, add an "expand" layer after the first one
    inner_layers_n = (layers_n - 2 - first_expand_layer) // 2
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
    logging.log(logging.INFO, "NN layers: {} + {} + {} + {} + {} + {} = {}".format(layers_n[0], layers_n[1], layers_n[2], layers_n[3], layers_n[4], layers_n[5], sum(layers_n)))

    # model inputs
    X = features["image"]
    X = tf.to_float(X) / 255.0 # input image format is uint8

    # simplified small squeezenet architecture

    #logging.log(logging.INFO, "NN layer {1:>2}: conv 3x3x{0} -> 256x256x{0}".format(base_depth*2, 1))
    info = {"layers": 0, "weights": 0}
    Y = layer_conv2d_batch_norm_relu(X, filters=base_depth, kernel_size=first_layer_fsize, strides=first_layer_fstride)
    info = log_and_update_layer_stats(info, "conv {0}x{0}x{1} stride {2}".format(first_layer_fsize, base_depth, first_layer_fstride), Y, 1, count_conv_weights(X,Y,3))

    if (first_expand_layer == 1):
        Y, info = layer_expand(Y, base_depth, info)

    Y, info = layer_maxpool(Y, info)
    for _ in range(layers_n[1]//2):
        Y, info = layers_squeeze_expand(Y, depth_increment*depth_incr_doubler[0], info)
    if (first_layer_fstride == 1):
        Y, info = layer_maxpool(Y, info)
    for _ in range(layers_n[2]//2):
        Y, info = layers_squeeze_expand(Y, depth_increment*depth_incr_doubler[1], info)
    Y, info = layer_maxpool(Y, info)
    for _ in range(layers_n[3]//2):
        Y, info = layers_squeeze_expand(Y, -depth_increment*depth_incr_doubler[2], info)
    Y, info = layer_maxpool(Y, info)
    for i in range(layers_n[4]//2, 0, -1):
        Y, info = layers_squeeze_expand(Y, -depth_increment*depth_incr_doubler[3], info, last=(i==1))

    # YOLO bounding box head
    #

    # not needed at GRID_N=16
    # for GRID_N=16, need pool_size=1, strides=1
    # for GRID_N=8, need pool_size=2, strides=2
    # for GRID_N=4, need pool_size=4, strides=4
    pool_size = 16//grid_nn
    T = tf.layers.average_pooling2d(Y, pool_size=pool_size, strides=pool_size, padding="valid") # [batch, grid_nn, grid_nn, cell_n*32]
    # for each cell, this has CELL_B predictions of bounding box (x,y,w,c)
    # apply tanh for x, y and sigmoid for w, c
    TX0, TY0, TW0, TC00, TC01 = tf.split(T, 5, axis=-1)  # shape 4 x [batch, grid_nn, grid_nn, 16]
    TC0 = tf.concat([TC00, TC01], axis=-1)
    # TODO: error here. In batch norm, scale=False works for Relu but it should be scale=True for sigmoid and tanh !
    TX = tf.nn.tanh(layer_conv1x1_batch_norm(TX0, depth=cell_n))  # shape [batch, grid_nn, grid_nn, cell_n]
    TY = tf.nn.tanh(layer_conv1x1_batch_norm(TY0, depth=cell_n))  # shape [batch, grid_nn, grid_nn, cell_n]
    TW = tf.nn.sigmoid(layer_conv1x1_batch_norm(TW0, depth=cell_n))  # shape [batch, grid_nn, grid_nn, cell_n]
    #  2 is the number of classes: planes, or no planes
    TClogits = layer_conv1x1(TC0, depth=cell_n*2)   # shape [batch, grid_nn, grid_nn, cell_n*2]

    info = log_and_update_layer_stats(info, "YOLO head - avg pool to {0}x{0} grid cells, split 5, 3 x conv 1x1x{1}, 1 x conv 1x1x{2}".format(grid_nn, cell_n, 2*cell_n), tf.concat([TX,TY,TW,TClogits], axis=-1), 1, 3*count_conv_weights(TX0, TX, 1)+count_conv_weights(TC0, TClogits, 1))
    logging.log(logging.INFO, "NN layers total weights {:,d}".format(info["weights"]))

    TClogits = tf.reshape(TClogits, [-1, grid_nn, grid_nn, cell_n, 2])
    TC = tf.nn.softmax(TClogits)                               # shape [batch, GRID_N,GRID_N,CELL_B,2]
    TC_noplane, TC_plane = tf.unstack(TC, axis=-1)
    TCsim = tf.cast(tf.argmax(TC, axis=-1), dtype=tf.float32)  # shape [batch, GRID_N,GRID_N,CELL_B]

    # leave some breathing room to the roi sizes so that rois from adjacent cells can reach into this one
    # only do this at training time to account for slightly misplaced ground truth ROIs. No need at prediction time.
    TX = TX * 1.0 * params["cell_grow"]
    TY = TY * 1.0 * params["cell_grow"]
    if mode == tf.estimator.ModeKeys.PREDICT:
        TX = tf.clip_by_value(TX, -1.0, 1.0)
        TY = tf.clip_by_value(TY, -1.0, 1.0)

    DETECTION_TRESHOLD = 0.5  # plane "detected" if predicted C>0.5
    detected_TW = tf.where(tf.greater(TCsim, DETECTION_TRESHOLD), TW, tf.zeros_like(TW))
    # all rois with confidence factors
    predicted_rois = tf.stack([TX,TY,TW], axis=-1)  # shape [batch, GRID_N, GRID_N, CELL_B, 3]
    predicted_rois = boxutils.grid_cell_to_tile_coords(predicted_rois, grid_nn, TILE_SIZE)/TILE_SIZE
    predicted_rois = tf.reshape(predicted_rois, [-1, grid_nn*grid_nn*cell_n, 4])
    predicted_C = tf.reshape(TCsim, [-1, grid_nn*grid_nn*cell_n])
    # only the rois with confidence > DETECTION_TRESHOLD
    detected_rois = tf.stack([TX,TY,detected_TW], axis=-1)  # shape [batch, GRID_N, GRID_N, CELL_B, 3]
    detected_rois = boxutils.grid_cell_to_tile_coords(detected_rois, grid_nn, TILE_SIZE)/TILE_SIZE
    detected_rois = tf.reshape(detected_rois, [-1, grid_nn*grid_nn*cell_n, 4])
    detected_rois, detected_rois_overflow = boxutils.remove_empty_rois(detected_rois, MAX_DETECTED_ROIS_PER_TILE)

    loss = train_op = eval_metrics = None
    if mode != tf.estimator.ModeKeys.PREDICT:
        ZERO_W = 0.0001
        C_ = labels["count"]
        T_ = labels["yolo_target_rois"]  # shape [4,4,3,3] = [batch, GRID_N, GRID_N, CEL_B, xyw]
        target_rois = labels["target_rois"] # shape [batch, MAX_TARGET_ROIS_PER_TILE, x1y1x2y2]
        TX_, TY_, TW_ = tf.unstack(T_, 3, axis=-1) # shape 3 x [batch, 4,4,3] = [batch, GRID_N, GRID_N, CELL_B]
        # target probability is 1 if there is a corresponding target box, 0 otherwise
        bTC_ = tf.greater(TW_, ZERO_W)
        onehotTC_ = tf.one_hot(tf.cast(bTC_, tf.int32), 2, dtype=tf.float32)
        fTC_ = tf.cast(bTC_, tf.float32) # shape [batch, 4,4,3] = [batch, GRID_N, GRID_N, CELL_B]

        # accuracy
        ERROR_TRESHOLD = 0.3  # plane correctly localized if predicted x,y,w within % of ground truth
        detect_correct = tf.logical_not(tf.logical_xor(tf.greater(TCsim, DETECTION_TRESHOLD), bTC_))
        ones = tf.ones(tf.shape(TW_))
        nonzero_TW_ = tf.where(bTC_, TW_, ones)
        # true if correct size where there is a plane, nonsense value where there is no plane
        size_correct = tf.less(tf.abs(TW - TW_) / nonzero_TW_, ERROR_TRESHOLD)
        # true if correct position where there is a plane, nonsense value where there is no plane
        position_correct = tf.less(tf.sqrt(tf.square(TX-TX_) + tf.square(TY-TY_)) / nonzero_TW_ / grid_nn, ERROR_TRESHOLD)
        truth_no_plane = tf.logical_not(bTC_)
        size_correct = tf.logical_or(size_correct, truth_no_plane)
        position_correct = tf.logical_or(position_correct, truth_no_plane)
        size_correct = tf.logical_and(detect_correct, size_correct)
        position_correct = tf.logical_and(detect_correct, position_correct)
        all_correct = tf.logical_and(size_correct, position_correct)
        mistakes = tf.reduce_sum(tf.cast(tf.logical_not(all_correct), tf.int32), axis=[1,2,3])  # shape [batch]

        # IOU (Intersection Over Union) accuracy
        iou_accuracy = IOUCalculator.batch_intersection_over_union(detected_rois*TILE_SIZE, target_rois*TILE_SIZE, SIZE=TILE_SIZE)
        iou_accuracy_overflow = tf.greater(tf.reduce_sum(detected_rois_overflow), 0)
        # check that we are not overflowing the tensor size. Issue a warning if we are. This should only happen at
        # the begining of the training with a completely uninitialized network.
        iou_accuracy = tf.cond(iou_accuracy_overflow, lambda: tf.Print(iou_accuracy, [detected_rois_overflow],
                                                                       summarize=250, message="ROI tensor overflow in IOU computation. The computed IOU is not correct and will"
                                                                                              "be reported as 0. Increase MAX_DETECTED_ROIS_PER_TILE to avoid."), lambda: tf.identity(iou_accuracy))
        iou_accuracy = IOUCalculator.batch_mean(iou_accuracy)
        # set iou_accuracy to 0 if there has been any overflow in its computation
        iou_accuracy = tf.where(iou_accuracy_overflow, tf.zeros_like(iou_accuracy), iou_accuracy)

        # debug images
        debug_img = X
        debug_img = image_compose(debug_img, get_top_right_red_white_digits(mistakes))
        # debug: ground truth boxes in grey
        debug_img = draw_color_boxes(debug_img, target_rois, 0.7, 0.7, 0.7)
        # debug: computed ROIs boxes in shades of yellow
        no_box = tf.zeros(tf.shape(predicted_rois))
        select = tf.stack([predicted_C, predicted_C, predicted_C, predicted_C], axis=-1)
        select_correct = tf.reshape(all_correct, [-1, grid_nn*grid_nn*cell_n])
        select_size_correct = tf.reshape(size_correct, [-1, grid_nn*grid_nn*cell_n])
        select_position_correct = tf.reshape(position_correct, [-1, grid_nn*grid_nn*cell_n])

        select_correct = tf.stack([select_correct,select_correct,select_correct,select_correct], axis=2)
        select_size_correct = tf.stack([select_size_correct,select_size_correct,select_size_correct,select_size_correct], axis=2)
        select_position_correct = tf.stack([select_position_correct,select_position_correct,select_position_correct,select_position_correct], axis=2)

        correct_rois = tf.where(select_correct, predicted_rois, no_box)
        other_rois = tf.where(tf.logical_not(select_correct), predicted_rois, no_box)
        correct_size_rois = tf.where(select_size_correct, other_rois, no_box)
        other_rois = tf.where(tf.logical_not(select_size_correct), other_rois, no_box)
        correct_pos_rois = tf.where(select_position_correct, other_rois, no_box)
        other_rois = tf.where(tf.logical_not(select_position_correct), other_rois, no_box)
        # correct rois in yellow
        for i in range(9):
            debug_rois = tf.where(tf.greater(select, 0.1*(i+1)), correct_rois, no_box)
            debug_img = draw_color_boxes(debug_img, debug_rois, 0.1*(i+2), 0.1*(i+2), 0)
        # size only correct rois in orange
        for i in range(9):
            debug_rois = tf.where(tf.greater(select, 0.1*(i+1)), correct_size_rois, no_box)
            debug_img = draw_color_boxes(debug_img, debug_rois, 0.1*(i+2), 0.05*(i+2), 0)
        # position only correct rois in purple
        for i in range(9):
            debug_rois = tf.where(tf.greater(select, 0.1*(i+1)), correct_pos_rois, no_box)
            debug_img = draw_color_boxes(debug_img, debug_rois, 0.05*(i+2), 0, 0.1*(i+2))
        # incorrect rois in red
        for i in range(9):
            debug_rois = tf.where(tf.greater(select, 0.1*(i+1)), other_rois, no_box)
            debug_img = draw_color_boxes(debug_img, debug_rois, 0.1*(i+2), 0, 0)
        tf.summary.image("input_image", debug_img, max_outputs=20)

        # model outputs
        position_loss = tf.reduce_mean(fTC_ * (tf.square(TX-TX_)+tf.square(TY-TY_)))
        size_loss = tf.reduce_mean(fTC_ * tf.square(TW-TW_) * 2)
        obj_loss = tf.losses.softmax_cross_entropy(onehotTC_, TClogits)

        # YOLO trick: weights the different losses differently
        LW1 = params['lw1']
        LW2 = params['lw2']
        LW3 = params['lw3']
        LWT = (LW1 + LW2 + LW3)*1.0 # 1.0 needed here to convert to float
        w_obj_loss = obj_loss*(LW1/LWT)
        w_position_loss = position_loss*(LW2/LWT)
        w_size_loss = size_loss*(LW3/LWT)
        loss = w_position_loss + w_size_loss + w_obj_loss

        # average number of mistakes per image
        nb_mistakes = tf.reduce_sum(mistakes)

        lr = learn_rate_decay(tf.train.get_or_create_global_step(), params)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = tf.contrib.training.create_train_op(loss, optimizer)
        eval_metrics = {
            "position_error": tf.metrics.mean(w_position_loss),
            "size_error": tf.metrics.mean(w_size_loss),
            "plane_cross_entropy_error": tf.metrics.mean(w_obj_loss),
            "mistakes": tf.metrics.mean(nb_mistakes),
            'IOU': tf.metrics.mean(iou_accuracy)
        }
        #debug
        tf.summary.scalar("position_error", w_position_loss)
        tf.summary.scalar("size_error", w_size_loss)
        tf.summary.scalar("plane_cross_entropy_error", w_obj_loss)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("mistakes", nb_mistakes)
        tf.summary.scalar("learning_rate", lr)
        #tf.summary.scalar("IOU", iou_accuracy) # This would run out of memory

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={"rois":predicted_rois, "rois_confidence": predicted_C},  # name these fields as you like
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metrics,
        # the visualisation GUI was coded for swapped coordinates y1x1y2x2
        export_outputs={'classes': tf.estimator.export.PredictOutput({"rois":boxutils.swap_xy(predicted_rois), "rois_confidence": predicted_C})}
    )