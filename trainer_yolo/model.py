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
import trainer_yolo.digits as dd
from trainer_yolo import boxutils

N_CLASSES = 2
GRID_N = 16  # must be the same as in train.py
CELL_B = 2   # must be the same as in train.py
TILE_SIZE = 256  # must be the same as in train.py

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
    black = tf.zeros(tf.shape(pix_r))
    box_r = tf.image.draw_bounding_boxes(black, boxes) * r
    box_g = tf.image.draw_bounding_boxes(black, boxes) * g
    box_b = tf.image.draw_bounding_boxes(black, boxes) * b
    box_img = tf.concat([box_r, box_g, box_b], axis=3)
    return image_compose(img, box_img)

# Model
def model_fn_squeeze(features, labels, mode, params):

    def learn_rate(lr, step):
        return params['lr1'] + tf.train.exponential_decay(lr, step, params['lr2'], 1/math.e)

    def batch_normalization(x):  # axis=-1 will work for both dense and convolutional layers
        return tf.layers.batch_normalization(x, axis=-1, momentum=params['bnexp'], epsilon=1e-5, center=True, scale=False, training=(mode == tf.estimator.ModeKeys.TRAIN))

    def layer_conv2d_batch_norm_relu(x, filters, kernel_size, strides=1):
        y = tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size, strides=strides, padding="same", activation=None, use_bias=False)
        return tf.nn.relu(batch_normalization(y))

    def layer_conv2d_relu(x, filters, kernel_size, strides=1):
        return tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size, strides=strides, padding="same", activation=tf.nn.relu)

    def layer_conv1x1_batch_norm(x, depth):
        y = tf.layers.conv2d(x, filters=depth, kernel_size=1, strides=1, padding="same", activation=None, use_bias=False)
        return batch_normalization(y)

    # model inputs
    X = features["image"]
    X = tf.to_float(X) / 255.0 # input image format is uint8

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
    f=2
    g=2
    # maxpool
    Y9 = tf.layers.max_pooling2d(Y6, pool_size=3, strides=2, padding="same")   # output 32x32x128
    Y10 = layer_conv2d_batch_norm_relu(Y9, filters=32*f, kernel_size=1, strides=1) #squeeze
    Y11l = layer_conv2d_batch_norm_relu(Y10, filters=32*f, kernel_size=1, strides=1) #expand 1x1
    Y11t = layer_conv2d_batch_norm_relu(Y10, filters=32*f, kernel_size=3, strides=1) #expand 3x3
    Y11 = tf.concat([Y11l, Y11t], 3)                                              # output 32x32x64
    Y12 = layer_conv2d_batch_norm_relu(Y11, filters=16*2*g*f, kernel_size=1, strides=1) #squeeze
    Y12l = layer_conv2d_batch_norm_relu(Y12, filters=16*2*g*f, kernel_size=1, strides=1) #expand 1x1
    Y12t = layer_conv2d_batch_norm_relu(Y12, filters=16*2*g*f, kernel_size=3, strides=1) #expand 3x3
    Y13 = tf.concat([Y12l, Y12t], 3)                                              # output 32x32x32
    #maxpool
    Y16 = tf.layers.max_pooling2d(Y13, pool_size=3, strides=2, padding="same")    # output 16x16x32
    Y17 = layer_conv2d_batch_norm_relu(Y16, filters=16*g*f, kernel_size=1, strides=1) #squeeze
    Y17l = layer_conv2d_batch_norm_relu(Y17, filters=16*g*f, kernel_size=1, strides=1) #expand 1x1
    Y17t = layer_conv2d_batch_norm_relu(Y17, filters=16*g*f, kernel_size=3, strides=1) #expand 3x3
    Y18 = tf.concat([Y17l, Y17t], 3)                                              # output 16x16x32

    # old bounding box head
    #T19 = layer_conv2d_batch_norm_relu(Y18, filters=CELL_B*4, kernel_size=1, strides=1) # output 16*16*12
    #T20 = tf.layers.average_pooling2d(T19, pool_size=4, strides=4, padding="valid") # 4x4x12 shape [batch, 4,4,12]
    # for each cell, this has CELL_B predictions of bounding box (x,y,w,c)
    # apply tanh for x, y and sigmoid for w, c
    #TX, TY, TW, TC = tf.split(T20, 4, axis=3)  # shape 4 x [batch, 4,4,3] = 4 x [batch, GRID_N, GRID_N, CELL_B]

    # bounding box head
    T19 = layer_conv2d_batch_norm_relu(Y18, filters=CELL_B*8*f*g, kernel_size=1, strides=1) # output 16*16*24 if CELL_B=3
    T20=T19
    # not needed at GRID_N=16
    # for GRID_N=8, need pool_size=2, strides=2
    # for GRID_N=4, need pool_size=4, strides=4
    #T20 = tf.layers.average_pooling2d(T19, pool_size=2, strides=2, padding="valid") # 4x4x12 shape [batch, 4,4,12] = [batch, GRID_N, GRID_N, CELL_B*8]
    # for each cell, this has CELL_B predictions of bounding box (x,y,w,c)
    # apply tanh for x, y and sigmoid for w, c
    TX0, TY0, TW0, TC0 = tf.split(T20, 4, axis=3)  # shape 4 x [batch, GRID_N, GRID_N, CELL_B*2]
    # TODO: idea: batch norm may be bad on this layer
    # TODO: try with a deeper layer as well
    # TODO: try a filtered convolution instead of pooling2d, maybe info from cell sides should be weighted differently
    TX = tf.nn.tanh(layer_conv1x1_batch_norm(TX0, depth=CELL_B))  # shape [batch, 4,4,CELL_B]
    TY = tf.nn.tanh(layer_conv1x1_batch_norm(TY0, depth=CELL_B))  # shape [batch, 4,4,CELL_B]
    TW = tf.nn.sigmoid(layer_conv1x1_batch_norm(TW0, depth=CELL_B))  # shape [batch, 4,4,CELL_B]
    TC = tf.nn.sigmoid(layer_conv1x1_batch_norm(TC0, depth=CELL_B))  # shape [batch, 4,4,CELL_B]

    # leave some breathing room to the roi sizes so that rois from adjacent cells can reach into this one
    TX = TX
    TY = TY

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

    rois = tf.stack([TX,TY,real_TW], axis=-1)  # shape [batch, GRID_N, GRID_N, CEL_B, 3]
    rois = boxutils.grid_cell_to_tile_coords(rois, GRID_N, TILE_SIZE)/TILE_SIZE # shape [batch, GRID_N, GRID_N, CELL_B, 4]
    rsrois = tf.reshape(rois, [-1, GRID_N*GRID_N*CELL_B, 4])
    rsroiC = tf.reshape(TC, [-1, GRID_N*GRID_N*CELL_B])

    loss = train_op = eval_metrics = None
    if mode != tf.estimator.ModeKeys.PREDICT:
        ZERO_W = 0.0001
        C_ = labels["count"]
        T_ = labels["target"]  # shape [4,4,3,3] = [batch, GRID_N, GRID_N, CEL_B, xyw]
        TX_, TY_, TW_ = tf.unstack(T_, 3, axis=-1) # shape 3 x [batch, 4,4,3] = [batch, GRID_N, GRID_N, CELL_B]
        # target probability is 1 if there is a corresponding target box, 0 otherwise
        TC_ = tf.greater(TW_, ZERO_W)
        fTC_ = tf.cast(tf.greater(TW_, ZERO_W), tf.float32) # shape [batch, 4,4,3] = [batch, GRID_N, GRID_N, CELL_B]

        # accuracy
        DETECTION_TRESHOLD = 0.5  # plane "detected" if predicted C>0.5
        ERROR_TRESHOLD = 0.3  # plane correctly localized if predicted x,y,w within % of ground truth
        detect_correct = tf.logical_not(tf.logical_xor(tf.greater(TC, DETECTION_TRESHOLD), TC_))
        ones = tf.ones(tf.shape(TW_))
        nonzero_TW_ = tf.where(TC_, TW_, ones)
        # true if correct size where there is a planne, nonsense value where there is no plane
        size_correct = tf.less(tf.abs(real_TW - TW_) / nonzero_TW_, ERROR_TRESHOLD)
        # true if correct position where there is a planne, nonsense value where there is no plane
        position_correct = tf.less(tf.sqrt(tf.square(TX-TX_) + tf.square(TY-TY_)) / nonzero_TW_ / GRID_N, ERROR_TRESHOLD)
        truth_no_plane = tf.logical_not(TC_)
        size_correct = tf.logical_or(size_correct, truth_no_plane)
        position_correct = tf.logical_or(position_correct, truth_no_plane)
        size_correct = tf.logical_and(detect_correct, size_correct)
        position_correct = tf.logical_and(detect_correct, position_correct)
        all_correct = tf.logical_and(size_correct, position_correct)
        mistakes = tf.reduce_sum(tf.cast(tf.logical_not(all_correct), tf.int32), axis=[1,2,3])  # shape [batch]

        # debug: expected and predicted counts
        debug_img = X
        #debug_img = image_compose(debug_img, get_bottom_left_digits(C_))
        #debug_img = image_compose(debug_img, get_bottom_right_digits(classes))
        debug_img = image_compose(debug_img, get_top_right_red_white_digits(mistakes))
        # debug: ground truth boxes in grey
        target_rois = boxutils.grid_cell_to_tile_coords(T_, GRID_N, TILE_SIZE)/TILE_SIZE
        rstarget_rois = tf.reshape(target_rois, [-1, GRID_N*GRID_N*CELL_B, 4])
        debug_img = draw_color_boxes(debug_img, rstarget_rois, 0.7, 0.7, 0.7)
        # debug: computed ROIs boxes in shades of yellow
        no_box = tf.zeros(tf.shape(rsrois))
        select = tf.stack([rsroiC, rsroiC, rsroiC, rsroiC], axis=-1)
        select_correct = tf.reshape(all_correct, [-1, GRID_N*GRID_N*CELL_B])
        select_size_correct = tf.reshape(size_correct, [-1, GRID_N*GRID_N*CELL_B])
        select_position_correct = tf.reshape(position_correct, [-1, GRID_N*GRID_N*CELL_B])

        select_correct = tf.stack([select_correct,select_correct,select_correct,select_correct], axis=2)
        select_size_correct = tf.stack([select_size_correct,select_size_correct,select_size_correct,select_size_correct], axis=2)
        select_position_correct = tf.stack([select_position_correct,select_position_correct,select_position_correct,select_position_correct], axis=2)

        correct_rois = tf.where(select_correct, rsrois, no_box)
        other_rois = tf.where(tf.logical_not(select_correct), rsrois, no_box)
        correct_size_rois = tf.where(select_size_correct, other_rois, no_box)
        other_rois = tf.where(tf.logical_not(select_size_correct), other_rois, no_box)
        correct_pos_rois = tf.where(select_position_correct, other_rois, no_box)
        other_rois = tf.where(tf.logical_not(select_position_correct), other_rois, no_box)
        # correct rois in yellow
        for i in range(9):
            debug_rois = tf.where(tf.greater(select, 0.1*(i+1)), correct_rois, no_box)
            debug_img = draw_color_boxes(debug_img, debug_rois, 0.1*(i+2), 0.1*(i+2), 0)
        # size only correct rois in blue
        for i in range(9):
            debug_rois = tf.where(tf.greater(select, 0.1*(i+1)), correct_size_rois, no_box)
            debug_img = draw_color_boxes(debug_img, debug_rois, 0, 0, 0.1*(i+2))
        # position only correct rois in green
        for i in range(9):
            debug_rois = tf.where(tf.greater(select, 0.1*(i+1)), correct_pos_rois, no_box)
            debug_img = draw_color_boxes(debug_img, debug_rois, 0, 0.1*(i+2), 0)
        # incorrect rois in red
        for i in range(9):
            debug_rois = tf.where(tf.greater(select, 0.1*(i+1)), other_rois, no_box)
            debug_img = draw_color_boxes(debug_img, debug_rois, 0.1*(i+2), 0, 0)
        tf.summary.image("input_image", debug_img, max_outputs=10)

        # model outputs
        position_loss = tf.reduce_mean(fTC_ * (tf.square(TX-TX_)+tf.square(TY-TY_)))
        # YOLO trick: take square root of predicted size for loss so as not to drown errors on small boxes

        # testing different options for W
        # Woption0
        #size_loss = tf.reduce_mean(fTC_ * tf.square(tf.sqrt(TW)-tf.sqrt(TW_)) * 2)
        # Woption1
        #size_loss = tf.reduce_mean(fTC_ * tf.square(TW-tf.sqrt(TW_)) * 2)
        # Woption2
        size_loss = tf.reduce_mean(fTC_ * tf.square(TW-TW_) * 2)
        # Woption3
        #size_loss = tf.reduce_mean(fTC_ * tf.square(tf.sqrt(TW)-TW_) * 2)
        # Woption4
        #size_loss = tf.reduce_mean(fTC_ * tf.square(TW-TW_*TW_) * 2)
        # Woption5
        #size_loss = tf.reduce_mean(fTC_ * tf.square(TW*TW-TW_) * 2)

        obj_loss = tf.reduce_mean(fTC_ * tf.square(TC - 1))
        noobj_loss = tf.reduce_mean((1-fTC_) * tf.square(TC - 0))

        # TODO: idea, add a per-cell plane/no plane detection head. Maybe it can force better gradients (?)
        # because current split of detections "per responsible bounding box" might be hard for a neural network
        # TODO: similar idea: if only one plane in cell, teach all CELL_B detectors to detect it
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
        LW0 = 10.0
        LW1 = params['lw1']
        LW2 = params['lw2']
        LW3 = params['lw3']
        LWT = LW0+LW1+LW2+LW3
        # TODO: hyperparam tune the hell out of these loss weights
        w_obj_loss = obj_loss*(LW0/LWT)
        w_position_loss = position_loss*(LW1/LWT)
        w_size_loss = size_loss*(LW2/LWT)
        w_noobj_loss = noobj_loss*(LW3/LWT)
        loss = w_position_loss + w_size_loss + w_obj_loss + w_noobj_loss

        # average number of mistakes per image
        nb_mistakes = tf.reduce_sum(mistakes)

        train_op = tf.contrib.layers.optimize_loss(loss, tf.train.get_global_step(), learning_rate=params['lr0'], optimizer="Adam", learning_rate_decay_fn=learn_rate)
        eval_metrics = {
                        "position_error": tf.metrics.mean(w_position_loss),
                        "size_error": tf.metrics.mean(w_size_loss),
                        "plane_confidence_error": tf.metrics.mean(w_obj_loss),
                        "no_plane_confidence_error": tf.metrics.mean(w_noobj_loss),
                        "mistakes": tf.metrics.mean(nb_mistakes)}
        #debug
        tf.summary.scalar("position_error", w_position_loss)
        tf.summary.scalar("size_error", w_size_loss)
        tf.summary.scalar("plane_confidence_error", w_obj_loss)
        tf.summary.scalar("no_plane_confidence_error", w_noobj_loss)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("mistakes", nb_mistakes)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={"rois":rsrois, "rois_confidence": rsroiC},  # name these fields as you like
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metrics,
        export_outputs={'classes': tf.estimator.export.PredictOutput({"rois":rsrois, "rois_confidence": rsroiC})}
    )