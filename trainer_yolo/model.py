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
from trainer_yolo.boxutils import IOUCalculator

TILE_SIZE = 256  # must be the same as in train.py
MAX_DETECTED_ROIS_PER_TILE = 100  # can be different from MAX_TARGET_ROIS_PER_TILE in train.py. The max possible is GRID_N * GRID_N * CELL_B.

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

# Model
def model_fn_squeeze(features, labels, mode, params):

    # YOLO parameters: each tile is divided into a grid_nn x grid_nn grid,
    # each grid cell predicts cell_n ROIs.
    grid_nn = params["grid_nn"]
    cell_n = params["cell_n"]

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
    Y18 = tf.concat([Y17l, Y17t], 3)                                              # output 16x16x128

    # old bounding box head
    #T19 = layer_conv2d_batch_norm_relu(Y18, filters=CELL_B*4, kernel_size=1, strides=1) # output 16*16*12
    #T20 = tf.layers.average_pooling2d(T19, pool_size=4, strides=4, padding="valid") # 4x4x12 shape [batch, 4,4,12]
    # for each cell, this has CELL_B predictions of bounding box (x,y,w,c)
    # apply tanh for x, y and sigmoid for w, c
    #TX, TY, TW, TC = tf.split(T20, 4, axis=3)  # shape 4 x [batch, 4,4,3] = 4 x [batch, GRID_N, GRID_N, CELL_B]

    # bounding box head
    T19 = layer_conv2d_batch_norm_relu(Y18, filters=20*f*g, kernel_size=1, strides=1) # output 16*16*(cell_n*32)
    # not needed at GRID_N=16
    # for GRID_N=16, need pool_size=1, strides=1
    # for GRID_N=8, need pool_size=2, strides=2
    # for GRID_N=4, need pool_size=4, strides=4
    pool_size = 16//grid_nn
    T20 = tf.layers.average_pooling2d(T19, pool_size=pool_size, strides=pool_size, padding="valid") # [batch, grid_nn, grid_nn, cell_n*32]
    # for each cell, this has CELL_B predictions of bounding box (x,y,w,c)
    # apply tanh for x, y and sigmoid for w, c
    TX0, TY0, TW0, TC00, TC01 = tf.split(T20, 5, axis=-1)  # shape 4 x [batch, grid_nn, grid_nn, 16]
    TC0 = tf.concat([TC00, TC01], axis=-1)
    # TODO: idea: batch norm may be bad on this layer
    # TODO: try with a deeper layer as well
    # TODO: try a filtered convolution instead of pooling2d, maybe info from cell sides should be weighted differently
    # TODO: try softmax for predicting confidence instead of C
    TX = tf.nn.tanh(layer_conv1x1_batch_norm(TX0, depth=cell_n))  # shape [batch, grid_nn, grid_nn, cell_n]
    TY = tf.nn.tanh(layer_conv1x1_batch_norm(TY0, depth=cell_n))  # shape [batch, grid_nn, grid_nn, cell_n]
    TW = tf.nn.sigmoid(layer_conv1x1_batch_norm(TW0, depth=cell_n))  # shape [batch, grid_nn, grid_nn, cell_n]
    #TC = tf.nn.sigmoid(layer_conv1x1_batch_norm(TC0, depth=CELL_B))  # shape [batch, GRID_N,GRID_N,CELL_B]
    #  2 is the number of classes: planes, or no planes
    TClogits = layer_conv1x1_batch_norm(TC0, depth=cell_n*2)   # shape [batch, grid_nn, grid_nn, cell_n*2]
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
        iou_accuracy = tf.reduce_mean(iou_accuracy)
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

        lr = learn_rate(params['lr0'], tf.train.get_or_create_global_step())
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
        export_outputs={'classes': tf.estimator.export.PredictOutput({"rois":predicted_rois, "rois_confidence": predicted_C})}
    )