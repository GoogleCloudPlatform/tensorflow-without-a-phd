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

Actual neural network model code"""

import math
import tensorflow as tf
from trainer_yolo import utils_box as box
from trainer_yolo import utils_imgdbg as imgdbg
from trainer_yolo import model_layers as layer
from trainer_yolo import settings
from tensorflow.python.platform import tf_logging as logging

def learn_rate_decay(step, params):
    """ Model building utility function. Learning rate decay parametrized from
    command-line parameters lr0, lr1 and lr2."""
    if params['decay_type'] == "exponential":
        lr = params['lr1'] + tf.train.exponential_decay(params['lr0'], step, params['lr2'], 1/math.e)
    elif params['decay_type'] == "cosine-restarts":
        # empirically  determined t_mul rates for cosine_restarts. With these rates, learning rate is
        # guaranteed to decay to its min value by the end of "iterations" with "decay-restarts" restarts
        # and a first restart at "iterations" / 8.
        t_muls = [1.0, 7.0, 2.1926, 1.48831, 1.23692, 1.11434, 1.04422]
        t_mul = t_muls[params["decay_restarts"]]
        m_mul = params["decay_restart_height"]
        first_decay_steps = params["iterations"] // 8 if params["decay_restarts"] > 0 else params["iterations"]
        lr = params['lr1'] + tf.train.cosine_decay_restarts(params['lr0'], step, first_decay_steps, t_mul, m_mul)
    return lr


def model_core_squeezenet12(x, mode, params, info):
    y, info = layer.conv2d_batch_norm_relu_dropout_l(x, mode, params, info, filters=32, kernel_size=6, strides=2)  # output 128x128
    y, info = layer.maxpool_l(y, info)  # output 64x64
    y, info = layer.sqnet_squeeze(y, mode, params, info, 21)
    y, info = layer.sqnet_expand(y, mode, params, info, 2*26)
    y, info = layer.sqnet_squeeze(y, mode, params, info, 36)
    y, info = layer.sqnet_expand(y, mode, params, info, 2*46)
    y, info = layer.maxpool_l(y, info)  # output 32x32
    y, info = layer.sqnet_squeeze(y, mode, params, info, 41)
    y, info = layer.sqnet_expand(y, mode, params, info, 2*36)
    y, info = layer.sqnet_squeeze(y, mode, params, info, 31)
    y, info = layer.sqnet_expand(y, mode, params, info, 2*26)
    y, info = layer.maxpool_l(y, info)  # output 16x16
    y, info = layer.sqnet_squeeze(y, mode, params, info, 21)
    y, info = layer.sqnet_expand(y, mode, params, info, 2*16, last=True)
    return y, info


def model_core_squeezenet17(x, mode, params, info):
    y, info = layer.conv2d_batch_norm_relu_dropout_l(x, mode, params, info, filters=128, kernel_size=3, strides=1)
    y, info = layer.sqnet_expand(y, mode, params, info, 2*64)
    y, info = layer.maxpool_l(y, info)  # output 128x128
    #y, info = layer.sqnet_squeeze_pool(y, mode, params, info, 80)
    y, info = layer.sqnet_squeeze(y, mode, params, info, 80)
    y, info = layer.sqnet_expand(y, mode, params, info, 2*96)
    y, info = layer.maxpool_l(y, info)  # output 64x64
    #y, info = layer.sqnet_squeeze_pool(y, mode, params, info, 104)
    y, info = layer.sqnet_squeeze(y, mode, params, info, 104)
    y, info = layer.sqnet_expand(y, mode, params, info, 2*112)
    y, info = layer.sqnet_squeeze(y, mode, params, info, 120)
    y, info = layer.sqnet_expand(y, mode, params, info, 2*128)
    y, info = layer.maxpool_l(y, info)  # output 32x32
    #y, info = layer.sqnet_squeeze_pool(y, mode, params, info, 120)
    y, info = layer.sqnet_squeeze(y, mode, params, info, 120)
    y, info = layer.sqnet_expand(y, mode, params, info, 2*112)
    y, info = layer.sqnet_squeeze(y, mode, params, info, 104)
    y, info = layer.sqnet_expand(y, mode, params, info, 2*96)
    y, info = layer.maxpool_l(y, info)  # output 16x16
    #y, info = layer.sqnet_squeeze_pool(y, mode, params, info, 88)
    y, info = layer.sqnet_squeeze(y, mode, params, info, 88)
    y, info = layer.sqnet_expand(y, mode, params, info, 2*80)
    y, info = layer.sqnet_squeeze(y, mode, params, info, 72)
    y, info = layer.sqnet_expand(y, mode, params, info, 2*65, last=True)
    return y, info


def model_core_darknet(x, mode, params, info):
    y, info = layer.conv2d_batch_norm_relu_dropout_l(x, mode, params, info, filters=64, kernel_size=3, strides=1)
    y, info = layer.conv2d_batch_norm_relu_dropout_l(y, mode, params, info, filters=50, kernel_size=1, strides=1)
    y, info = layer.maxpool_l(y, info) # output 128x128
    y, info = layer.conv2d_batch_norm_relu_dropout_l(y, mode, params, info, filters=52, kernel_size=3, strides=1)
    y, info = layer.conv2d_batch_norm_relu_dropout_l(y, mode, params, info, filters=54, kernel_size=1, strides=1)
    y, info = layer.maxpool_l(y, info) # output 64x64
    y, info = layer.conv2d_batch_norm_relu_dropout_l(y, mode, params, info, filters=56, kernel_size=3, strides=1)
    y, info = layer.conv2d_batch_norm_relu_dropout_l(y, mode, params, info, filters=58, kernel_size=1, strides=1)
    y, info = layer.maxpool_l(y, info) # output 32x32
    y, info = layer.conv2d_batch_norm_relu_dropout_l(y, mode, params, info, filters=60, kernel_size=3, strides=1)
    y, info = layer.conv2d_batch_norm_relu_dropout_l(y, mode, params, info, filters=62, kernel_size=1, strides=1)
    y, info = layer.maxpool_l(y, info) # output 16x16
    y, info = layer.conv2d_batch_norm_relu_dropout_l(y, mode, params, info, filters=64, kernel_size=3, strides=1)
    y, info = layer.conv2d_batch_norm_relu_dropout_l(y, mode, params, info, filters=65, kernel_size=1, strides=1)
    # the last number of filters must be multiple of 5 so that the YOLO head can be added afterwards
    return y, info

def model_core_darknet17(x, mode, params, info):
    y, info = layer.conv2d_batch_norm_relu_dropout_l(x, mode, params, info, filters=100, kernel_size=3, strides=1)
    y, info = layer.conv2d_batch_norm_relu_dropout_l(y, mode, params, info, filters=100, kernel_size=1, strides=1)
    y, info = layer.maxpool_l(y, info) # output 128x128
    y, info = layer.conv2d_batch_norm_relu_dropout_l(y, mode, params, info, filters=100, kernel_size=3, strides=1)
    y, info = layer.conv2d_batch_norm_relu_dropout_l(y, mode, params, info, filters=100, kernel_size=1, strides=1)
    y, info = layer.conv2d_batch_norm_relu_dropout_l(y, mode, params, info, filters=100, kernel_size=3, strides=1)
    y, info = layer.conv2d_batch_norm_relu_dropout_l(y, mode, params, info, filters=100, kernel_size=1, strides=1)
    y, info = layer.maxpool_l(y, info) # output 64x64
    y, info = layer.conv2d_batch_norm_relu_dropout_l(y, mode, params, info, filters=100, kernel_size=3, strides=1)
    y, info = layer.conv2d_batch_norm_relu_dropout_l(y, mode, params, info, filters=100, kernel_size=1, strides=1)
    y, info = layer.conv2d_batch_norm_relu_dropout_l(y, mode, params, info, filters=100, kernel_size=3, strides=1)
    y, info = layer.conv2d_batch_norm_relu_dropout_l(y, mode, params, info, filters=100, kernel_size=1, strides=1)
    y, info = layer.maxpool_l(y, info) # output 32x32
    y, info = layer.conv2d_batch_norm_relu_dropout_l(y, mode, params, info, filters=100, kernel_size=3, strides=1)
    y, info = layer.conv2d_batch_norm_relu_dropout_l(y, mode, params, info, filters=100, kernel_size=1, strides=1)
    y, info = layer.conv2d_batch_norm_relu_dropout_l(y, mode, params, info, filters=100, kernel_size=3, strides=1)
    y, info = layer.conv2d_batch_norm_relu_dropout_l(y, mode, params, info, filters=100, kernel_size=1, strides=1)
    y, info = layer.maxpool_l(y, info) # output 16x16
    y, info = layer.conv2d_batch_norm_relu_dropout_l(y, mode, params, info, filters=100, kernel_size=3, strides=1)
    y, info = layer.conv2d_batch_norm_relu_dropout_l(y, mode, params, info, filters=100, kernel_size=1, strides=1)
    # the last number of filters must be multiple of 5 so that the YOLO head can be added afterwards
    return y, info

def model_core_configurable_squeezenet(x, mode, params, info):
    """This configurable model tries to spread the layers evenly between the maxpool layers
    and also spread depth increases and decreases in a progressive way. Depth at the end is
    the same as it was initially."""
    nlayers = params["layers"]
    depth_increment = params["depth_increment"]
    first_layer_fdepth = params["first_layer_filter_depth"]
    first_layer_fstride = params["first_layer_filter_stride"]
    first_layer_fsize = params["first_layer_filter_size"]
    layers_n, first_expand_layer, depth_incr_doubler = layer.sqnet_spread_layers(nlayers)


    def sqnet_squeeze_expand(x, mode, params, info, depth_increment, last=False):
        """Squeezenet "fire" module, i.e. a "squeeze" module followed by an "expand" module."""
        depth = int(x.get_shape()[3])//2
        depth += depth_increment
        x, info = layer.sqnet_squeeze(x, mode, params, info, depth)
        depth += depth_increment
        x, info = layer.sqnet_expand(x, mode, params, info, 2*depth, last=last)
        return x, info

    y, info = layer.conv2d_batch_norm_relu_dropout_l(x, mode, params, info,
                                                     filters=first_layer_fdepth,
                                                     kernel_size=first_layer_fsize,
                                                     strides=first_layer_fstride)
    if first_expand_layer == 1:
        y, info = layer.sqnet_expand(y, mode, params, info, first_layer_fdepth)
    y, info = layer.maxpool_l(y, info)
    for _ in range(layers_n[1]//2):
        y, info = sqnet_squeeze_expand(y, mode, params, info, depth_increment*depth_incr_doubler[0])
    if first_layer_fstride == 1:
        y, info = layer.maxpool_l(y, info)
    for _ in range(layers_n[2]//2):
        y, info = sqnet_squeeze_expand(y, mode, params, info, depth_increment*depth_incr_doubler[1])
    y, info = layer.maxpool_l(y, info)
    for _ in range(layers_n[3]//2):
        y, info = sqnet_squeeze_expand(y, mode, params, info, -depth_increment*depth_incr_doubler[2])
    y, info = layer.maxpool_l(y, info)
    for i in range(layers_n[4]//2, 0, -1):
        y, info = sqnet_squeeze_expand(y, mode, params, info, -depth_increment*depth_incr_doubler[3], last=(i==1))

    return y, info

def model_fn(features, labels, mode, params):
    """The model, with loss, metrics and debug summaries"""

    # YOLO parameters
    grid_nn = params["grid_nn"]  # each tile is divided into a grid_nn x grid_nn grid
    cell_n = params["cell_n"]  # each grid cell predicts cell_n bounding boxes.
    info = None

    # model inputs
    X = tf.to_float(features["image"]) / 255.0 # input image format is uint8 with range 0 to 255

    # The model itself is here
    #Y, info = model_core_squeezenet12(X, mode, params, info)
    #Y, info = model_core_squeezenet17(X, mode, params, info)
    #Y, info = model_core_darknet(X, mode, params, info)
    #Y, info = model_core_darknet17(X, mode, params, info)
    Y, info = model_core_configurable_squeezenet(X, mode, params, info)

    # YOLO head: predicts bounding boxes around airplanes
    box_x, box_y, box_w, box_c, box_c_logits, info = layer.YOLO_head(Y, mode, params, info, grid_nn, cell_n)

    # Debug: print the model structure
    if mode == tf.estimator.ModeKeys.TRAIN:
        logging.log(logging.INFO, info["description"])
        logging.log(logging.INFO, "NN {} layers / {:,d} total weights".format(info["layers"], info["weights"]))

    # TODO: refactor predicted_rois and predicted_c (or keep it to keep the conde compatible with confidence factor implem?)
    # with the current softmax implementation, confidence factors are either 0 or 1.
    box_c_sim = tf.cast(tf.argmax(box_c, axis=-1), dtype=tf.float32)  # shape [batch, GRID_N,GRID_N,CELL_B]
    DETECTION_TRESHOLD = 0.5  # plane "detected" if predicted C>0.5
    detected_w = tf.where(tf.greater(box_c_sim, DETECTION_TRESHOLD), box_w, tf.zeros_like(box_w))
    # all rois with confidence factors
    predicted_rois = tf.stack([box_x, box_y, box_w], axis=-1)  # shape [batch, GRID_N, GRID_N, CELL_B, 3]
    predicted_rois = box.grid_cell_to_tile_coords(predicted_rois, grid_nn, settings.TILE_SIZE) / settings.TILE_SIZE
    predicted_rois = tf.reshape(predicted_rois, [-1, grid_nn*grid_nn*cell_n, 4])
    predicted_c = tf.reshape(box_c_sim, [-1, grid_nn*grid_nn*cell_n])
    # only the rois where a plane was detected
    detected_rois = tf.stack([box_x, box_y, detected_w], axis=-1)  # shape [batch, GRID_N, GRID_N, CELL_B, 3]
    detected_rois = box.grid_cell_to_tile_coords(detected_rois, grid_nn, settings.TILE_SIZE) / settings.TILE_SIZE
    detected_rois = tf.reshape(detected_rois, [-1, grid_nn*grid_nn*cell_n, 4])
    detected_rois, detected_rois_overflow = box.remove_empty_rois(detected_rois, settings.MAX_DETECTED_ROIS_PER_TILE)

    loss = train_op = eval_metrics = None
    if mode != tf.estimator.ModeKeys.PREDICT:

        # Target labels
        target_count = labels["count"]  # not used
        # Ground truth boxes. Used to compute IOU accuracy and display debug ground truth boxes.
        target_rois = labels["target_rois"] # shape [batch, MAX_TARGET_ROIS_PER_TILE, x1y1x2y2]
        # Ground truth boxes assigned to YOLO grid cells. Used to compute loss.
        target_rois_yolo = labels["yolo_target_rois"]  # shape [4,4,3,3] = [batch, GRID_N, GRID_N, CEL_B, xyw]
        target_x, target_y, target_w = tf.unstack(target_rois_yolo, 3, axis=-1) # shape 3 x [batch, 4,4,3] = [batch, GRID_N, GRID_N, CELL_B]
        # target probability is 1 if there is a corresponding target box, 0 otherwise
        target_is_plane = tf.greater(target_w, 0.0001)
        target_is_plane_onehot = tf.one_hot(tf.cast(target_is_plane, tf.int32), 2, dtype=tf.float32)
        target_is_plane_float = tf.cast(target_is_plane, tf.float32) # shape [batch, 4,4,3] = [batch, GRID_N, GRID_N, CELL_B]

        # Mistakes and correct detections for visualisation and debugging.
        # This is computed against the ground truth boxes assigned to YOLO grid cells.
        mistakes, size_correct, position_correct, all_correct = box.compute_mistakes(box_x, box_y,
                                                                                     box_w, box_c_sim,
                                                                                     target_x, target_y,
                                                                                     target_w, target_is_plane, grid_nn)
        # Debug image for logging in Tensorboad.
        debug_img = imgdbg.debug_image(X, mistakes, target_rois, predicted_rois, predicted_c,
                                       size_correct, position_correct, all_correct,
                                       grid_nn, cell_n, settings.TILE_SIZE)

        # IOU (Intersection Over Union) accuracy
        # IOU computation removed from training mode because it used an op not yet supported with MirroredStrategy
        if mode == tf.estimator.ModeKeys.EVAL:
            iou_accuracy = box.compute_safe_IOU(target_rois, detected_rois, detected_rois_overflow, settings.TILE_SIZE)

        # Improvement ideas and experiment results
        # 1) YOLO trick: take square root of predicted size for loss so as not to drown errors on small boxes: tested, no benefit
        # 2) if only one plane in cell, teach all cell_n detectors to detect it: implemented in box.n_experimental_roi_selection_strategy, beneficial
        # 3) TODO: try two or more grids, shifted by 1/2 cell size: This could make it easier to have cells detect planes in their center, if that is an actual problem they have (no idea)
        # 4) try using TC instead of TC_ in position loss and size loss: tested, no benefit
        # 5) TODO: one run without batch norm for comparison
        # 6) TODO: add dropout, tested, weird resukts: eval accuracy goes up signicantly but model performs worse in real life. Probably not enough training data.
        # 7) TODO: idea, compute detection box loss agains all ROI, not just assigned ROIs: if neighboring cell detects something that aligns well with ground truth, no reason to penalise
        # 8) TODO: add tile rotations, tile color inversion (data augmentation)

        # Loss function
        position_loss = tf.reduce_mean(target_is_plane_float * (tf.square(box_x - target_x) + tf.square(box_y - target_y)))
        size_loss = tf.reduce_mean(target_is_plane_float * tf.square(box_w - target_w) * 2)
        obj_loss = tf.losses.softmax_cross_entropy(target_is_plane_onehot, box_c_logits)

        # YOLO trick: weights the different losses differently
        loss_weight_total = (params['lw1'] + params['lw2'] + params['lw3']) * 1.0  # 1.0 to force conversion to float
        w_obj_loss = obj_loss*(params['lw1'] / loss_weight_total)
        w_position_loss = position_loss*(params['lw2'] / loss_weight_total)
        w_size_loss = size_loss*(params['lw3'] / loss_weight_total)
        loss = w_position_loss + w_size_loss + w_obj_loss

        # average number of mistakes per image
        nb_mistakes = tf.reduce_sum(mistakes)

        lr = learn_rate_decay(tf.train.get_or_create_global_step(), params)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = tf.contrib.training.create_train_op(loss, optimizer)

        if mode == tf.estimator.ModeKeys.EVAL:
            # metrics removed from training mode because they are not yet supported with MirroredStrategy
            eval_metrics = {"position_error": tf.metrics.mean(w_position_loss),
                            "size_error": tf.metrics.mean(w_size_loss),
                            "plane_cross_entropy_error": tf.metrics.mean(w_obj_loss),
                            "mistakes": tf.metrics.mean(nb_mistakes),
                            'IOU': tf.metrics.mean(iou_accuracy)}
        else:
            eval_metrics = None


        # Tensorboard summaries for debugging
        tf.summary.scalar("position_error", w_position_loss)
        tf.summary.scalar("size_error", w_size_loss)
        tf.summary.scalar("plane_cross_entropy_error", w_obj_loss)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("mistakes", nb_mistakes)
        tf.summary.scalar("learning_rate", lr)
        tf.summary.image("input_image", debug_img, max_outputs=20)
        # a summary on iou_accuracy would be nice but it goes Out Of Memory

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={"rois":predicted_rois, "rois_confidence": predicted_c},  # name these fields as you like
        loss=loss, train_op=train_op, eval_metric_ops=eval_metrics,
        export_outputs={'classes': tf.estimator.export.PredictOutput({"rois": box.swap_xy(predicted_rois), # TODO: the visualisation GUI was coded for swapped coordinates y1 x1 y2 x2
                                                                      "rois_confidence": predicted_c})}  # TODO: remove legacy C
    )
