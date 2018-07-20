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

Debugging functions for images. Utilities to print digits and rectangles
on images so that they can be logged for visualisation in Tensorboard."""


import numpy as np
import tensorflow as tf
from trainer_yolo import utils_box as box  # TODO: refactor to remove this dependency

# helper to print expected and inferred digits on pictures.

RAW_DIGIT_W = 5
RAW_DIGIT_H = 7


def raw_digits():
    d = np.array(
        [[[0, 0, 0, 0, 0],
          [0, 1, 1, 1, 0],
          [0, 1, 0, 1, 0],
          [0, 1, 0, 1, 0],
          [0, 1, 0, 1, 0],
          [0, 1, 1, 1, 0],
          [0, 0, 0, 0, 0]],  # digit 0
         [[0, 0, 0, 0, 0],
          [0, 0, 1, 0, 0],
          [0, 1, 1, 0, 0],
          [0, 0, 1, 0, 0],
          [0, 0, 1, 0, 0],
          [0, 0, 1, 0, 0],
          [0, 0, 0, 0, 0]],  # digit 1
         [[0, 0, 0, 0, 0],
          [0, 1, 1, 1, 0],
          [0, 0, 0, 1, 0],
          [0, 1, 1, 1, 0],
          [0, 1, 0, 0, 0],
          [0, 1, 1, 1, 0],
          [0, 0, 0, 0, 0]],  # digit 2
         [[0, 0, 0, 0, 0],
          [0, 1, 1, 1, 0],
          [0, 0, 0, 1, 0],
          [0, 1, 1, 1, 0],
          [0, 0, 0, 1, 0],
          [0, 1, 1, 1, 0],
          [0, 0, 0, 0, 0]],  # digit 3
         [[0, 0, 0, 0, 0],
          [0, 0, 0, 1, 0],
          [0, 0, 1, 1, 0],
          [0, 1, 0, 1, 0],
          [0, 1, 1, 1, 0],
          [0, 0, 0, 1, 0],
          [0, 0, 0, 0, 0]],  # digit 4
         [[0, 0, 0, 0, 0],
          [0, 1, 1, 1, 0],
          [0, 1, 0, 0, 0],
          [0, 1, 1, 1, 0],
          [0, 0, 0, 1, 0],
          [0, 1, 1, 1, 0],
          [0, 0, 0, 0, 0]],  # digit 5
         [[0, 0, 0, 0, 0],
          [0, 1, 1, 1, 0],
          [0, 1, 0, 0, 0],
          [0, 1, 1, 1, 0],
          [0, 1, 0, 1, 0],
          [0, 1, 1, 1, 0],
          [0, 0, 0, 0, 0]],  # digit 6
         [[0, 0, 0, 0, 0],
          [0, 1, 1, 1, 0],
          [0, 0, 0, 1, 0],
          [0, 0, 1, 0, 0],
          [0, 1, 0, 0, 0],
          [0, 1, 0, 0, 0],
          [0, 0, 0, 0, 0]],  # digit 7
         [[0, 0, 0, 0, 0],
          [0, 1, 1, 1, 0],
          [0, 1, 0, 1, 0],
          [0, 1, 1, 1, 0],
          [0, 1, 0, 1, 0],
          [0, 1, 1, 1, 0],
          [0, 0, 0, 0, 0]],  # digit 8
         [[0, 0, 0, 0, 0],
          [0, 1, 1, 1, 0],
          [0, 1, 0, 1, 0],
          [0, 1, 1, 1, 0],
          [0, 0, 0, 1, 0],
          [0, 1, 1, 1, 0],
          [0, 0, 0, 0, 0]]], np.uint8)   # digit 9
    return d


def digits_bottom_left(w, h):
    d = raw_digits()
    padded_d = np.pad(d, [(0, 0), (w - RAW_DIGIT_W, 0), (0, h - RAW_DIGIT_H)], 'edge')
    return tf.expand_dims(tf.constant(padded_d, tf.float32), -1)


def digits_bottom_right(w, h):
    d = raw_digits()
    padded_d = np.pad(d, [(0, 0), (w - RAW_DIGIT_W, 0), (h - RAW_DIGIT_H, 0)], 'edge')
    return tf.expand_dims(tf.constant(padded_d, tf.float32), -1)


def digits_top_left(w, h):
    d = raw_digits()
    padded_d = np.pad(d, [(0, 0), (0, w - RAW_DIGIT_W), (0, h - RAW_DIGIT_H)], 'edge')
    return tf.expand_dims(tf.constant(padded_d, tf.float32), -1)


def digits_top_right(w, h):
    d = raw_digits()
    padded_d = np.pad(d, [(0, 0), (0, w - RAW_DIGIT_W), (h - RAW_DIGIT_H, 0)], 'edge')
    return tf.expand_dims(tf.constant(padded_d, tf.float32), -1)


def get_bottom_left_digits(classes, tile_size):
    digits = tf.image.grayscale_to_rgb(digits_bottom_left(tile_size//8, tile_size//8))
    digits = tf.image.resize_bilinear(digits, [tile_size, tile_size])
    return tf.gather(digits, tf.minimum(classes, 9))  # correct digits to be printed on the images


def get_bottom_right_digits(classes, tile_size):
    digits = tf.image.grayscale_to_rgb(digits_bottom_right(tile_size//8, tile_size//8))
    digits = tf.image.resize_bilinear(digits, [tile_size, tile_size])
    return tf.gather(digits, tf.minimum(classes, 9))  # correct digits to be printed on the images


def get_top_right_red_white_digits(classes, tile_size):
    digits = digits_top_right(tile_size//8, tile_size//8)
    zeros = tf.zeros(tf.shape(digits))
    digits_red = tf.concat([digits, zeros, zeros], -1)
    digits_white = tf.concat([digits, digits, digits], -1)
    d0,d1,d2,d3,d4,d5,d6,d7,d8,d9 = tf.split(digits_red, 10)
    b0,b1,b2,b3,b4,b5,b6,b7,b8,b9 = tf.split(digits_white, 10)
    # zero is white, other digits are red
    digits = tf.concat([b0,d1,d2,d3,d4,d5,d6,d7,d8,d9], axis=0)
    digits = tf.image.resize_bilinear(digits, [tile_size, tile_size])
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
    white_boxes = tf.image.draw_bounding_boxes(black, box.swap_xy(boxes))
    box_img = tf.concat([white_boxes * r, white_boxes * g, white_boxes * b], axis=3)
    white_boxes = tf.concat([white_boxes, white_boxes, white_boxes], axis=3)
    return tf.where(tf.greater(white_boxes, 0.0), box_img, img)


def debug_image(img, mistakes, target_rois, predicted_rois, predicted_c,
                size_correct, position_correct, all_correct,
                grid_nn, cell_n, tile_size):
    """Writes on top of image tile "img" all the debug data necessary: number of mistakes, detected
    boxes and ground truth boxes. Color codes mis-detections:
        Yellow: correct detection
        Orange: size OK but wrong position
        Purple: position OK but wrong size
        Red: all wrong"""

    debug_img = image_compose(img, get_top_right_red_white_digits(mistakes, tile_size))
    # debug: ground truth boxes in grey
    debug_img = draw_color_boxes(debug_img, target_rois, 0.7, 0.7, 0.7)
    # debug: computed ROIs boxes in shades of yellow
    no_box = tf.zeros(tf.shape(predicted_rois))
    select = tf.stack([predicted_c, predicted_c, predicted_c, predicted_c], axis=-1)
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
    return debug_img