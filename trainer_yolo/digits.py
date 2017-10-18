# encoding: UTF-8
# Copyright 2016 Google.com
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

import numpy as np
import tensorflow as tf

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
          [0, 0, 0, 0, 0]],
         [[0, 0, 0, 0, 0],
          [0, 0, 1, 0, 0],
          [0, 1, 1, 0, 0],
          [0, 0, 1, 0, 0],
          [0, 0, 1, 0, 0],
          [0, 0, 1, 0, 0],
          [0, 0, 0, 0, 0]],
         [[0, 0, 0, 0, 0],
          [0, 1, 1, 1, 0],
          [0, 0, 0, 1, 0],
          [0, 1, 1, 1, 0],
          [0, 1, 0, 0, 0],
          [0, 1, 1, 1, 0],
          [0, 0, 0, 0, 0]],
         [[0, 0, 0, 0, 0],
          [0, 1, 1, 1, 0],
          [0, 0, 0, 1, 0],
          [0, 1, 1, 1, 0],
          [0, 0, 0, 1, 0],
          [0, 1, 1, 1, 0],
          [0, 0, 0, 0, 0]],
         [[0, 0, 0, 0, 0],
          [0, 0, 0, 1, 0],
          [0, 0, 1, 1, 0],
          [0, 1, 0, 1, 0],
          [0, 1, 1, 1, 0],
          [0, 0, 0, 1, 0],
          [0, 0, 0, 0, 0]],
         [[0, 0, 0, 0, 0],
          [0, 1, 1, 1, 0],
          [0, 1, 0, 0, 0],
          [0, 1, 1, 1, 0],
          [0, 0, 0, 1, 0],
          [0, 1, 1, 1, 0],
          [0, 0, 0, 0, 0]],
         [[0, 0, 0, 0, 0],
          [0, 1, 1, 1, 0],
          [0, 1, 0, 0, 0],
          [0, 1, 1, 1, 0],
          [0, 1, 0, 1, 0],
          [0, 1, 1, 1, 0],
          [0, 0, 0, 0, 0]],
         [[0, 0, 0, 0, 0],
          [0, 1, 1, 1, 0],
          [0, 0, 0, 1, 0],
          [0, 0, 1, 0, 0],
          [0, 1, 0, 0, 0],
          [0, 1, 0, 0, 0],
          [0, 0, 0, 0, 0]],
         [[0, 0, 0, 0, 0],
          [0, 1, 1, 1, 0],
          [0, 1, 0, 1, 0],
          [0, 1, 1, 1, 0],
          [0, 1, 0, 1, 0],
          [0, 1, 1, 1, 0],
          [0, 0, 0, 0, 0]],
         [[0, 0, 0, 0, 0],
          [0, 1, 1, 1, 0],
          [0, 1, 0, 1, 0],
          [0, 1, 1, 1, 0],
          [0, 0, 0, 1, 0],
          [0, 1, 1, 1, 0],
          [0, 0, 0, 0, 0]]], np.uint8)
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
