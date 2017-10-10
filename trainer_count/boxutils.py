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

import tensorflow as tf

def one_d_intersect(px1, px2, qx1, qx2):
    # this assumes px2>=px1 and qx2>=qx1

    # force broadcasting
    px1 = tf.add(px1, qx1-qx1)
    px2 = tf.add(px2, qx2-qx2)
    zeros = tf.subtract(px1, px1)

    interA = tf.greater(px1, qx1)
    interB = tf.greater(px2, qx1)
    interC = tf.greater(px2, qx2)
    interD = tf.greater(qx2, px1)
    inter = tf.logical_and(interB, interD)

    inter_x1 = tf.where(tf.logical_and(tf.logical_not(interA), interB), qx1, px1)
    inter_x2 = tf.where(tf.logical_and(interC, interD), qx2, px2)
    inter_w = inter_x2 - inter_x1
    inter_w = tf.where(inter, inter_w, zeros)
    return inter, inter_x1, inter_w

def boxintersect(primeroi, rois, min_intersect=0):
    # primeroi: single region shape=[4] Tensor: [x1, y1, x2, y2]
    # rois: multiple regions shape=[n, 4] Tensor: n x [x1, y1, x2, y2]
    # min_intersect: value between 0 and 1.
    #   area(intersection) >= min_intersect * min(area(primeroi), area(roi)) to count as intersection
    # return value: [n] Tensor type bool indicating which rois intersect the primeroi

    px1, py1, px2, py2 = tf.unstack(primeroi, axis=0)
    x1, y1, x2, y2 = tf.unstack(rois, axis=1)
    is_inter_x, inter_x, inter_w = one_d_intersect(px1, px2, x1, x2)
    is_inter_y, inter_y, inter_h = one_d_intersect(py1, py2, y1, y2)
    inter_area = inter_w * inter_h
    parea = (px2-px1)*(py2-py1)
    areas = (x2-x1)*(y2-y1)
    min_areas = tf.minimum(areas, parea)
    inter = tf.logical_and(is_inter_x, is_inter_y)
    inter_with_area = tf.greater_equal(inter_area, min_areas*min_intersect)
    return tf.logical_and(inter, inter_with_area)

