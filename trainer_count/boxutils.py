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
    # force broadcasting
    px1 = tf.add(px1, qx1-qx1)
    px2 = tf.add(px2, qx2-qx2)
    zeros = tf.subtract(px1, px1)

    inter1 = tf.greater(px2, qx1)
    inter2 = tf.greater(qx2, px1)
    inter = tf.logical_and(inter1, inter2)
    inter1 = tf.greater(qx1, px1)
    inter_x = tf.where(inter1, qx1, px1)
    inter_w = tf.where(inter1, px2-qx1, qx2-px1)
    inter_w = tf.where(inter, inter_w, zeros)
    return inter_x, inter_w

def boxintersect(primeroi, rois, min_intersect=0):
    # primeroi: single region shape=[4] Tensor: [x1, y1, x2, y2]
    # rois: multiple regions shape=[n, 4] Tensor: n x [x1, y1, x2, y2]
    # min_intersect: value between 0 and 1.
    #   area(intersection) > min_intesect * min(area(primeroi), ara(roi)) to count as intersection
    # return value: [n] Tensor type bool indicating which rois intersect the primeroi

    px1, py1, px2, py2 = tf.unstack(primeroi, axis=0)
    x1, y1, x2, y2 = tf.unstack(rois, axis=1)
    inter_x, inter_w = one_d_intersect(px1, px2, x1, x2)
    inter_y, inter_h = one_d_intersect(py1, py2, y1, y2)
    inter_area = inter_w * inter_h
    parea = (px2-px1)*(py2-py1)
    areas = (x2-x1)*(y2-y1)
    min_areas = tf.minimum(areas, parea)
    inter = tf.greater(inter_area, min_areas*min_intersect)
    return inter


