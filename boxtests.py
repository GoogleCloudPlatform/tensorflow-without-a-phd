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
import numpy as np
import unittest
from trainer_yolo.boxutils import *

class BoxRoiUtilsTest(unittest.TestCase):

    def setUp(self):
        self.tile0 = tf.constant([3.0, 2.0, 7.0, 6.0], dtype=tf.float32)  # format x1, y1, x2, y2
        self.roi0 = tf.constant([4.3, 2.0, 7.0, 6.0], dtype=tf.float32)
        self.roi1 = tf.constant([2.5, 2.5, 4.7, 4.2], dtype=tf.float32)
        self.roi2 = tf.constant([3.1, 2.1, 3.2, 2.2], dtype=tf.float32)
        self.roi3 = tf.constant([3.0, 2.1, 3.2, 2.2], dtype=tf.float32)
        self.roi4 = tf.constant([5.0, 2.5, 6.5, 6.0], dtype=tf.float32)
        self.rois = tf.constant([[3.1, 1.5, 4.3, 2.7],
                                 [3.0, 2.0, 4.0, 3.0],
                                 [6.0, 5.0, 7.0, 6.0],
                                 [6.0, 5.0, 6.9, 5.9],
                                 [4.5, 3.1, 4.9, 3.5],
                                 [6.1, 5.1, 7.0, 6.0]], dtype=tf.float32)
        self.rois2 = tf.constant([[3.0, 2.0, 4.0, 3.0]], dtype=tf.float32)
        #[batch, grid_n, grid_n, n, 4] grid 3x3
        self.relative_rois = tf.constant([
            [[[[[0],[0],[0.1]],[[1],[1],[1]]],[[[-1],[-1],[0.5]],[[0],[0],[0.1]]],[[[0.1],[0.1],[0.1]],[[0],[0],[0]]]],
             [[[[-0.1],[0.1],[0.1]],[[0.1],[-0.1],[0.1]]],[[[0],[0],[0]],[[0],[0],[0]]],[[[0],[0],[0]],[[0.2],[0.3],[0.5]]]],
             [[[[0],[0],[0]],[[0],[0],[0]]],[[[0],[0],[0]],[[0],[0],[0]]],[[[0],[0],[0]],[[0],[0],[0]]]]], # batch 1
            [[[[[0],[0],[0]],[[0],[0],[0]]],[[[0],[0],[0]],[[0],[0],[0]]],[[[0],[0],[0]],[[0],[0],[0.1]]]],
             [[[[0.5],[-0.5],[0.3]],[[-0.5],[-0.5],[0.5]]],[[[-0.5],[-0.5],[0.5]],[[-0.5],[-0.5],[0.5]]],[[[0.5],[-0.5],[0.5]],[[0.5],[0.5],[0.5]]]],
             [[[[0],[0],[0]],[[0.1],[0.1],[0.1]]],[[[0],[0],[0]],[[0],[0],[0]]],[[[0.1],[-0.1],[0.1]],[[-0.1],[0],[0.1]]]]]  # batch 2
        ], dtype=tf.float32)

    def test_grid_cell_to_tile_coords(self):
        rel_rois = tf.reshape(self.relative_rois, [2, 3, 3, 2, 3]) # mistake in entering test data, adde last dim 1, now removing it
        new_coords = grid_cell_to_tile_coords(rel_rois, grid_n=3, tile_size=6)
        correct = np.array([[[[[0.7, 0.7, 1.3, 1.3], [-1., -1., 5., 5.]],
                              [[-1.5, 0.5, 1.5, 3.5], [0.7, 2.7, 1.3, 3.3]],
                              [[0.8, 4.8, 1.4, 5.4], [1., 5., 1., 5.]]],
                             [[[2.8, 0.6, 3.4, 1.2], [2.6, 0.8, 3.2, 1.4]],
                              [[3., 3., 3., 3.], [3., 3., 3., 3.]],
                              [[3., 5., 3., 5.], [1.8, 3.7, 4.8, 6.7]]],
                             [[[5., 1., 5., 1.], [5., 1., 5., 1.]],
                              [[5., 3., 5., 3.], [5., 3., 5., 3.]],
                              [[5., 5., 5., 5.], [5., 5., 5., 5.]]]],
                            [[[[1., 1., 1., 1.], [1., 1., 1., 1.]],
                              [[1., 3., 1., 3.], [1., 3., 1., 3.]],
                              [[1., 5., 1., 5.], [0.7, 4.7, 1.3, 5.3]]],
                             [[[1.6, 0.6, 3.4, 2.4], [1., -1., 4., 2.]],
                              [[1., 1., 4., 4.], [1., 1., 4., 4.]],
                              [[1., 4., 4., 7.], [2., 4., 5., 7.]]],
                             [[[5., 1., 5., 1.], [4.8, 0.8, 5.4, 1.4]],
                              [[5., 3., 5., 3.], [5., 3., 5., 3.]],
                              [[4.6, 4.8, 5.2, 5.4], [4.7, 4.6, 5.3, 5.2]]]]])
        with tf.Session() as sess:
            res = sess.run(new_coords)
        d = np.linalg.norm(np.reshape(res, [-1])-np.reshape(correct, [-1]))
        self.assertTrue(d<1e-6, "grid_cell_to_tile_coords test failed")

    def test_make_rois_tile_cell_relative(self):
        rois_n = tf.shape(self.rois)[0]
        tiled_rois = n_largest_rois_in_cell(self.tile0, self.rois, rois_n=rois_n, grid_n=4, n=3)
        relative_rois = make_rois_tile_cell_relative(self.tile0, tiled_rois, grid_n=4)
        with tf.Session() as sess:
            res, original = sess.run([relative_rois, tiled_rois])
        correct = np.array([[[[0.4, -0.8, 0.3], [0.0, 0.0, 0.25], [0.0, 0.0, 0.0]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                               [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0.4, -0.4, 0.1], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                               [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                               [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.25], [-0.1, -0.1, 0.225],
                                 [0.1, 0.1, 0.225]]]])
        d = np.linalg.norm(np.reshape(res, [-1])-np.reshape(correct, [-1]))
        self.assertTrue(d<1e-6, "make_rois_tile_cell_relative test failed")

    def test_boxintersect(self):
        ints1 = boxintersect(self.roi0, self.rois)
        ints2 = boxintersect(self.roi1, self.rois)
        ints3 = boxintersect(self.roi1, self.rois, 0.4999)  # 50% intersect min to count
        ints4 = boxintersect(self.roi2, self.rois)
        ints5 = boxintersect(self.roi3, self.rois)
        ints6 = boxintersect(self.roi3, self.rois, 0.51)
        ints7 = boxintersect(self.roi3, self.rois, 0.49)
        ints8 = boxintersect(self.roi4, self.rois)
        ints9 = boxintersect(self.roi4, self.rois, 0.49)
        with tf.Session() as sess:
            res1 = sess.run(ints1)
            res2 = sess.run(ints2)
            res3 = sess.run(ints3)
            res4 = sess.run(ints4)
            res5 = sess.run(ints5)
            res6 = sess.run(ints6)
            res7 = sess.run(ints7)
            res8 = sess.run(ints8)
            res9 = sess.run(ints9)
        lcorrect1 = np.array([False, False, True, True, True, True])
        lcorrect2 = np.array([True, True, False, False, True, False])
        lcorrect3 = np.array([False, True, False, False, True, False])
        lcorrect4 = np.array([True, True, False, False, False, False])
        lcorrect5 = np.array([True, True, False, False, False, False])
        lcorrect6 = np.array([False, True, False, False, False, False])
        lcorrect7 = np.array([True, True, False, False, False, False])
        lcorrect8 = np.array([False, False, True, True, False, True])
        lcorrect9 = np.array([False, False, True, True, False, False])
        self.assertTrue(np.array_equal(res1, lcorrect1), "Box intersection test failed")
        self.assertTrue(np.array_equal(res2, lcorrect2), "Box intersection test failed")
        self.assertTrue(np.array_equal(res3, lcorrect3), "Box intersection test failed")
        self.assertTrue(np.array_equal(res4, lcorrect4), "Box intersection test failed")
        self.assertTrue(np.array_equal(res5, lcorrect5), "Box intersection test failed")
        self.assertTrue(np.array_equal(res6, lcorrect6), "Box intersection test failed")
        self.assertTrue(np.array_equal(res7, lcorrect7), "Box intersection test failed")
        self.assertTrue(np.array_equal(res8, lcorrect8), "Box intersection test failed")
        self.assertTrue(np.array_equal(res9, lcorrect9), "Box intersection test failed")

    def test_gen_grid(self):
        grid = gen_grid(grid_n=4)
        grid = size_and_move_grid(grid, cell_w=3, origin=tf.constant([0.0, 0.0]))
        grid2 = np.array([[[0.0, 0.0], [3.0, 0.0], [6.0, 0.0], [9.0, 0.0]],
                          [[0.0, 3.0], [3.0, 3.0], [6.0, 3.0], [9.0, 3.0]],
                          [[0.0, 6.0], [3.0, 6.0], [6.0, 6.0], [9.0, 6.0]],
                          [[0.0, 9.0], [3.0, 9.0], [6.0, 9.0], [9.0, 9.0]]])
        with tf.Session() as sess:
            res = sess.run(grid)
        self.assertTrue(np.array_equal(res, grid2), "Grid generation test failed")

    def test_center_in_grid_cell(self):
        grid = gen_grid(grid_n=4)
        grid = size_and_move_grid(grid, cell_w=1, origin=self.tile0[0:2])
        res = center_in_grid_cell(grid, grid_n=4, cell_w=1, rois=cxyw_rois(self.rois))
        with tf.Session() as sess:
            res, grid = sess.run([res, grid])
            grid2 = np.array([[[3.0, 2.0], [4.0, 2.0], [5.0, 2.0], [6.0, 2.0]],
                              [[3.0, 3.0], [4.0, 3.0], [5.0, 3.0], [6.0, 3.0]],
                              [[3.0, 4.0], [4.0, 4.0], [5.0, 4.0], [6.0, 4.0]],
                              [[3.0, 5.0], [4.0, 5.0], [5.0, 5.0], [6.0, 5.0]]])
            centers = np.array([[[True, True, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                                [[False, False, False, False, False, False], [False, False, False, False, True, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                                [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False]],
                                [[False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, False, False, False, False], [False, False, True, True, False, True]]])
            self.assertTrue(np.array_equal(grid, grid2), "Grid generation test failed")
            self.assertTrue(np.array_equal(res, centers), "ROI center tests on grid failed")


    def test_n_largest_rois_in_cell(self):
        rois_n = tf.shape(self.rois)[0]
        res3 = n_largest_rois_in_cell(self.tile0, self.rois, rois_n=rois_n, grid_n=4, n=3)
        res2 = n_largest_rois_in_cell(self.tile0, self.rois, rois_n=rois_n, grid_n=4, n=2)
        res1 = n_largest_rois_in_cell(self.tile0, self.rois, rois_n=rois_n, grid_n=4, n=1)

        correct1 = np.array([[[[3.7, 2.1, 1.2]],[[0.0, 0.0, 0.0]],[[0.0, 0.0, 0.0]],[[0.0, 0.0, 0.0]]],
                            [[[0.0, 0.0, 0.0]],[[4.7, 3.3, 0.4]],[[0.0, 0.0, 0.0]],[[0.0, 0.0, 0.0]]],
                            [[[0.0, 0.0, 0.0]],[[0.0, 0.0, 0.0]],[[0.0, 0.0, 0.0]],[[0.0, 0.0, 0.0]]],
                            [[[0.0, 0.0, 0.0]],[[0.0, 0.0, 0.0]],[[0.0, 0.0, 0.0]],[[6.5, 5.5, 1.0]]]])

        correct2 = np.array([[[[3.7, 2.1, 1.2],[3.5, 2.5, 1.0]],
                              [[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]],
                              [[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]],
                              [[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]]],
                             [[[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]],
                              [[4.7, 3.3, 0.4],[0.0, 0.0, 0.0]],
                              [[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]],
                              [[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]]],
                             [[[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]],
                              [[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]],
                              [[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]],
                              [[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]]],
                             [[[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]],
                              [[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]],
                              [[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]],
                              [[6.5, 5.5, 1.0],[6.45, 5.45, 0.9]]]])

        correct3 = np.array([[[[3.7, 2.1, 1.2],[3.5, 2.5, 1.0],[0.0, 0.0, 0.0]],
                              [[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]],
                              [[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]],
                              [[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]]],
                             [[[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]],
                              [[4.7, 3.3, 0.4],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]],
                              [[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]],
                              [[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]]],
                             [[[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]],
                              [[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]],
                              [[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]],
                              [[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]]],
                             [[[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]],
                              [[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]],
                              [[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]],
                              [[6.5, 5.5, 1.0],[6.45, 5.45, 0.9],[6.55, 5.55, 0.9]]]])

        with tf.Session() as sess:
            out1, out2, out3 = sess.run([res1, res2, res3])
            d1 = np.linalg.norm(np.reshape(out1, [-1])-np.reshape(correct1, [-1]))
            d2 = np.linalg.norm(np.reshape(out2, [-1])-np.reshape(correct2, [-1]))
            d3 = np.linalg.norm(np.reshape(out3, [-1])-np.reshape(correct3, [-1]))
        self.assertTrue(d1<1e-6, "n_largest_rois_in_cell generation test failed")
        self.assertTrue(d2<1e-6, "n_largest_rois_in_cell generation test failed")
        self.assertTrue(d3<1e-6, "n_largest_rois_in_cell generation test failed")



