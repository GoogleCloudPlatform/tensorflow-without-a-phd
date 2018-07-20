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

Unit tests for utils_box. To run the tests, execute
python -m unittest discover -s trainer_yolo"""

# TODO: move this under a tests/ directory

import tensorflow as tf
import numpy as np
import unittest
from trainer_yolo.utils_box import *
from trainer_yolo.utils_imgdbg import *
from trainer_yolo.datagen import log_tensor

class BoxRoiUtilsTest(unittest.TestCase):

    def setUp(self):
        self.tile0 = tf.constant([3.0, 2.0, 7.0, 6.0], dtype=tf.float32)  # format x1, y1, x2, y2
        self.roi0 = tf.constant([4.3, 2.0, 7.0, 6.0], dtype=tf.float32)
        self.roi1 = tf.constant([2.5, 2.5, 4.7, 4.2], dtype=tf.float32)
        self.roi2 = tf.constant([3.1, 2.1, 3.2, 2.2], dtype=tf.float32)
        self.roi3 = tf.constant([3.0, 2.1, 3.2, 2.2], dtype=tf.float32)
        self.roi4 = tf.constant([5.0, 2.5, 6.5, 6.0], dtype=tf.float32)
        self.roi256 = tf.constant([120.0, 10.0, 130.0, 20.0], dtype=tf.float32)
        self.rois = tf.constant([[3.1, 1.5, 4.3, 2.7],
                                 [3.0, 2.0, 4.0, 3.0],
                                 [6.0, 5.0, 7.0, 6.0],
                                 [6.0, 5.0, 6.9, 5.9],
                                 [4.5, 3.1, 4.9, 3.5],
                                 [6.1, 5.1, 7.0, 6.0]], dtype=tf.float32)
        self.rois_1 = tf.constant([[3.0, 2.0, 4.0, 3.0]], dtype=tf.float32)
        # zero-sized rois
        self.rois_0, _ = tf.split(tf.constant([[1.0, 1.0, 1.0, 1.0]], dtype=tf.float32), [0, 1], axis=0)
        #[batch, grid_n, grid_n, n, 4] grid 3x3
        self.relative_rois = tf.constant([
            [[[[[0],[0],[0.1]],[[1],[1],[1]]],[[[-1],[-1],[0.5]],[[0],[0],[0.1]]],[[[0.1],[0.1],[0.1]],[[0],[0],[0]]]],
             [[[[-0.1],[0.1],[0.1]],[[0.1],[-0.1],[0.1]]],[[[0],[0],[0]],[[0],[0],[0]]],[[[0],[0],[0]],[[0.2],[0.3],[0.5]]]],
             [[[[0],[0],[0]],[[0],[0],[0]]],[[[0],[0],[0]],[[0],[0],[0]]],[[[0],[0],[0]],[[0],[0],[0]]]]], # batch 1
            [[[[[0],[0],[0]],[[0],[0],[0]]],[[[0],[0],[0]],[[0],[0],[0]]],[[[0],[0],[0]],[[0],[0],[0.1]]]],
             [[[[0.5],[-0.5],[0.3]],[[-0.5],[-0.5],[0.5]]],[[[-0.5],[-0.5],[0.5]],[[-0.5],[-0.5],[0.5]]],[[[0.5],[-0.5],[0.5]],[[0.5],[0.5],[0.5]]]],
             [[[[0],[0],[0]],[[0.1],[0.1],[0.1]]],[[[0],[0],[0]],[[0],[0],[0]]],[[[0.1],[-0.1],[0.1]],[[-0.1],[0],[0.1]]]]]  # batch 2
        ], dtype=tf.float32)

    def test_box_rot90(self):
        rotated_rois_anticlockwise = rot90(self.rois, 6.0)
        rotated_rois_clockwise = rot90(self.rois, 6.0, k=-1)
        rotated_rois_180_1 = rot90(self.rois, 6.0, k=-2)
        rotated_rois_180_2 = rot90(self.rois, 6.0, k=2)
        rotated_rois_id_1 = rot90(self.rois, 6.0, k=0)
        rotated_rois_id_2 = rot90(self.rois, 6.0, k=4)
        flipped_rois_lr = flip_left_right(self.rois, 6.0)
        flipped_rois_ud = flip_up_down(self.rois, 6.0)
        correct1 = np.array([[4.5, 3.1, 3.3, 4.3],
                            [4.0, 3.0, 3.0, 4.0],
                            [1.0, 6.0, 0.0, 7.0],
                            [1.0, 6.0, 0.1, 6.9],
                            [2.9, 4.5, 2.5, 4.9],
                            [0.9, 6.1, 0.0, 7.0]])
        correct2 = np.array([[1.5, 2.9, 2.7, 1.7],
                             [2.0, 3.0, 3.0, 2.0],
                             [5.0, 0.0, 6.0, -1.0],
                             [5.0, 0.0, 5.9, -0.9],
                             [3.1, 1.5, 3.5, 1.1],
                             [5.1, -0.1, 6.0, -1.0]])
        correct3 = np.array([[2.9, 4.5, 1.7, 3.3],
                             [3.0, 4.0, 2.0, 3.0],
                             [0.0, 1.0, -1.0, 0.0],
                             [0.0, 1.0, -0.9, 0.1],
                             [1.5, 2.9, 1.1, 2.5],
                             [-0.1, 0.9, -1.0, 0.0]])
        correct4 = np.array([[3.1, 4.5, 4.3, 3.3],
                             [3.0, 4.0, 4.0, 3.0],
                             [6.0, 1.0, 7.0, 0.0],
                             [6.0, 1.0, 6.9, 0.1],
                             [4.5, 2.9, 4.9, 2.5],
                             [6.1, 0.9, 7.0, 0.0]])
        correct5 = np.array([[2.9, 1.5, 1.7, 2.7],
                             [3.0, 2.0, 2.0, 3.0],
                             [0.0, 5.0, -1.0, 6.0],
                             [0.0, 5.0, -0.9, 5.9],
                             [1.5, 3.1, 1.1, 3.5],
                             [-0.1, 5.1, -1.0, 6.0]])
        with tf.Session() as sess:
            rotated_rois_anticlockwise_ = sess.run(rotated_rois_anticlockwise)
            rotated_rois_clockwise_ = sess.run(rotated_rois_clockwise)
            rotated_rois_180_1_ = sess.run(rotated_rois_180_1)
            rotated_rois_180_2_ = sess.run(rotated_rois_180_2)
            rotated_rois_id_1_ = sess.run(rotated_rois_id_1)
            rotated_rois_id_2_ = sess.run(rotated_rois_id_2)
            unchanged_rois_ = sess.run(self.rois)
            flipped_rois_lr_ = sess.run(flipped_rois_lr)
            flipped_rois_ud_ = sess.run(flipped_rois_ud)
        #print(rotated_rois_anticlockwise_)
        d = np.linalg.norm(rotated_rois_anticlockwise_ - correct2) + \
            np.linalg.norm(rotated_rois_clockwise_ - correct1) + \
            np.linalg.norm(rotated_rois_180_1_ - correct3) + \
            np.linalg.norm(rotated_rois_180_2_ - correct3) + \
            np.linalg.norm(rotated_rois_id_1_ - unchanged_rois_) + \
            np.linalg.norm(rotated_rois_id_2_ - unchanged_rois_) + \
            np.linalg.norm(flipped_rois_lr_ - correct5) + \
            np.linalg.norm(flipped_rois_ud_ - correct4)
        #print(d)
        self.assertTrue(d<1e-5, "test_box_rot90 test failed")

    def test_remove_non_intersecting_rois(self):
        tiles = tf.constant([[0.0, 0.0, 3.0, 3.0],  # tile0
                 [1, -1, 3, 1],         # tile1
                 [4, 3, 7, 8]])          # tile2
        rois = tf.constant([[0.0, 0.0, 1.0, 1.0],   # intersects tile0
                [-1, -1, 5, 5],         # intersects tile0, tile1, tile2
                [-1, -5, 0, 0],         # intersect nothing
                [2, 2, 5, 8]])           # intersects tile0, tile2
        correct = np.array([[[0.0, 0.0, 1.0, 1.0], [-1, -1, 5, 5], [2, 2, 5, 8]],  # tile0 rois
                   [[-1, -1, 5, 5], [0, 0, 0, 0], [0, 0, 0, 0]],           # tile1 rois
                   [[-1, -1, 5, 5], [2, 2, 5, 8], [0, 0, 0, 0]]])          # tile2 rois
        filtered_rois = remove_non_intersecting_rois(tiles, rois, max_per_tile=3)
        with tf.Session() as sess:
            res1, overflow = sess.run(filtered_rois)
            #print(overflow)
        d = np.linalg.norm(res1-correct)
        ovf = np.sum(overflow)
        self.assertTrue(d<1e-6, "remove_non_intersecting_rois test failed")
        self.assertTrue(ovf==0, "remove_non_intersecting_rois overflow test failed")


    def test_rois_in_tile_relative(self):
        tiles = tf.constant([[0.0, 0.0, 3.0, 3.0],  # tile0
                             [1, -1, 4, 2],         # tile1
                             [4, 3, 7, 6]])         # tile2
        rois = tf.constant([[0.0, 0.0, 1.0, 1.0],   # intersects tile0
                            [-1, -1, 5, 5],         # intersects tile0, tile1, tile2
                            [-1, -5, 0, 0],         # intersect nothing
                            [2, 2, 5, 8],           # intersects tile0, tile2
                            [5, 5, 6, 6]])          # intersects tile2
        correct = np.array([[[0.0, 0.0, 1.0, 1.0], [-1, -1, 5, 5], [2, 2, 5, 8]],  # tile0 rois
                            [[-2, 0, 4, 6], [0, 0, 0, 0], [0, 0, 0, 0]],           # tile1 rois
                            [[-5, -4, 1, 2], [-2, -1, 1, 5], [1, 2, 2, 3]]])   / 3.0       # tile2 rois
        correct2 = np.array([[[0.0, 0.0, 1.0, 1.0], [-1, -1, 5, 5]],  # tile0 rois
                            [[-2, 0, 4, 6], [0, 0, 0, 0]],           # tile1 rois
                            [[-5, -4, 1, 2], [-2, -1, 1, 5]]])   / 3.0       # tile2 rois
        filtered_rois = rois_in_tiles_relative(tiles, rois, tile_size=3.0, max_per_tile=3)
        filtered_rois2 = rois_in_tiles_relative(tiles, rois, tile_size=3.0, max_per_tile=2, assert_on_overflow=False)  # this overflows max_per_tile
        with tf.Session() as sess:
            res1 = sess.run(filtered_rois)
            res2 = sess.run(filtered_rois2)
            #print(res1)
            #print(correct)
        d = np.linalg.norm(res1-correct) + np.linalg.norm(res2-correct2)
        self.assertTrue(d<1e-6, "rois_in_tile_relative test failed")

    def test_batch_iou(self):
        rois1 = tf.constant([[1, 1, 4, 3],
                             [2, 3, 7, 5],
                             [3, 4, 4, 5]], dtype=tf.float32)
        rois1b = tf.constant([[1, 1, 4, 3],
                             [2, 3, 7, 5]], dtype=tf.float32)
        rois2 = tf.constant([[1, 1, 4, 3],
                             [2, 3, 7, 5],
                             [3, 4, 4, 5]], dtype=tf.float32)
        rois2b = tf.constant([[1, 1, 4, 3],
                             [2, 3, 7, 5]], dtype=tf.float32)
        rois3 = tf.constant([[1, 1, 3, 3],
                             [2, 3, 7, 5],
                             [3, 4, 4, 5]], dtype=tf.float32)
        rois3b = tf.constant([[1, 1, 3, 3],
                             [2, 3, 7, 5]], dtype=tf.float32)
        norois1 = tf.constant([[1, 1, 1, 3],
                             [2, 3, 7, 3],
                             [3, 4, 4, 4]], dtype=tf.float32)
        norois1b = tf.constant([[1, 1, 1, 3],
                               [2, 3, 7, 3]], dtype=tf.float32)
        norois2 = tf.constant([[10, 10, 12, 12],
                               [2, 3, 2, 3],
                               [-3, -4, -2, -3]], dtype=tf.float32)
        norois2b = tf.constant([[10, 10, 12, 12],
                               [2, 3, 2, 3]], dtype=tf.float32)
        partrois = tf.constant([[1, 1, 3, 3],
                               [2, 3, 2, 3],
                               [-3, -4, -2, -3]], dtype=tf.float32)
        partroisb = tf.constant([[1, 1, 3, 3],
                                [2, 3, 2, 3]], dtype=tf.float32)
        batch_roisA = tf.stack([rois1, rois3, rois1, norois1, rois3], axis=0)
        batch_roisB = tf.stack([rois2, rois2, norois1, norois2, partrois], axis=0)
        batch_roisBb = tf.stack([rois2b, rois2b, norois1b, norois2b, partroisb], axis=0)
        iou1 = IOUCalculator.batch_intersection_over_union(batch_roisA, batch_roisB, tile_size=5)
        iou2 = IOUCalculator.batch_intersection_over_union(batch_roisA, batch_roisBb, tile_size=5)
        correct1 = np.array([1.0, 10.0/12.0, 0.0, 1.0, 4.0/10.0])
        correct2 = np.array([1.0, 10.0/12.0, 0.0, 1.0, 4.0/10.0])
        with tf.Session() as sess:
            res1 = sess.run(iou1)
            res2 = sess.run(iou2)
            #print(res2)
        d1 = np.linalg.norm(res1-correct1)
        d2 = np.linalg.norm(res2-correct2)
        self.assertTrue((d1+d2)<1e-6, "IOUCalculator.batch_iou test failed")

    def test_batch_iou_mean(self):
        ious1 = tf.constant([0, 1, 0.3, 0.9, 0.8], dtype=tf.float32)  # 1s are ignored in the average
        ious2 = tf.constant([0, 0, 0, 0, 0], dtype=tf.float32)
        ious3 = tf.constant([1, 1, 1, 1, 1], dtype=tf.float32) # the average is undefined when 1s are ignored, 1 is returned
        ious4 = tf.constant([1, 1, 1, 0.99, 1], dtype=tf.float32)
        mean1 = IOUCalculator.batch_mean(ious1)
        mean2 = IOUCalculator.batch_mean(ious2)
        mean3 = IOUCalculator.batch_mean(ious3)
        mean4 = IOUCalculator.batch_mean(ious4)
        with tf.Session() as sess:
            res1 = sess.run(mean1)
            res2 = sess.run(mean2)
            res3 = sess.run(mean3)
            res4 = sess.run(mean4)
            #print(res4)
        correct1 = 0.5
        correct2 = 0
        correct3 = 1.0
        correct4 = 0.99
        d1 = res1-correct1
        d2 = res2-correct2
        d3 = res3-correct3
        d4 = np.linalg.norm(res4-correct4)
        #print(d4)
        self.assertTrue(d1==0 and d2==0 and d3==0 and d4<1e-6, "IOUCalculator.batch_mean test failed")



    def test_grid_cell_to_tile_coords(self):
        rel_rois = tf.reshape(self.relative_rois, [2, 3, 3, 2, 3]) # mistake in entering test data, added last dim 1, now removing it
        new_coords = grid_cell_to_tile_coords(rel_rois, grid_n=3, tile_size=6)
        correct = tf.constant([[[[[0.7, 0.7, 1.3, 1.3], [-1., -1., 5., 5.]],
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
        correct = swap_xy(correct)  # the API for grid_cell_to_tile_coords has changed, now producing coordinates in the x1y1x2y2 format. Previously, it was y1x1y2x2.
        with tf.Session() as sess:
            res, cor = sess.run([new_coords, correct])
            #print(res)
            #print(cor)
        d = np.linalg.norm(np.reshape(res, [-1])-np.reshape(cor, [-1]))
        self.assertTrue(d<1e-5, "grid_cell_to_tile_coords test failed")

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
        res = center_in_grid_cell(grid, grid_n=4, cell_w=1, rois=x1y1x2y2_to_cxcyw(self.rois))
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

        res_1 = n_largest_rois_in_cell(self.tile0, self.rois_1, rois_n=1, grid_n=4, n=3)
        res_0 = n_largest_rois_in_cell(self.tile0, self.rois_0, rois_n=0, grid_n=4, n=3)

        correct_0 = np.array([[[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                              [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                              [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                              [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]])

        correct_1 = np.array([[[[3.5, 2.5, 1.0], [0., 0., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                              [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                              [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                              [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]])

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
            out1, out2, out3, out_0, out_1 = sess.run([res1, res2, res3, res_0, res_1])
            d1 = np.linalg.norm(np.reshape(out1, [-1])-np.reshape(correct1, [-1]))
            d2 = np.linalg.norm(np.reshape(out2, [-1])-np.reshape(correct2, [-1]))
            d3 = np.linalg.norm(np.reshape(out3, [-1])-np.reshape(correct3, [-1]))
            d_0 = np.linalg.norm(np.reshape(out_0, [-1])-np.reshape(correct_0, [-1]))
            d_1 = np.linalg.norm(np.reshape(out_1, [-1])-np.reshape(correct_1, [-1]))
        self.assertTrue(d1<1e-6, "n_largest_rois_in_cell generation test failed")
        self.assertTrue(d2<1e-6, "n_largest_rois_in_cell generation test failed")
        self.assertTrue(d3<1e-6, "n_largest_rois_in_cell generation test failed")
        self.assertTrue(d_0<1e-6, "n_largest_rois_in_cell generation test failed")
        self.assertTrue(d_1<1e-6, "n_largest_rois_in_cell generation test failed")

    def test_digits(self):
        correct_tl = np.array([[0, 0, 0, 0, 0, 0],
                               [0, 1, 1, 1, 0, 0],
                               [0, 1, 0, 1, 0, 0],
                               [0, 1, 1, 1, 0, 0],
                               [0, 0, 0, 1, 0, 0],
                               [0, 1, 1, 1, 0, 0],
                               [0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0]], np.uint8)
        correct_bl = np.array([[0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0],
                               [0, 1, 1, 1, 0, 0],
                               [0, 1, 0, 1, 0, 0],
                               [0, 1, 1, 1, 0, 0],
                               [0, 0, 0, 1, 0, 0],
                               [0, 1, 1, 1, 0, 0],
                               [0, 0, 0, 0, 0, 0]], np.uint8)
        correct_br = np.array([[0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0],
                               [0, 0, 1, 1, 1, 0],
                               [0, 0, 1, 0, 1, 0],
                               [0, 0, 1, 1, 1, 0],
                               [0, 0, 0, 0, 1, 0],
                               [0, 0, 1, 1, 1, 0],
                               [0, 0, 0, 0, 0, 0]], np.uint8)
        correct_tr = np.array([[0, 0, 0, 0, 0, 0],
                               [0, 0, 1, 1, 1, 0],
                               [0, 0, 1, 0, 1, 0],
                               [0, 0, 1, 1, 1, 0],
                               [0, 0, 0, 0, 1, 0],
                               [0, 0, 1, 1, 1, 0],
                               [0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0]], np.uint8)
        dtl = digits_top_left(6, 8)
        dtr = digits_top_right(6, 8)
        dbl = digits_bottom_left(6, 8)
        dbr = digits_bottom_right(6, 8)
        with tf.Session() as sess:
            tl, tr, bl, br = sess.run([dtl, dtr, dbl, dbr])
            tl9 = tl[9]
            tr9 = tr[9]
            bl9 = bl[9]
            br9 = br[9]
            d1 = np.linalg.norm(np.reshape(tl9, [-1])-np.reshape(correct_tl, [-1]))
            d2 = np.linalg.norm(np.reshape(tr9, [-1])-np.reshape(correct_tr, [-1]))
            d3 = np.linalg.norm(np.reshape(bl9, [-1])-np.reshape(correct_bl, [-1]))
            d4 = np.linalg.norm(np.reshape(br9, [-1])-np.reshape(correct_br, [-1]))
            self.assertTrue(d1+d2+d3+d4<1e-6, "digits test failed")

if __name__ == '__main__':
    unittest.main()