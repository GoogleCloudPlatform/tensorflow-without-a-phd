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

Utility functions for rectangle manipulation in Tensorflow. Unit tests
are available for all functions."""

from builtins import zip
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
    inter_w = tf.where(inter, inter_w, zeros)  # for consistency
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

def gen_grid(grid_n):
    cell_x = tf.range(0, grid_n, dtype=tf.float32)
    cell_x = tf.tile(tf.expand_dims(cell_x, axis=0), [grid_n, 1])
    cell_x = cell_x
    cell_y = tf.range(0, grid_n, dtype=tf.float32)
    cell_y = tf.tile(tf.expand_dims(cell_y, axis=0), [grid_n, 1])
    cell_y = tf.transpose(cell_y)
    cell_y = cell_y
    grid = tf.stack([cell_x, cell_y], axis=2)  # shape [grid_n, grid_n, 2]
    return grid

def size_and_move_grid(grid, cell_w, origin):
    return grid * cell_w + origin

def x1y1x2y2_to_cxcyw(rois):
    rois_x1, rois_y1, rois_x2, rois_y2 = tf.unstack(rois, axis=1)  # rois shape [n, 4]
    # center coordinates of the roi
    rois_x = (rois_x1 + rois_x2) / 2.0
    rois_y = (rois_y1 + rois_y2) / 2.0
    rois_w = (rois_x2 - rois_x1)
    rois = tf.stack([rois_x, rois_y, rois_w], axis=1) # rois shape [rois_n, 3]
    return rois

def xyw_to_x1y1x2y2(rois):
    rois_x1, rois_y1, rois_w = tf.unstack(rois, axis=1)  # rois shape [n, 3]
    rois_x2 = rois_x1 + rois_w
    rois_y2 = rois_y1 + rois_w
    rois = tf.stack([rois_x1, rois_y1, rois_x2, rois_y2], axis=1) # rois shape [n, 3]
    return rois

def reshape_rois(rois, grid_n):
    cross_rois = tf.expand_dims(tf.expand_dims(rois, axis=0), axis=0)
    cross_rois = tf.tile(cross_rois, [grid_n, grid_n, 1, 1]) # shape [grid_n, grid_n, rois_n, 3]]
    return cross_rois

# returns set of booleans stating if ROI is centered in grid cell
# grid cells coordinates x,y represent top left corner of cell (not center)
# if expand>1.0, expands cells before applying condition
def center_in_grid_cell(grid, grid_n, cell_w, rois, expand=1.0):
    cross_rois = reshape_rois(rois, grid_n) # shape [grid_n, grid_n, rois_n, 3]]
    cross_rois_cx, cross_rois_cy, cross_rois_w = tf.unstack(cross_rois, axis=-1)
    grid_x, grid_y = tf.unstack(grid, axis=-1)
    has_center_x = tf.logical_and(tf.greater_equal(cross_rois_cx, tf.expand_dims(grid_x-(expand-1.0)*cell_w, -1)),  # broadcast !
                                  tf.less(cross_rois_cx, tf.expand_dims(grid_x+expand*cell_w, -1)))    # broadcast ! and broadcast !
    has_center_y = tf.logical_and(tf.greater_equal(cross_rois_cy, tf.expand_dims(grid_y-(expand-1.0)*cell_w, -1)),  # broadcast !
                                  tf.less(cross_rois_cy, tf.expand_dims(grid_y+expand*cell_w, -1)))    # broadcast ! and broadcast !
    has_center = tf.logical_and(has_center_x, has_center_y) # shape [grid_n, grid_n, rois_n]
    return has_center

# returns set of booleans stating if ROI is centered in grid cell periphery
# expand must be > 1.0 for this function to return positive results
# True for rois centered in expanded cell but not in non-expanded cell.
def center_in_grid_cell_periphery(grid, grid_n, cell_w, rois, expand=1.0):
    has_center = center_in_grid_cell(grid, grid_n, cell_w, rois, expand=1.0)
    has_center_expanded = center_in_grid_cell(grid, grid_n, cell_w, rois, expand=expand)
    has_center_peri = tf.logical_and(has_center_expanded, tf.logical_not(has_center))
    return has_center_peri

def gen_grid_for_tile(tile, grid_n):
    tile_x1, tile_y1, tile_x2, tile_y2 = tf.unstack(tile, axis=0)  # tile shape [4]
    cell_w = (tile_x2 - tile_x1) / grid_n
    grid = gen_grid(grid_n)
    grid = size_and_move_grid(grid, cell_w, [tile_x1, tile_y1])
    return grid, cell_w


# Splits the tile into grid_n x grid_n cells.
# For each cell, computes the n largest rois that are centered in the cell.
# Returns them ordered by decreasing size. Output shape [grid_n, grid_n, n, 3]
# (For now also converts rectangular ROIs to square ones.)
# If no roi centered in a cell, returns empty roi (0,0,0) for that cell.
# Supports alternative comparison types:
# comparator="largest_w": largest roi by width
# comparator="furthest_from_center": roi furthest from cell center
# comparator="closest_to_center": roi closest to cell center
def n_largest_rois_in_cell(tile, rois, rois_n, grid_n, n, comparator="largest_w", expand=1.0):

    # handle the case of rois_n == 0 by creating one dummy empty roi, otherwise the code will not work with rois_n=0
    rois, rois_n = tf.cond(tf.equal(rois_n, 0),
                           true_fn=lambda: (tf.constant([[0.0, 0.0, 0.0, 0.0]]), tf.constant(1)),
                           false_fn=lambda: (rois, rois_n))

    grid, cell_w = gen_grid_for_tile(tile, grid_n)

    # grid shape [grid_n, grid_n, 2]
    # rois shape [rois_n, 3]

    rois = x1y1x2y2_to_cxcyw(rois)
    cross_rois = reshape_rois(rois, grid_n)  # shape [grid_n, grid_n, rois_n, 3]]
    cross_rois_cx, cross_rois_cy, cross_rois_w = tf.unstack(cross_rois, axis=-1) # shape [grid_n, grid_n, rois_n]]
    has_center = center_in_grid_cell(grid, grid_n, cell_w, rois, expand=expand)

    grid_centers = (grid + grid + cell_w) / 2.0  # shape [grid_n, grid_n, 2]
    g_cx, g_cy = tf.unstack(grid_centers, axis=-1)  # shape [grid_n, grid_n]
    g_cx = tf.expand_dims(g_cx, axis=-1) # force broadcasting on correct axis
    g_cy = tf.expand_dims(g_cy, axis=-1)

    # iterate on largest a fixed number of times to get N largest
    n_largest = []
    zeros = tf.zeros(shape=[grid_n, grid_n, 3])
    for i in range(n):
        any_roi_in_cell = tf.reduce_any(has_center, axis=2) # shape [grid_n, grid_n]
        if comparator=="largest_w":
            largest_indices = tf.argmax(tf.cast(has_center, tf.float32) * cross_rois_w, axis=2)  # shape [grid_n, grid_n]
        elif comparator=="furthest_from_center":
            d_from_cell_center = tf.abs(cross_rois_cx - g_cx) + tf.abs(cross_rois_cy - g_cy)
            largest_indices = tf.argmax(tf.cast(has_center, tf.float32) * d_from_cell_center, axis=2)  # shape [grid_n, grid_n]
        elif comparator=="closest_to_center":
            d_from_cell_center = tf.abs(cross_rois_cx - g_cx) + tf.abs(cross_rois_cy - g_cy)
            ones = tf.ones(tf.shape(d_from_cell_center))
            largest_indices = tf.argmin(tf.where(has_center, d_from_cell_center, 1000*ones), axis=2)  # shape [grid_n, grid_n]
        # as of TF1.3 can use tf.gather(axis=2)
        rs_largest_indices = tf.reshape(largest_indices, [grid_n*grid_n])
        rs_largest_indices = tf.unstack(rs_largest_indices, axis=0)  # list
        rs_cross_rois = tf.reshape(cross_rois, [grid_n*grid_n, rois_n, 3])
        rs_cross_rois = tf.unstack(rs_cross_rois, axis=0) # list
        rs_largest_roi_in_cell = [tf.gather(cr, li) for cr, li in zip(rs_cross_rois, rs_largest_indices)]
        largest_roi_in_cell = tf.stack(rs_largest_roi_in_cell, axis=0)  # shape [grid_n * grid_n, 3]
        largest_roi_in_cell = tf.reshape(largest_roi_in_cell, [grid_n, grid_n, 3]) # shape [grid_n, grid_n, 3]
        # cells that do not have a roi in them, set their "largest roi in cell" to (x=0,y=0,w=0)
        any_roi_in_cell = tf.tile(tf.expand_dims(any_roi_in_cell, axis=-1), [1, 1, 3])  # shape [grid_n, grid_n, 3]
        largest_roi_in_cell = tf.where(any_roi_in_cell, largest_roi_in_cell, zeros) # shape [grid_n, grid_n, 3]
        n_largest.append(largest_roi_in_cell)
        # zero-out the largest element per cell to get the next largest on the next iteration
        zero_mask = tf.logical_not(tf.cast(tf.one_hot(largest_indices, rois_n), dtype=tf.bool))
        has_center = tf.logical_and(has_center, zero_mask)
    n_largest = tf.stack(n_largest, axis=2)  # shape [grid_n, grid_n, n, 3]
    return n_largest  # shape [grid_n, grid_n, n, 3]


def make_rois_tile_cell_relative(tile, tiled_rois, grid_n):
    grid, cell_w = gen_grid_for_tile(tile, grid_n)
    tile_w = cell_w * grid_n

    # tiled_rois shape [grid_n, grid_n, cell_n, 3]

    # compute grid cell centers
    grid_centers = (grid + grid + cell_w) / 2.0  # shape [grid_n, grid_n, 2]

    gc_x, gc_y = tf.unstack(grid_centers, axis=-1)  # shape [grid_n, grid_n]
    # force broadcasting on correct axis
    gc_x = tf.expand_dims(gc_x, axis=-1)
    gc_y = tf.expand_dims(gc_y, axis=-1)
    tr_x, tr_y, tr_w = tf.unstack(tiled_rois, axis=-1) # shape [grid_n, grid_n, cell_n]

    ctr_x = (tr_x - gc_x) / (cell_w/2.0)  # constrain x within [-1, 1] in cell center relative coordinates
    ctr_y = (tr_y - gc_y) / (cell_w/2.0)  # constrain y within [-1, 1] in cell center relative coordinates
    ctr_w = tr_w / tile_w  # constrain w within [0, 1] in tile-relative coordinates

    # leave x, y coordinates unchanged (as 0) if the width is zero (empty box)
    ctr_x = tf.where(tf.greater(tr_w, 0), ctr_x, tr_x)
    ctr_y = tf.where(tf.greater(tr_w, 0), ctr_y, tr_x)

    rois = tf.stack([ctr_x, ctr_y, ctr_w], axis=-1)
    return rois


def n_largest_rois_in_cell_relative(tile, rois, rois_n, grid_n, n, comparator="largest_w", expand=1.0):
    rois = n_largest_rois_in_cell(tile, rois, rois_n, grid_n, n, comparator=comparator, expand=expand)
    rois = make_rois_tile_cell_relative(tile, rois, grid_n)
    return rois


def n_experimental_roi_selection_strategy(tile, rois, rois_n, grid_n, n, cell_grow):
    assert n == 2  # only implemented for CELL_B=2
    normal_rois = n_largest_rois_in_cell_relative(tile, rois, rois_n, grid_n, n, comparator="closest_to_center", expand=1.0)
    periph_rois = n_largest_rois_in_cell_relative(tile, rois, rois_n, grid_n, n, comparator="closest_to_center", expand=1.0*cell_grow)

    # TODO: count number of non-zero rois in both, then use decision table
    # normal_rois   periph_rois   result
    #    0     0      0     0      0    0   (a0)
    #    x     0      0     0      x    x   (a0)
    #    x     y      0     0      x    y   (a1)
    #    0     0      z     0      z    z   (a2)
    #    0     0      z     t      z    t   (a3)
    #    x     0      z     0      x    z   (a4)
    #    x     y      z     0      x    y   (a1)
    #    x     0      z     t      x    z   (a4)
    #    x     y      z     t      x    y   (a1)

    def roi_select(rois):
        r1, r2, p1, p2 = tf.unstack(rois, axis=0)  # result shape [3]
        a0 = tf.stack([r1, r1])
        a1 = tf.stack([r1, r2])
        a2 = tf.stack([p1, p1])
        a3 = tf.stack([p1, p2])
        a4 = tf.stack([r1, p1])
        a5 = tf.stack([p2, p2])
        a6 = tf.stack([r1, p2])
        a7 = tf.stack([r2, r2])
        a8 = tf.stack([r2, p2])
        a9 = tf.stack([r2, p1])
        _, _, w = tf.unstack(rois, axis=1)  # result shape [4]
        nz = tf.greater(w, 0)
        zero = tf.zeros(tf.shape(a0))
        r = tf.where(tf.reduce_all(tf.equal(nz, [False, False, False, False])), a0, zero)
        r = tf.where(tf.reduce_all(tf.equal(nz, [False, False, False, True])), a5, r)  # cannot happen
        r = tf.where(tf.reduce_all(tf.equal(nz, [False, False, True, False])), a2, r)
        r = tf.where(tf.reduce_all(tf.equal(nz, [False, False, True, True])), a3, r)
        r = tf.where(tf.reduce_all(tf.equal(nz, [False, True, False, False])), a7, r)  # cannot happen
        r = tf.where(tf.reduce_all(tf.equal(nz, [False, True, False, True])), a8, r)  # cannot happen
        r = tf.where(tf.reduce_all(tf.equal(nz, [False, True, True, False])), a9, r)  # cannot happen
        r = tf.where(tf.reduce_all(tf.equal(nz, [False, True, True, True])), a9, r)  # cannot happen
        r = tf.where(tf.reduce_all(tf.equal(nz, [True, False, False, False])), a0, r)
        r = tf.where(tf.reduce_all(tf.equal(nz, [True, False, False, True])), a6, r)  # yes, can happen
        r = tf.where(tf.reduce_all(tf.equal(nz, [True, False, True, False])), a4, r)
        r = tf.where(tf.reduce_all(tf.equal(nz, [True, False, True, True])), a4, r)
        r = tf.where(tf.reduce_all(tf.equal(nz, [True, True, False, False])), a1, r)
        r = tf.where(tf.reduce_all(tf.equal(nz, [True, True, False, True])), a1, r)  # cannot happen
        r = tf.where(tf.reduce_all(tf.equal(nz, [True, True, True, False])), a1, r)
        r = tf.where(tf.reduce_all(tf.equal(nz, [True, True, True, True])), a1, r)
        return r

    rsnormal_rois = tf.reshape(normal_rois, [grid_n * grid_n, n, 3])
    rx, ry, rw = tf.unstack(rsnormal_rois, axis=-1)
    rsperiph_rois = tf.reshape(periph_rois, [grid_n * grid_n, n, 3])
    px, py, pw = tf.unstack(rsperiph_rois, axis=-1)
    roi_exclude = tf.equal(rw, pw)
    zero = tf.zeros_like(pw)
    pw = tf.where(roi_exclude, zero, pw)  # keep in periphery rois only rois that are NOT in normal rois, i.e. rois further than 1 cell radius
    rsperiph_rois = tf.stack([px, py, pw], axis=2)
    rscombined_rois = tf.concat([rsnormal_rois, rsperiph_rois], axis=1)
    rscombined_rois = tf.map_fn(roi_select, rscombined_rois)
    combined_rois = tf.reshape(rscombined_rois, [grid_n, grid_n, n, 3])
    return combined_rois

# input coordinates x1, y1, x2, y2
def swap_xy(rois):
    x1, y1, x2, y2 = tf.unstack(rois, axis=-1)
    return tf.stack([y1, x1, y2, x2], axis=-1)

def grid_cell_to_tile_coords(rois, grid_n, tile_size):
    # converts between coordinates used internally by the model
    # and coordinates expected by Tensorflow's draw_bounding_boxes function
    #
    # input coords:
    # shape [batch, grid_n, grid_n, n, 3]
    # coordinates in last dimension are x, y, w
    # x and y are in [-1, 1] relative to grid cell center and size of grid cell
    # w is in [0, 1] relatively to tile size. w is a "diameter", not "radius"
    #
    # output coords:
    # shape [batch, grid_n, grid_n, n, 4]
    # coordinates in last dimension are y1, x1, y2, x2
    # relatively to tile_size

    # grid for (0,0) based tile of size tile_size
    cell_w = tile_size/grid_n
    grid = gen_grid(grid_n) * cell_w
    # grid cell centers
    grid_centers = (grid + grid + cell_w) / 2.0  # shape [grid_n, grid_n, 2]
    # roi coordinates
    roi_cx, roi_cy, roi_w = tf.unstack(rois, axis=-1) # shape [batch, grid_n, grid_n, n]
    # grid centers unstacked
    gr_cx, gr_cy = tf.unstack(grid_centers, axis=-1) # shape [grid_n, grid_n]
    gr_cx = tf.expand_dims(tf.expand_dims(gr_cx, 0), 3) # shape [1, grid_n, grid_n, 1]
    gr_cy = tf.expand_dims(tf.expand_dims(gr_cy, 0), 3) # shape [1, grid_n, grid_n, 1]
    roi_cx = roi_cx * cell_w/2 # roi_x=1 means cell center + cell_w/2
    roi_cx = roi_cx+gr_cx
    roi_cy = roi_cy * cell_w/2 # roi_x=1 means cell center + cell_w/2
    roi_cy = roi_cy+gr_cy
    roi_w = roi_w * tile_size
    roi_x1 = roi_cx - roi_w/2
    roi_x2 = roi_cx + roi_w/2
    roi_y1 = roi_cy - roi_w/2
    roi_y2 = roi_cy + roi_w/2
    rois = tf.stack([roi_x1, roi_y1, roi_x2, roi_y2], axis=4)  # shape [batch, grid_n, grid_n, n, 4]
    return rois

# rois shape [n_rois, 4] coordinates (x1, y1, x2, y2) in aerial image coordinates
def find_empty_rois(rois):
    roi_x1, roi_y1, roi_x2, roi_y2 = tf.unstack(rois, axis=-1)
    empty = tf.logical_or(tf.equal(roi_x1, roi_x2), tf.equal(roi_y1, roi_y2))
    return empty


# Selects items from rois acccording to boolean mask. Pads the result to max_n items.
# rois: shape [n, 4]
# mask: shape [n]
# output: shape [max_n, 4]
def filter_by_bool(rois, mask, max_n):
    rois = tf.boolean_mask(rois, mask)
    n = tf.shape(rois)[0]
    # make sure we have enough space in the tensor for all ROIs.
    # If not, pad to max_n and return a boolean to signal the overflow.
    pad_n = tf.maximum(max_n-n, 0)
    rois = tf.pad(rois, [[0, pad_n], [0, 0]])  # pad to max_n elements
    rois = tf.slice(rois, [0,0], [max_n, 4])  # truncate to max_n elements
    return rois


# Selects items from rois acccording to boolean mask. Pads the result to max_n items.
# rois: shape [batch, n, 4]
# mask: shape [batch, n]
# output: shape [batch, max_n, 4]
def batch_filter_by_bool(rois, mask, max_n):
    rois_n = tf.count_nonzero(mask, axis=1)
    overflow = tf.maximum(rois_n - max_n, 0)
    rois = tf.map_fn(lambda rois__mask: filter_by_bool(*rois__mask, max_n=max_n), (rois, mask), dtype=tf.float32)  # shape [batch, max_n, 4]
    rois = tf.reshape(rois, [-1, max_n, 4])  # Tensorflow needs a hint about the shape
    return rois, overflow


# tiles shape [n_tiles, 4] coordinates (x1, y1, x2, y2) in aerial image coordinates
# rois shape [n_rois, 4] coordinates (x1, y1, x2, y2) in aerial image coordinates
# output: shape [n_tiles, max_per_tile, 4] in aerial image coordinates. Roi list padded with empty ROIs.
def remove_non_intersecting_rois(tiles, rois, max_per_tile):
    n_tiles = tf.shape(tiles)[0]
    # compute which rois are contained in which tiles
    rois = tf.expand_dims(rois, axis=0)  # shape [1, n_rois, 4]
    rois = tf.tile(rois, [n_tiles, 1, 1])  # shape [n_tiles, n_rois, 4]
    is_roi_in_tile = tf.map_fn(lambda tiles_rois: boxintersect(*tiles_rois), (tiles, rois), dtype=bool)  # shape [n_tiles, n_rois]
    rois, overflow = batch_filter_by_bool(rois, is_roi_in_tile, max_per_tile)
    return rois, overflow


# rois shape [batch, n_rois, 4] coordinates (x1, y1, x2, y2) in aerial image coordinates
# output: shape [batch, max_per_tile, 4] in aerial image coordinates. Roi list padded with empty ROIs.
def remove_empty_rois(rois, max_per_tile):
    is_non_empty_roi = tf.logical_not(find_empty_rois(rois))
    rois, overflow = batch_filter_by_bool(rois, is_non_empty_roi, max_per_tile)
    return rois, overflow


# tiles shape [n_tiles, 4] coordinates (x1, y1, x2, y2) in aerial image coordinates
# rois shape [n_rois, 4] coordinates (x1, y1, x2, y2) in aerial image coordinates
# max_per_tile: max number of possible rois in one tile
# output: shape [n_tiles, max_per_tile, 4] coordinates (x1, y1, x2, y2) in tile
#         relative coordinates in which tile width = 1.0
# Assumes all tiles have the same size tile_size
def rois_in_tiles_relative(tiles, rois, tile_size, max_per_tile, assert_on_overflow=True):
    rois, overflow = remove_non_intersecting_rois(tiles, rois, max_per_tile)  # [n_tiles, max_per_tile, 4]
    if assert_on_overflow:
        with tf.control_dependencies([tf.assert_non_positive(overflow,
                                     message="ROI per tile overflow. Set MAX_TARGET_ROIS_PER_TILE to a larger value.")]):
            rois = tf.identity(rois)
    is_roi_empty = find_empty_rois(rois)
    is_roi_empty = tf.stack([is_roi_empty, is_roi_empty, is_roi_empty, is_roi_empty], axis=-1)
    tiles = tf.expand_dims(tiles, axis=1)  # force broadcasting on correct axis
    tile_x1, tile_y1, tile_x2, tile_y2 = tf.unstack(tiles, axis=2)  # shape [n_tiles, 1]
    roi_x1, roi_y1, roi_x2, roi_y2 = tf.unstack(rois, axis=2)  # shape [n_tiles, max_per_tile]
    roi_x1 = (roi_x1 - tile_x1) / tile_size  # shapes [n_tiles, max_per_tile] x [n_tiles] broadcast
    roi_x2 = (roi_x2 - tile_x1) / tile_size
    roi_y1 = (roi_y1 - tile_y1) / tile_size
    roi_y2 = (roi_y2 - tile_y1) / tile_size
    rois = tf.stack([roi_x1, roi_y1, roi_x2, roi_y2], axis=-1)  # shape [n_tiles, max_per_tile, 4]
    # replace empty ROIs by (0,0,0,0) for clarity
    return tf.where(is_roi_empty, tf.zeros_like(rois), rois)


class IOUCalculator(object):

    @staticmethod
    def __iou_tile_coordinate(x, tile_size):
        """Replicate a number across a bitmap of size tile_size"""

        xx = tf.cast(tf.round(x), dtype=tf.int16)
        xx = tf.expand_dims(xx, axis=-1)
        xx = tf.tile(xx, [1, 1, tile_size])
        xx = tf.expand_dims(xx, axis=2)
        xx = tf.tile(xx, [1, 1, tile_size, 1])
        return xx

    @staticmethod
    def __iou_gen_linmap(batch, n, tile_size):
        """Creates two bitmaps filled with numbers increasing in X and Y direction.
        This trick makes it easier to draw filled rectangles usinf tf.less and tf.greater."""

        row = tf.cast(tf.linspace(0.0, (tile_size - 1) * 1.0, tile_size), dtype=tf.int16)
        linmap = tf.tile([row], [tile_size, 1])
        linmap = tf.tile([linmap], [n, 1, 1])
        linmap = tf.tile([linmap], [batch, 1, 1, 1])  # shape [batch, n, SIZE, SIZE]
        return linmap

    @classmethod
    def __iou_gen_rectmap(cls, linmap, rects, tile_size):
        """Draws filled rectangles"""

        x1, y1, x2, y2 = tf.unstack(rects, axis=-1)  # shapes [batch, n]
        x1tile = cls.__iou_tile_coordinate(x1, tile_size)
        x2tile = cls.__iou_tile_coordinate(x2, tile_size)
        y1tile = cls.__iou_tile_coordinate(y1, tile_size)
        y2tile = cls.__iou_tile_coordinate(y2, tile_size)
        zeros = tf.zeros_like(linmap, dtype=tf.uint8)
        ones = tf.ones_like(linmap, dtype=tf.uint8)
        mapx = tf.where(tf.greater_equal(linmap, x1tile), ones, zeros)
        mapx = tf.where(tf.less(linmap, x2tile), mapx, zeros)
        mapy = tf.where(tf.greater_equal(linmap, y1tile), ones, zeros)
        mapy = tf.where(tf.less(linmap, y2tile), mapy, zeros)
        mapy = tf.matrix_transpose(mapy)
        map = tf.logical_and(tf.cast(mapx, tf.bool), tf.cast(mapy, tf.bool))
        return map


    @classmethod
    def batch_intersection_over_union(cls, rects1, rects2, tile_size):
        """Computes the intersection over union of two sets of rectangles.
        The actual computation is:
            intersection_area(union(rects1), union(rects2)) / union_area(rects1, rects2)
        This works on batches of rectangles but instantiates a bitmap of size tile_size to compute
        the intersections and is therefore both slow and memory-intensive. Use sparingly.

        Args:
            rects1: detected rectangles, shape [batch, n, 4] with coordinates x1, y1, x2, y2
            rects2: ground truth rectangles, shape [batch, n, 4] with coordinates x1, y1, x2, y2
                The size of the rectangles is [x2-x1, y2-y1].
            tile_size: size of the images where the rectangles apply (also size of internal bitmaps)

        Returns:
            An array of shape [batch]. Use batch_mean() to correctly average it.
            Returns 1 in cases in the batch where both rects1 and rects2 contain
            no rectangles (correctly detected nothing when there was nothing to detect).
        """
        batch = tf.shape(rects1)[0]
        n1 = tf.shape(rects1)[1]  # number of rectangles per batch element in rect1
        n2 = tf.shape(rects2)[1]  # number of rectangles per batch element in rect2
        linmap1 = cls.__iou_gen_linmap(batch, n1, tile_size)
        linmap2 = cls.__iou_gen_linmap(batch, n2, tile_size)
        map1 = cls.__iou_gen_rectmap(linmap1, rects1, tile_size)  # shape [batch, n, tile_size, tile_size]
        map2 = cls.__iou_gen_rectmap(linmap2, rects2, tile_size)  # shape [batch, n, tile_size, tile_size]
        union_all = tf.concat([map1, map2], axis=1)
        union_all = tf.reduce_any(union_all, axis=1)
        union1 = tf.reduce_any(map1, axis=1)  # shape [batch, SIZE, SIZE]
        union2 = tf.reduce_any(map2, axis=1)  # shape [batch, SIZE, SIZE]
        intersect = tf.logical_and(union1, union2)  # shape [batch, SIZE, SIZE]
        union_area = tf.reduce_sum(tf.cast(union_all, tf.float32), axis=[1, 2])  #  can still be empty because of rectangle cropping
        safe_union_area = tf.where(tf.equal(union_area, 0.0), tf.ones_like(union_area), union_area)
        inter_area = tf.reduce_sum(tf.cast(intersect, tf.float32), axis=[1, 2])
        safe_inter_area = tf.where(tf.equal(union_area, 0.0), tf.ones_like(inter_area), inter_area)
        iou = safe_inter_area / safe_union_area  # returns 0 even if the union is null
        return iou


    @staticmethod
    def batch_mean(ious):
        """Computes the average IOU across a batch of IOUs
        IOUs of value 1 mean that the network correctly detected nothing when there was
        nothing to detect. To compute the average IOU, 1 values are eliminated. The result
        is the average IOU across all instances where either something was detected or
        there was something to detect. In the rare case where the result would be 0/0,
        the return value is 1 which is not really correct but should be rare and offset
        a further average of batch_mean() results only a little.

        Args:
            ious: shape[batch]

        Returns:
            mean IOU
        """
        correct_non_detections = tf.equal(ious, 1.0)
        other_detections = tf.logical_not(correct_non_detections)
        n = tf.reduce_sum(tf.cast(other_detections, tf.float32))
        m = tf.reduce_sum(tf.where(correct_non_detections, tf.zeros_like(ious), ious))
        safe_n = tf.where(tf.equal(n, 0.0), tf.ones_like(n), n)
        safe_m = tf.where(tf.equal(n, 0.0), tf.ones_like(m), m)
        return safe_m/safe_n


def compute_safe_IOU(target_rois, detected_rois, detected_rois_overflow, tile_size):
    """Computes the Intersection Over Union (IOU) of a batch of detected boxes
    against a batch of target boxes. Logs a message if a problem occurs."""

    iou_accuracy = IOUCalculator.batch_intersection_over_union(detected_rois * tile_size, target_rois * tile_size, tile_size=tile_size)
    iou_accuracy_overflow = tf.greater(tf.reduce_sum(detected_rois_overflow), 0)
    # check that we are not overflowing the tensor size. Issue a warning if we are. This should only happen at
    # the beginning of the training with a completely uninitialized network.
    iou_accuracy = tf.cond(iou_accuracy_overflow,
                           lambda: tf.Print(iou_accuracy, [detected_rois_overflow],
                                            summarize=250, message="ROI tensor overflow in IOU computation. "
                                                                   "The computed IOU is not correct and will "
                                                                   "be reported as 0. This can be normal in initial "
                                                                   "training iteration when all weights are random. "
                                                                   "Increase MAX_DETECTED_ROIS_PER_TILE to avoid."),
                           lambda: tf.identity(iou_accuracy))
    iou_accuracy = IOUCalculator.batch_mean(iou_accuracy)
    # set iou_accuracy to 0 if there has been any overflow in its computation
    iou_accuracy = tf.where(iou_accuracy_overflow, tf.zeros_like(iou_accuracy), iou_accuracy)
    return iou_accuracy


def compute_mistakes(box_x, box_y, box_w, box_c_sim, target_x, target_y, target_w, target_is_plane, grid_nn):
    DETECTION_TRESHOLD = 0.5  # plane "detected" if predicted C>0.5 TODO: refactor this
    ERROR_TRESHOLD = 0.3  # plane correctly localized if predicted x,y,w within % of ground truth
    detect_correct = tf.logical_not(tf.logical_xor(tf.greater(box_c_sim, DETECTION_TRESHOLD), target_is_plane))
    ones = tf.ones(tf.shape(target_w))
    nonzero_target_w = tf.where(target_is_plane, target_w, ones)
    # true if correct size where there is a plane, nonsense value where there is no plane
    size_correct = tf.less(tf.abs(box_w - target_w) / nonzero_target_w, ERROR_TRESHOLD)
    # true if correct position where there is a plane, nonsense value where there is no plane
    position_correct = tf.less(tf.sqrt(tf.square(box_x - target_x) + tf.square(box_y - target_y)) / nonzero_target_w / grid_nn, ERROR_TRESHOLD)
    truth_no_plane = tf.logical_not(target_is_plane)
    size_correct = tf.logical_or(size_correct, truth_no_plane)
    position_correct = tf.logical_or(position_correct, truth_no_plane)
    size_correct = tf.logical_and(detect_correct, size_correct)
    position_correct = tf.logical_and(detect_correct, position_correct)
    all_correct = tf.logical_and(size_correct, position_correct)
    mistakes = tf.reduce_sum(tf.cast(tf.logical_not(all_correct), tf.int32), axis=[1,2,3])  # shape [batch]
    return mistakes, size_correct, position_correct, all_correct


def rotate(rois, tile_size, rot_matrix):
    # rois: shape [batch, 4] 4 numbers for x1, y1, x2, y2
    translation = tf.constant([tile_size/2.0, tile_size/2.0], tf.float32)
    translation = tf.expand_dims(translation, axis=0)  # to be applied to a batch of points
    batch = tf.shape(rois)[0]
    rois = tf.reshape(rois, [-1, 2])  # batch of points
    # standard trick to apply a rotation matrix to a batch of vectors:
    # do vectors * matrix instead of the usual matrix * vector
    rois = rois - translation
    rois = tf.matmul(rois, rot_matrix)
    rois = rois + translation
    rois = tf.reshape(rois, [batch, 4])
    return rois


def rot90(rois, tile_size, k=1):
    rotation = tf.constant([[0.0, -1.0], [1.0, 0.0]], tf.float32)
    rot_mat = tf.constant([[1.0, 0.0], [0.0, 1.0]], tf.float32)
    k = k % 4  # always a positive number in python
    for _ in range(k):
        rot_mat = tf.matmul(rot_mat, rotation)
    return rotate(rois, tile_size, rot_mat)


def flip_left_right(rois, tile_size):
    transformation = tf.constant([[-1.0, 0.0], [0.0, 1.0]], tf.float32)
    return rotate(rois, tile_size, transformation)


def flip_up_down(rois, tile_size):
    transformation = tf.constant([[1.0, 0.0], [0.0, -1.0]], tf.float32)
    return rotate(rois, tile_size, transformation)


def random_orientation(image_tile, rois, tile_size):
    # This function will output boxes x1, y1, x2, y2 in the standard orientation where x1 <= x2 and y1 <= y2
    rnd = tf.random_uniform([], 0, 8, tf.int32)
    img = image_tile

    def f0(): return tf.image.rot90(img, k=0), rot90(rois, tile_size, k=0)
    def f1(): return tf.image.rot90(img, k=1), rot90(rois, tile_size, k=1)
    def f2(): return tf.image.rot90(img, k=2), rot90(rois, tile_size, k=2)
    def f3(): return tf.image.rot90(img, k=3), rot90(rois, tile_size, k=3)
    def f4(): return tf.image.rot90(tf.image.flip_left_right(img), k=0), rot90(flip_left_right(rois, tile_size), tile_size, k=0)
    def f5(): return tf.image.rot90(tf.image.flip_left_right(img), k=1), rot90(flip_left_right(rois, tile_size), tile_size, k=1)
    def f6(): return tf.image.rot90(tf.image.flip_left_right(img), k=2), rot90(flip_left_right(rois, tile_size), tile_size, k=2)
    def f7(): return tf.image.rot90(tf.image.flip_left_right(img), k=3), rot90(flip_left_right(rois, tile_size), tile_size, k=3)

    image_tile, rois = tf.case({tf.equal(rnd, 0): f0,
                                tf.equal(rnd, 1): f1,
                                tf.equal(rnd, 2): f2,
                                tf.equal(rnd, 3): f3,
                                tf.equal(rnd, 4): f4,
                                tf.equal(rnd, 5): f5,
                                tf.equal(rnd, 6): f6,
                                tf.equal(rnd, 7): f7})

    return image_tile, standardize(rois)


def standardize(rois):
    # rois: shape [batch, 4] 4 numbers for x1, y1, x2, y2
    # put the boxes in the standard orientation where x1 <= x2 and y1 <= y2
    # boxintersect assumes boxes are in the standard format
    x1, y1, x2, y2 = tf.unstack(rois, axis=-1)
    stdx1 = tf.minimum(x1, x2)
    stdy1 = tf.minimum(y1, y2)
    stdx2 = tf.maximum(x1, x2)
    stdy2 = tf.maximum(y1, y2)
    return tf.stack([stdx1, stdy1, stdx2, stdy2], axis=-1)

