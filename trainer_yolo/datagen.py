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

Data generation for YOLO (You Look Only Once) detection model.
Utility data generation functions are here. You can also run this
file directly as a script to convert large aerial pics with ROIs
into 256x256 tiles with the ROIs adjusted accordingly. The resulting
tiles will be stored in TFRecord format"""

import os
import sys
import json
import math
import argparse
import tempfile
import numpy as np
import tensorflow as tf
from tensorflow.python.lib.io import file_io as gcsfile
from tensorflow.python.platform import tf_logging as logging

from trainer_yolo import settings
from trainer_yolo import utils_box as box

from collections import namedtuple

YOLOConfig = namedtuple('yolocfg', 'grid_nn cell_n cell_swarm cell_grow')

tf.logging.set_verbosity(tf.logging.INFO)


def random_hue(images):
    """
    A better random hue algorithm that can also change the color
    of white surfaces.
    :param images:
    :return:
    """
    mask = tf.constant([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [1, 1, 0],
                        [1, 0, 1],
                        [0, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1]], tf.float32)
    batch_size = tf.shape(images)[0]
    rnd_mask = tf.random_uniform([batch_size], 0, 18, dtype=tf.int32)
    mask = tf.gather(mask, rnd_mask)
    strength = tf.random_uniform([batch_size, 3], 0.5, 1.0, dtype=tf.float32)
    inverse_mask = (1 - mask) * strength
    # put masks in image format [batch, 1, 1, 3] the 1-dimensions will broadcast
    mask = tf.expand_dims(mask, axis=1)
    mask = tf.expand_dims(mask, axis=2)
    inverse_mask = tf.expand_dims(inverse_mask, axis=1)
    inverse_mask = tf.expand_dims(inverse_mask, axis=2)
    # partially drop color channels
    images = tf.to_float(images)
    images = images * mask + images * inverse_mask
    image = tf.cast(images, tf.uint8)
    return tf.image.random_hue(image, 0.5)  # rotate the color channels too


def almost_sqrt_factors(x):
    """Returns two integers that are close to each other and 
    multiply to a product that is close to x.

    Args:
        x: the integer to factor into a nd b
    returns:
        a, b integers
    """
    y = math.sqrt(x)
    a = math.floor(y)
    b = math.ceil(y)
    return int(a), int(b)


def log_tensor(message, tensor):
    """Log the value of a tensor at graph execution time.

    Warning: this will only work if the tensor is evaluated in your graph.

    Args:
        message: Prefix message string
        tensor: The tensor to evaluate
    """
    tf.Print(tensor, [tensor], message, summarize=10)


def extract_filename_without_extension(filename):
    basename = os.path.basename(filename)
    barename, extension = os.path.splitext(basename)
    return barename, filename


def load_file_list(directory):
    # load images, load jsons, associate them by name, XYZ.jpg with XYZ.json
    img_files1 = gcsfile.get_matching_files(directory + "/*.jpg")
    img_files2 = gcsfile.get_matching_files(directory + "/*.jpeg")
    img_files = img_files1 + img_files2
    roi_files = gcsfile.get_matching_files(directory + "/*.json")
    img_kv = list(map(extract_filename_without_extension, img_files))
    roi_kv = list(map(extract_filename_without_extension, roi_files))
    all_kv = img_kv + roi_kv
    img_dict = dict(img_kv)
    roi_dict = dict(roi_kv)
    all_dict = dict(all_kv)
    outer_join = [(img_dict[k] if k in img_dict else None,
                   roi_dict[k] if k in roi_dict else None) for k in all_dict]
    # keep only those where the jpg and the json are both available
    inner_join = list(filter(lambda e: e[0] is not None and e[1] is not None, outer_join))
    if len(inner_join) == 0:
        return [], []
    else:
        img_list, roi_list = zip(*inner_join)  # unzip, results are a tuple of img names and a tuple of roi names
        return list(img_list), list(roi_list)


def gcsload(filename):
    # this function is here just to log loaded files
    logging.info("loaded: {}".format(filename))
    return gcsfile.read_file_to_string(filename, binary_mode=True)


def batch_random_orientation(images, rois, tile_size):
    return tf.map_fn(lambda a: box.random_orientation(*a, tile_size=tile_size), (images, rois))


def batch_yolo_roi_attribution(tiles, target_rois, yolo_cfg):
    #  target_rois format: [tiles_n, max_per_tile, 4] 4 for x1, y1, x2, y2 scale 0..1 where 1.0 is the tile size
    tile = tf.constant([0, 0, 1.0, 1.0], tf.float32)
    return tf.map_fn(lambda rois: yolo_roi_attribution(tile, rois, yolo_cfg), target_rois)


def yolo_roi_attribution(tile, rois, yolo_cfg):
    # Tile divided in grid_nn x grid_nn grid
    # Recognizing cell_n boxes per grid cell
    # For each tile, for each grid cell, determine the cell_n largest ROIs centered in that cell
    # Output shape [tiles_n, grid_nn, grid_nn, cell_n, 3] 3 for x, y, w

    # dynamic number of rois
    rois = tf.reshape(rois, [-1, 4])  # I know the shape but Tensorflow does not
    rois_n = tf.shape(rois)[0]  # known shape [n, 4]

    if yolo_cfg.cell_n == 2 and yolo_cfg.cell_swarm:
        yolo_target_rois = box.n_experimental_roi_selection_strategy(tile, rois, rois_n,
                                                                     yolo_cfg.grid_nn,
                                                                     yolo_cfg.cell_n,
                                                                     yolo_cfg.cell_grow)
    elif not yolo_cfg.cell_swarm:
        yolo_target_rois = box.n_largest_rois_in_cell_relative(tile, rois, rois_n,
                                                               yolo_cfg.grid_nn,
                                                               yolo_cfg.cell_n)
    else:
        raise ValueError('Ground truth ROI selection strategy cell_swarm is only implemented for cell_n=2')

    # maybe not needed
    yolo_target_rois = tf.reshape(yolo_target_rois, [yolo_cfg.grid_nn,
                                                     yolo_cfg.grid_nn,
                                                     yolo_cfg.cell_n, 3])  # 3 for x, y, w

    return yolo_target_rois


def generate_slice(pixels, rois, fname, yolo_cfg, rnd_hue, rnd_orientation, repeat_tiles, rnd_distmax, idx):
    # dynamic image shapes
    img_shape = tf.cast(tf.shape(pixels), tf.float32)  # known shape [height, width, 3]
    img_shape = tf.reshape(img_shape, [3])  # tensorflow needs help here
    img_h, img_w, _ = tf.unstack(img_shape)

    # dynamic number of rois
    rois = tf.reshape(rois, [-1, 4])  # I know the shape but Tensorflow does not
    rois_n = tf.shape(rois)[0] # known shape [n, 4]

    TILE_INTERSECT_FRACTION = 0.75

    # random displacements around each ROI (typically 1.4 to 3.0. Fixed at 2.0 for all evals)
    # adjusted so that tiles with planes and no planes happen with roughly equal frequency
    RANDOM_MAX_DISTANCE = rnd_distmax*settings.TILE_SIZE
    N_RANDOM_POSITIONS = repeat_tiles  # repeat_tiles * max nb of planes in one input image = nb of tiles generated in RAM (watch out!)
    # you can increase sdtev to reach more zones without airplanes
    rnd_x = tf.round(tf.truncated_normal([N_RANDOM_POSITIONS], mean=0.0, stddev=RANDOM_MAX_DISTANCE/2.0))
    rnd_y = tf.round(tf.truncated_normal([N_RANDOM_POSITIONS], mean=0.0, stddev=RANDOM_MAX_DISTANCE/2.0))

    def many_tiles_around_this_one(roi):
        roi_x1, roi_y1, roi_x2, roi_y2 = tf.unstack(roi, axis=0)
        # center coordinates of the roi
        roi_x = (roi_x1 + roi_x2) / 2.0
        roi_y = (roi_y1 + roi_y2) / 2.0
        # create N_RANDOM_POSITIONS rois centered on the original
        # but with a random translation and of size [TILE_SIZE, TILE_SIZE]
        roi_x = tf.add(roi_x, rnd_x)  # broadcasting !
        roi_y = tf.add(roi_y, rnd_y)  # broadcasting !
        roi_x1 = tf.add(roi_x, -settings.TILE_SIZE/2.0)
        roi_y1 = tf.add(roi_y, -settings.TILE_SIZE/2.0)
        roi_x2 = tf.add(roi_x, settings.TILE_SIZE/2.0)
        roi_y2 = tf.add(roi_y, settings.TILE_SIZE/2.0)
        roisx = tf.stack([roi_x1, roi_y1, roi_x2, roi_y2], axis=1)
        return roisx

    # for each roi, generate N_RANDOM_POSITIONS translated ROIs
    tiles = tf.map_fn(many_tiles_around_this_one, rois, dtype=tf.float32, name="jitter")
    tiles = tf.reshape(tiles, [-1, 4])  # flatten all generated random ROIs
    # dynamic number of tiles
    tiles_n = tf.shape(tiles)[0]  # known shape [n, 4]

    def count_planes(roi):
        inter = box.boxintersect(roi, rois, TILE_INTERSECT_FRACTION)
        return tf.reduce_sum(tf.cast(inter, dtype=tf.int32))

    # plane counting
    plane_counts = tf.map_fn(count_planes, tiles, dtype=tf.int32)
    # count up to 1 max (planes/no planes)
    # or count up to 3 max (0, 1, 2, lots of planes)
    plane_counts = tf.minimum(plane_counts, 1)

    # debug
    #plane_counts3 = tf.count_nonzero(tf.floor_div(plane_counts, 3))
    #plane_counts2 = tf.count_nonzero(tf.floor_div(plane_counts, 2)) - plane_counts3
    plane_counts1 = tf.count_nonzero(tf.floor_div(plane_counts, 1)) #- plane_counts3 - plane_counts2
    plane_counts0 = tf.count_nonzero(tf.add(plane_counts, 1))       - plane_counts1 #- plane_counts2 - plane_counts3
    tf.Print(tiles_n, [tiles_n, plane_counts0, plane_counts1],
             "Generating tiles [total tiles][tiles with no planes][tiles with 1+ planes]: ")

    # Vocabulary:
    # "tile": a 256x256 region under consideration
    # "cell": tiles are evenly divided into 4 x 4 = 16 cells
    # "roi": a plane bounding box (ground truth)

    # Compute ground truth ROIs
    target_rois = box.rois_in_tiles_relative(tiles, rois, settings.TILE_SIZE, settings.MAX_TARGET_ROIS_PER_TILE)  # shape [n_tiles, MAX_TARGET_ROIS_PER_TILE, 4]

    # resize rois to units used by crop_and_resize
    # TODO: refactor unit conversion into utils_box
    tile_x1, tile_y1, tile_x2, tile_y2 = tf.unstack(tiles, axis=1)
    tile_y1 = tile_y1 / img_h
    tile_x1 = tile_x1 / img_w
    tile_y2 = tile_y2 / img_h
    tile_x2 = tile_x2 / img_w
    # crop_and_resize expects coordinates in format [y1, x1, y2, x2]
    tiles = tf.stack([tile_y1, tile_x1, tile_y2, tile_x2], axis=1)

    indices = tf.zeros([tiles_n], dtype=tf.int32) # all the rois refer to image #0 (there is only one)
    # expand_dims needed because crop_and_resize expects a batch of images
    image_tiles = tf.image.crop_and_resize(tf.expand_dims(pixels, 0), tiles, indices, [settings.TILE_SIZE, settings.TILE_SIZE])
    # crop_and_resize does not output a defined pixel depth but the convolutional layers need it
    image_tiles = tf.reshape(image_tiles, [-1, settings.TILE_SIZE, settings.TILE_SIZE, 3])  # 3 for r, g, b
    image_tiles = tf.cast(image_tiles, tf.uint8)

    if rnd_orientation:
        image_tiles, target_rois = batch_random_orientation(image_tiles, target_rois, 1.0)

    # Compute ground truth ROIs assigned to YOLO grid cells
    yolo_target_rois = batch_yolo_roi_attribution(tiles, target_rois, yolo_cfg)

    if rnd_hue:  # random hue shift for all training images
        image_tiles = random_hue(image_tiles)

    # filename containing the airport name for logging and debugging
    filenames = tf.tile([fname], [tiles_n])

    features, labels = features_and_labels(image_tiles, yolo_target_rois, target_rois, plane_counts, filenames)
    return tf.data.Dataset.from_tensor_slices((features, labels))


def decode_json_py(str):
    obj = json.loads(str.decode('utf-8'))
    rois = np.array([(roi['x'], roi['y'], roi['x']+roi['w'], roi['y']+roi['w']) for roi in obj["markers"]], dtype=np.float32)
    return rois


def decode_image(img_bytes):
    pixels = tf.image.decode_image(img_bytes, channels=3)
    return tf.cast(pixels, tf.uint8)


def decode_image_and_json_bytes(img_bytes, json_bytes):
    # decode jpeg
    pixels = decode_image(img_bytes)
    # parse json
    rois = tf.py_func(decode_json_py, [json_bytes], [tf.float32])
    rois = tf.reshape(rois[0], [-1, 4])
    return pixels, rois


def load_img_and_json_files(img_filename, roi_filename):
    log_tensor("Loading ", img_filename)
    img_bytes = tf.read_file(img_filename)
    json_bytes = tf.read_file(roi_filename)
    pixels, rois = decode_image_and_json_bytes(img_bytes, json_bytes)
    return pixels, rois, img_filename


def generate(pixels, rois, fname, repeat_slice, repeat_tiles, yolo_cfg, rnd_hue, rnd_orientation, rnd_distmax):
    # generate_slice generates random image tiles in memory from a large aerial shot
    # we call it multiple tiles to get more random tiles from the same image, without exceeding available memory.
    return tf.data.Dataset.range(repeat_slice).flat_map(lambda i: generate_slice(pixels, rois, fname, yolo_cfg,
                                                                                 rnd_hue, rnd_orientation,
                                                                                 repeat_tiles, rnd_distmax, i))


#TODO: rename shuffle_buf to shuffle_buf_size for clarity
def init_dataset_from_tfrecords(tfrec_filelist, batch_size, shuffle_buf, yolo_cfg, rnd_hue, rnd_orientation):
    fileset = np.array(tfrec_filelist)
    np.random.shuffle(fileset)  # shuffle filenames
    dataset = tf.data.TFRecordDataset(fileset, buffer_size=10*1024*1024, num_parallel_reads=16)
    if shuffle_buf > 0:
        dataset = dataset.shuffle(shuffle_buf)
    dataset = dataset.apply(tf.contrib.data.map_and_batch(lambda tfrec: read_tfrecord_features(tfrec, yolo_cfg, rnd_hue, rnd_orientation),
                                                          batch_size,
                                                          num_parallel_batches=16))
    return dataset


def init_train_dataset_from_images(img_filelist, roi_filelist, batch_size, shuffle_buf, yolo_cfg, rnd_hue, rnd_orientation, tiles_per_gt_roi, rnd_distmax):
    # Each call to generate_slice produces all the tiles in memory. Calling it once per airport would OOM.
    # To generate 100 tiles around each ROI (for example) we call generate_slice 10 times, generating 10
    # tiles around each ROI every time.
    repeat_slice, repeat_tiles = almost_sqrt_factors(tiles_per_gt_roi)
    fileset = tf.data.Dataset.from_tensor_slices((tf.constant(img_filelist), tf.constant(roi_filelist)))
    fileset = fileset.shuffle(1000)  # shuffle filenames
    dataset = fileset.map(load_img_and_json_files)
    dataset = dataset.flat_map(lambda pix, rois, fname: generate(pix, rois, fname,
                                                                 repeat_slice=repeat_slice,
                                                                 repeat_tiles=repeat_tiles,
                                                                 yolo_cfg=yolo_cfg,
                                                                 rnd_hue=rnd_hue,
                                                                 rnd_orientation=rnd_orientation,
                                                                 rnd_distmax=rnd_distmax))
    if shuffle_buf > 0:
        dataset = dataset.shuffle(shuffle_buf)
    dataset = dataset.batch(batch_size)
    return dataset


def init_eval_dataset_from_images(img_filelist, roi_filelist, eval_batch_size, yolo_cfg):
    fileset = tf.data.Dataset.from_tensor_slices((tf.constant(img_filelist), tf.constant(roi_filelist)))
    dataset = fileset.map(load_img_and_json_files)
    dataset = dataset.flat_map(lambda pix, rois, fname: generate(pix, rois, fname,
                                                                 repeat_slice=1,
                                                                 repeat_tiles=20,  # 1*20 tiles per ground truth ROI
                                                                 yolo_cfg=yolo_cfg,
                                                                 rnd_hue=False,
                                                                 rnd_orientation=False,
                                                                 rnd_distmax=2.0))
    dataset = dataset.batch(eval_batch_size)
    return dataset


def train_dataset_finalize(dataset, cache_after_n_epochs):
    if cache_after_n_epochs > 0:
        dataset = dataset.repeat(cache_after_n_epochs)
        dataset = dataset.cache(tempfile.mkdtemp(prefix="datacache") + "/datacache")
    dataset = dataset.repeat()  # indefinitely
    dataset = dataset.prefetch(1)
    return dataset

def eval_dataset_finalize(dataset):
    # caching does not work for the eval dataset
    #dataset = dataset.cache(tempfile.mkdtemp(prefix="evaldatacache") + "/evaldatacache")
    dataset = dataset.repeat(1)
    dataset = dataset.prefetch(1)
    return dataset


def features_and_labels(image, yolo_target_rois, target_rois, count, fnames):
    features = {'image': image}
    labels = {'yolo_target_rois': yolo_target_rois, 'target_rois': target_rois, 'count': count, 'fnames': fnames}
    return features, labels


def train_dataset_from_images(img_filelist, roi_filelist, batch_size, shuffle_buf, yolo_cfg, rnd_hue, rnd_orientation, tiles_per_gt_roi, rnd_distmax, cache_after_n_epochs=0):
    dataset = init_train_dataset_from_images(img_filelist, roi_filelist, batch_size, shuffle_buf, yolo_cfg, rnd_hue, rnd_orientation, tiles_per_gt_roi, rnd_distmax)
    return train_dataset_finalize(dataset, cache_after_n_epochs)


def train_dataset_from_tfrecords(tfrec_filelist, batch_size, shuffle_buf, yolo_cfg, rnd_hue, rnd_orientation, cache_after_n_epochs=0):
    dataset = init_dataset_from_tfrecords(tfrec_filelist, batch_size, shuffle_buf, yolo_cfg, rnd_hue, rnd_orientation)
    return train_dataset_finalize(dataset, cache_after_n_epochs)


def eval_dataset_from_images(img_filelist, roi_filelist, eval_batch_size, yolo_cfg):
    dataset = init_eval_dataset_from_images(img_filelist, roi_filelist, eval_batch_size, yolo_cfg)
    return eval_dataset_finalize(dataset)


def eval_dataset_from_tfrecords(tfrec_filelist, eval_batch_size, yolo_cfg):
    dataset = init_dataset_from_tfrecords(tfrec_filelist, eval_batch_size, 0, yolo_cfg, False, False)
    # 0 = no shuffling, False = no random hue shift, False = no random orientation
    return eval_dataset_finalize(dataset)


def write_tfrecord_features(tfrec_filewriter, img_bytes, roi_floats, name_bytes):
    # helper function for TFRecords generation
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))  # [value] for inputs of type 'bytes'
    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))  # value for inputs if type 'list'

    tfrec_filewriter.write(tf.train.Example(features=tf.train.Features(feature={
        "img": _bytes_feature(img_bytes),
        "rois": _float_feature(roi_floats),
        "name": _bytes_feature(name_bytes)})).SerializeToString())


def read_tfrecord_features(example, yolo_cfg, rnd_hue, rnd_orientation):
    features = {
        "img": tf.FixedLenFeature((), tf.string),
        "rois": tf.VarLenFeature(tf.float32),
        "name": tf.FixedLenFeature((), tf.string)
    }
    parsed_example = tf.parse_single_example(example, features)
    pixels = decode_image(parsed_example["img"])
    rois = tf.sparse_tensor_to_dense(parsed_example["rois"])
    rois = tf.reshape(rois * settings.TILE_SIZE, [-1, 4])
    airport_name = parsed_example["name"]

    # rois format: x1, y1, x2, y2 in [0..TILE_SIZE]

    if rnd_orientation:
        pixels, rois = box.random_orientation(pixels, rois, settings.TILE_SIZE)

    if rnd_hue:
        pixels = random_hue(tf.expand_dims(pixels, axis=0))

    # TODO: refactor coordinate formats so that target_rois and yolo_target_rois use the same format
    pixels = tf.reshape(pixels, [settings.TILE_SIZE, settings.TILE_SIZE, 3])  # 3 for r, g, b
    # the tile is already cut
    tile = tf.constant([0, 0, settings.TILE_SIZE, settings.TILE_SIZE], tf.float32)
    one_tile = tf.expand_dims(tile, axis=0)
    # Compute ground truth ROIs
    target_rois = box.rois_in_tiles_relative(one_tile, rois, settings.TILE_SIZE, settings.MAX_TARGET_ROIS_PER_TILE)  # shape [n_tiles, MAX_TARGET_ROIS_PER_TILE, 4]
    target_rois = tf.reshape(target_rois, [settings.MAX_TARGET_ROIS_PER_TILE, 4])  # 4 for x1, y1, x2, y2
    # Compute ground truth ROIs assigned to YOLO grid cells
    yolo_target_rois = yolo_roi_attribution(tile, rois, yolo_cfg)
    yolo_target_rois = tf.reshape(yolo_target_rois, [yolo_cfg.grid_nn, yolo_cfg.grid_nn, yolo_cfg.cell_n, 3])  # 3 for x, y, w
    # TODO: remove plane_counts entirely from the model (dummy 0 for the time being)
    return features_and_labels(pixels, yolo_target_rois, target_rois, tf.constant(0), airport_name)


def run_data_generation(data, output_dir, record_batch_size, shuffle_buf, tiles_per_gt_roi, rnd_distmax, rnd_orientation, is_eval):

    img_filelist, roi_filelist = load_file_list(data)

    # sanity checks and log messages
    if len(img_filelist) > 0:
        logging.log(logging.INFO, "Generating {} data.".format("eval" if is_eval else "training"))
    else:
        logging.log(logging.INFO, "No image/json pairs found in folder {}. Skipping.".format(data))
        return

    # dummy args only used in YOLO box assignments, which will be discarded anyway
    # TODO: refactor these outside of the generate_slice function
    yolo_cfg = YOLOConfig(grid_nn = 16, cell_n = 2, cell_swarm = True, cell_grow = 1.0)

    if is_eval:
        dataset = init_eval_dataset_from_images(img_filelist, roi_filelist, record_batch_size, yolo_cfg)
    else:
        dataset = init_train_dataset_from_images(img_filelist, roi_filelist, record_batch_size, shuffle_buf, yolo_cfg,
                                                 False, rnd_orientation, tiles_per_gt_roi, rnd_distmax)  # False = no rnd hue

    dataset = dataset.repeat(1)

    ###
    # TF graph for JPEG image encoding
    features, labels = dataset.make_one_shot_iterator().get_next()
    image_tiles = features['image']
    fname = labels['fnames']
    target_rois = labels['target_rois']  # shape [n_tiles, MAX_TARGET_ROIS_PER_TILE, 4]
    encoded_jpegs = tf.map_fn(lambda image_bytes:
                              tf.image.encode_jpeg(image_bytes, optimize_size=True, chroma_downsampling=False),
                              image_tiles, dtype=tf.string)
    # end of TF graph for image encoding
    ###

    i = 0
    with tf.Session() as sess:
        while True:
            try:
                image_jpegs_r, target_rois_r, fname_r = sess.run([encoded_jpegs, target_rois, fname])
            except tf.errors.OutOfRangeError:
                break
            except tf.errors.NotFoundError:
                break
            i += 1
            # write ROIs
            basename = os.path.basename(fname_r[0].decode("utf-8"))
            basename, _ = os.path.splitext(basename)
            filename = os.path.join(output_dir, "{}tiles{:06}_{}.tfrecord".format(record_batch_size, i, basename))
            with tf.python_io.TFRecordWriter(filename) as file:
                for one_image_jpeg, per_image_target_rois in zip(image_jpegs_r, target_rois_r):
                    nonempty_target_rois = filter(lambda roi: abs(roi[2]-roi[0]) > 0 and  # roi format is x1y1x2y2
                                                              abs(roi[3]-roi[1]) > 0, per_image_target_rois)
                    nonempty_target_rois = np.array(list(nonempty_target_rois), np.float32)
                    nonempty_target_rois = np.reshape(nonempty_target_rois, [-1]).tolist()
                    write_tfrecord_features(file, one_image_jpeg, nonempty_target_rois, fname_r[0])  # write TFRecord


def datagen_main(argv):
    parser = argparse.ArgumentParser()
    def str2bool(v): return v=='True'
    parser.add_argument('--job-dir', default="checkpoints", help='Not used in datagen mode but required by ML engine')
    parser.add_argument('--data', default="sample_data/USGS_public_domain_airports", help='Path to data file (can be on Google cloud storage gs://...)')
    parser.add_argument('--output-dir', default="tilecache", help='Folder where generated training and eval tiles will be stored (can be on Google cloud storage gs://...)')
    parser.add_argument('--record-batch-size', default=100, type=int, help='How many tiles per TFRecord file in the output')
    parser.add_argument('--shuffle-buf', default=10000, type=int, help='Size of the shuffle buffer for shuffling tiles. 0 to disable shuffling.')
    parser.add_argument('--hp-data-tiles-per-gt-roi', default=100, type=int, help='Data generation hyperparameter: number of training tiles generated around each ground truth ROI')
    parser.add_argument('--hp-data-rnd-distmax', default=2.0, type=float, help='Data generation hyperparameter: training tiles selection max random distance from ground truth ROI (always 2.0 for eval tiles)')
    parser.add_argument('--hp-data-rnd-orientation', default=True, type=str2bool, help='Data generation hyperparameter: data augmentation by rotating and flipping tiles.')
    args = parser.parse_args()

    data_eval = args.data + "_eval"
    output_dir_eval = args.output_dir + "_eval"
    if not gcsfile.file_exists(args.output_dir) or not gcsfile.file_exists(output_dir_eval):
        logging.log(logging.ERROR, "Error: both the otput path \"{}\" and the eval "
                                   "output path \"{}\" must exist. Please create them "
                                   "before starting data generation.".format(args.output_dir, output_dir_eval))
        exit(-1)

    logging.log(logging.INFO, "Training data path: " + args.data)
    logging.log(logging.INFO, "Eval data path: " + data_eval)
    logging.log(logging.INFO, "Command-line parameters only affect training data generation. "
                              "Eval data is generated with hard-coded parameters so as to offer "
                              "a consistent evaluation benchmark.")

    rnd_distmax = args.hp_data_rnd_distmax
    tiles_per_gt_roi = args.hp_data_tiles_per_gt_roi
    rnd_orientation = args.hp_data_rnd_orientation

    # training and eval data generation
    run_data_generation(args.data, args.output_dir, args.record_batch_size, args.shuffle_buf, tiles_per_gt_roi, rnd_distmax, rnd_orientation, is_eval=False)
    run_data_generation(data_eval, output_dir_eval, args.record_batch_size, args.shuffle_buf, tiles_per_gt_roi, rnd_distmax, rnd_orientation, is_eval=True)


if __name__ == '__main__':
    datagen_main(sys.argv)
