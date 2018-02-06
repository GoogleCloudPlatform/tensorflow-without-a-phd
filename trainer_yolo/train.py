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

import os
import sys
import json
import tempfile
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.python.lib.io import file_io as gcsfile
from tensorflow.python.platform import tf_logging as logging

from trainer_yolo import model
from trainer_yolo import boxutils
from trainer_yolo.boxutils import tf_logten

logging.set_verbosity(logging.INFO)
logging.log(logging.INFO, "Tensorflow version " + tf.__version__)

def extract_filename_without_extension(filename):
    basename = os.path.basename(filename)
    barename, extension = os.path.splitext(basename)
    return (barename, filename)

def load_file_list(directory):
    # load images, load jsons, associate them by name, XYZ.jpg with XYZ.json
    img_files = gcsfile.get_matching_files(directory + "/*.jpg")
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
    img_list, roi_list = zip(*inner_join)  # unzip, results are a tuple of img names and a tuple of roi names
    return list(img_list), list(roi_list)

def decode_json_py(str):
    obj = json.loads(str.decode('utf-8'))
    rois = np.array([(roi['x'], roi['y'], roi['x']+roi['w'], roi['y']+roi['w']) for roi in obj["markers"]], dtype=np.float32)
    return rois

def gcsload(filename):
    # this function is here just to log loaded files
    logging.info("loaded: {}".format(filename))
    return gcsfile.read_file_to_string(filename, binary_mode=True)

# This should no longer be necessary when a batch version of random_hue ships in Tensorflow 1.5
# Tensorflow 1.4.1 does not yet have this. Tracking: https://github.com/tensorflow/tensorflow/issues/8926
def batch_random_hue(images):
    return tf.map_fn(lambda img: tf.image.random_hue(img, 0.5), images)


def generate_slice(pixels, rois, grid_nn, cell_n, cell_swarm, cell_grow, rnd_hue, rnd_distmax, idx):
    # dynamic image shapes
    img_shape = tf.cast(tf.shape(pixels), tf.float32)  # known shape [height, width, 3]
    img_shape = tf.reshape(img_shape, [3])  # tensorflow needs help here
    img_h, img_w, _ = tf.unstack(img_shape)

    # dynamic number of rois
    rois = tf.reshape(rois, [-1, 4])  # I know the shape but Tensorflow does not
    rois_n = tf.shape(rois)[0] # known shape [n, 4]

    TILE_SIZE = 256
    TILE_INTERSECT_FRACTION = 0.75
    MAX_TARGET_ROIS_PER_TILE = 50  # max number of rois in training or test images

    # random displacements around each ROI (typically 1.4 to 3.0. Fixed at 2.0 for all evals)
    # adjusted so that tiles with planes and no planes happen with roughly equal frequency
    RANDOM_MAX_DISTANCE = rnd_distmax*TILE_SIZE
    N_RANDOM_POSITIONS = 20  # 20 * max nb of planes in one input image = nb of tiles generated in RAM (watch out!)
    # you can increase sdtev to reach more zones without airplanes
    rnd_x = tf.truncated_normal([N_RANDOM_POSITIONS], mean=0.0, stddev=RANDOM_MAX_DISTANCE/2.0)
    rnd_y = tf.truncated_normal([N_RANDOM_POSITIONS], mean=0.0, stddev=RANDOM_MAX_DISTANCE/2.0)

    def many_tiles_around_this_one(roi):
        roi_x1, roi_y1, roi_x2, roi_y2 = tf.unstack(roi, axis=0)
        # center coordinates of the roi
        roi_x = (roi_x1 + roi_x2) / 2.0
        roi_y = (roi_y1 + roi_y2) / 2.0
        # create N_RANDOM_POSITIONS rois centered on the original
        # but with a random translation and of size [TILE_SIZE, TILE_SIZE]
        roi_x = tf.add(roi_x, rnd_x)  # broadcasting !
        roi_y = tf.add(roi_y, rnd_y)  # broadcasting !
        roi_x1 = tf.add(roi_x, -TILE_SIZE/2.0)
        roi_y1 = tf.add(roi_y, -TILE_SIZE/2.0)
        roi_x2 = tf.add(roi_x, TILE_SIZE/2.0)
        roi_y2 = tf.add(roi_y, TILE_SIZE/2.0)
        roisx = tf.stack([roi_x1, roi_y1, roi_x2, roi_y2], axis=1)
        return roisx

    # for each roi, generate N_RANDOM_POSITIONS translated ROIs
    tiles = tf.map_fn(many_tiles_around_this_one, rois, dtype=tf.float32, name="jitter")
    tiles = tf.reshape(tiles, [-1, 4])  # flatten all generated random ROIs
    # dynamic number of tiles
    tiles_n = tf.shape(tiles)[0]  # known shape [n, 4]

    def count_planes(roi):
        inter = boxutils.boxintersect(roi, rois, TILE_INTERSECT_FRACTION)
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
    tf_logten("Generating training tiles: ", tiles_n)
    tf_logten("Tiles with 0 planes : ", plane_counts0)
    tf_logten("Tiles with 1+ planes  : ", plane_counts1)
    #tf_logten("Labels 2: ", plane_counts2)
    #tf_logten("Labels 3: ", plane_counts3)

    # Vocabulary:
    # "tile": a 256x256 region under consideration
    # "cell": tiles are evenly divided into 4 x 4 = 16 cells
    # "roi": a plane bounding box (gorund thruth)

    # Tile divided in grid_nn x grid_nn grid
    # Recognizing cell_n boxes per grid cell
    # For each tile, for each grid cell, determine the cell_n largest ROIs centered in that cell
    # Output shape [tiles_n, grid_nn, grid_nn, cell_n, 3]
    if cell_n == 2 and cell_swarm:
        yolo_target_rois = tf.map_fn(lambda tile: boxutils.n_experimental_roi_selection_strategy(tile, rois, rois_n, grid_nn, cell_n, cell_grow), tiles)
    elif not cell_swarm:
        yolo_target_rois = tf.map_fn(lambda tile: boxutils.n_largest_rois_in_cell_relative(tile, rois, rois_n, grid_nn, cell_n), tiles)
    else:
        raise ValueError('Ground truth ROI selection strategy cell_swarm is only implemented for cell_n=2')

    # Compute ground truth ROIs (required coordinate format)
    target_rois = boxutils.rois_in_tile_relative(tiles, rois, TILE_SIZE, MAX_TARGET_ROIS_PER_TILE)  # shape [n_tiles, MAX_TARGET_ROIS_PER_TILE, 4]

    # resize rois to units used by crop_and_resize
    tile_x1, tile_y1, tile_x2, tile_y2 = tf.unstack(tiles, axis=1)
    tile_y1 = tile_y1 / img_h
    tile_x1 = tile_x1 / img_w
    tile_y2 = tile_y2 / img_h
    tile_x2 = tile_x2 / img_w
    # crop_and_resize expects coordinates in format [y1, x1, y2, x2]
    tiles = tf.stack([tile_y1, tile_x1, tile_y2, tile_x2], axis=1)

    indices = tf.zeros([tiles_n], dtype=tf.int32) # all the rois refer to image #0 (there is only one)
    # expand_dims needed because crop_and_resize expects a batch of images
    image_tiles = tf.image.crop_and_resize(tf.expand_dims(pixels, 0), tiles, indices, [TILE_SIZE, TILE_SIZE])
    # crop_and_resize does not output a defined pixel depth but the convolutional layers need it
    image_tiles = tf.reshape(image_tiles, [-1, TILE_SIZE, TILE_SIZE, 3])
    image_tiles = tf.cast(image_tiles, tf.uint8)
    yolo_target_rois = tf.reshape(yolo_target_rois, [-1, grid_nn, grid_nn, cell_n, 3])

    if rnd_hue:  # random hue shift for all training images
        image_tiles = batch_random_hue(image_tiles)

    return tf.data.Dataset.from_tensor_slices((image_tiles, plane_counts, yolo_target_rois, target_rois))


def load_files(img_filename, roi_filename):
    tf_logten("Loading ", img_filename)
    img_bytes = tf.read_file(img_filename)
    pixels = tf.image.decode_image(img_bytes, channels=3)
    pixels = tf.cast(pixels, tf.uint8)
    json_bytes = tf.read_file(roi_filename)
    return pixels, json_bytes


def features_and_labels(dataset):
    it = dataset.make_one_shot_iterator()
    image_tiles, counts, yolo_target_rois, target_rois = it.get_next()
    features = {'image': image_tiles}
    labels = {'count': counts, 'yolo_target_rois': yolo_target_rois, 'target_rois': target_rois}
    return features, labels


def generate(pixels, json_bytes, repeat_slice, grid_nn, cell_n, cell_swarm, cell_grow, rnd_hue, rnd_distmax):
    # parse json
    rois = tf.py_func(decode_json_py, [json_bytes], [tf.float32])
    # generate_slice generates random image tiles in memory from a large aerial shot
    # we call it multiple tiles to get more random tiles from the same image, without exceeding available memory.
    return tf.data.Dataset.range(repeat_slice).flat_map(lambda i: generate_slice(pixels, rois, grid_nn, cell_n, cell_swarm, cell_grow, rnd_hue, rnd_distmax, i))


def dataset_input_fn(img_filelist, roi_filelist, grid_nn, cell_n, cell_swarm, cell_grow, shuffle_buf, rnd_hue, rnd_distmax):
    fileset = tf.data.Dataset.from_tensor_slices((tf.constant(img_filelist), tf.constant(roi_filelist)))
    #fileset.repeat(6)
    dataset = fileset.map(load_files)
    dataset = dataset.flat_map(lambda pix,json: generate(pix, json,
                                                         repeat_slice=5,
                                                         grid_nn=grid_nn,
                                                         cell_n=cell_n,
                                                         cell_swarm=cell_swarm,
                                                         cell_grow=cell_grow,
                                                         rnd_hue=rnd_hue,
                                                         rnd_distmax=rnd_distmax))

    dataset = dataset.cache(tempfile.mkdtemp(prefix="datacache") + "/datacache")
    dataset = dataset.repeat()  # indefinitely
    dataset = dataset.shuffle(shuffle_buf)
    dataset = dataset.prefetch(50)
    dataset = dataset.batch(50)
    return features_and_labels(dataset)


def dataset_eval_input_fn(img_filelist, roi_filelist, grid_nn, cell_n, cell_swarm):
    dataset = tf.data.Dataset.from_tensor_slices((tf.constant(img_filelist), tf.constant(roi_filelist)))
    dataset = dataset.map(load_files)
    dataset = dataset.flat_map(lambda pix,json: generate(pix, json,
                                                         repeat_slice=1,
                                                         grid_nn=grid_nn,
                                                         cell_n=cell_n,
                                                         cell_swarm=cell_swarm,
                                                         cell_grow=1.0,
                                                         rnd_hue=False,
                                                         rnd_distmax=2.0))

    dataset = dataset.cache(tempfile.mkdtemp(prefix="evaldatacache") + "/evaldatacache")
    # eval dataset was 3820 tiles (60 batches of 64). A larger batch will OOM.
    # eval dataset is 8380 tiles (131 batches of 64). A larger batch will OOM.
    dataset = dataset.batch(64)
    return features_and_labels(dataset)


# input function for base64 encoded JPEG in JSON, with automatic scanning
# Called when the model is deployed for online predictions on Cloud ML Engine.
def serving_input_fn():

    # input expects a list of jpeg images

    input_bytes = {'image_bytes': tf.placeholder(tf.string),
                   'square_size': tf.placeholder(tf.int32)}

    input_images = input_bytes['image_bytes']

    def jpeg_to_bytes(jpeg):
        pixels = tf.image.decode_jpeg(jpeg, channels=3)
        pixels = tf.cast(pixels, dtype=tf.uint8)
        return pixels

    images = tf.map_fn(jpeg_to_bytes, input_images, dtype=tf.uint8)
    feature_dic = {'image': images}
    return tf.estimator.export.ServingInputReceiver(feature_dic, input_bytes)

def start_training(output_dir, hparams, data, **kwargs):
    # load data
    img_filelist, roi_filelist = load_file_list(data)
    img_filelist_eval, roi_filelist_eval = load_file_list(data + "_eval")

    export_latest = tf.estimator.LatestExporter(name="planesnet",
                                                serving_input_receiver_fn=serving_input_fn,
                                                exports_to_keep=1)

    train_spec = tf.estimator.TrainSpec(input_fn=lambda:dataset_input_fn(img_filelist, roi_filelist,
                                                                         hparams["grid_nn"],
                                                                         hparams["cell_n"],
                                                                         hparams["cell_swarm"],
                                                                         hparams["cell_grow"],
                                                                         hparams["shuffle_buf"],
                                                                         hparams["rnd_hue"],
                                                                         hparams["rnd_distmax"]),
                                        max_steps=hparams["iterations"])

    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: dataset_eval_input_fn(img_filelist_eval, roi_filelist_eval,
                                                                             hparams["grid_nn"],
                                                                             hparams["cell_n"],
                                                                             hparams["cell_swarm"]),
                                      steps=hparams["evalsteps"], # 477 to exhaust all eval data with eval batch size 8
                                      exporters=export_latest,
                                      start_delay_secs=1,  # ??
                                      throttle_secs=1)  # eval every 10 min in non distributed mode, 5 min in distributed

    training_config = tf.estimator.RunConfig(model_dir=output_dir,
                                             save_summary_steps=100,
                                             save_checkpoints_steps=1000,
                                             keep_checkpoint_max=1)

    estimator=tf.estimator.Estimator(model_fn=model.model_fn_squeeze,
                                     model_dir=output_dir,
                                     config=training_config,
                                     params=hparams)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

def main(argv):
    parser = argparse.ArgumentParser()
    # mandatory arguments format for ML Engine:
    # gcloud ml-engine jobs submit training jobXXX --job-dir=... --ml-engine-args -- --user-args
    def str2bool(v): return v=='True'
    parser.add_argument('--job-dir', default="checkpoints", help='GCS or local path where to store training checkpoints')
    parser.add_argument('--data', default="sample_data/USGS_public_domain_airports", help='Path to data file (can be on Google cloud storage gs://...)')
    parser.add_argument('--hp-iterations', default=50000, type=int, help='Hyperparameter: number of training iterations')
    parser.add_argument('--hp-evalsteps', default=131, type=int, help='Hyperparameter: number of training iterations')
    parser.add_argument('--hp-shuffle-buf', default=50000, type=int, help='Hyperparameter: data shuffle buffer size')
    parser.add_argument('--hp-lr0', default=0.01, type=float, help='Hyperparameter: initial (max) learning rate')
    parser.add_argument('--hp-lr1', default=0.0001, type=float, help='Hyperparameter: target (min) learning rate')
    parser.add_argument('--hp-lr2', default=3000, type=float, help='Hyperparameter: learning rate decay speed in steps. Learning rate decays by exp(-1) every N steps.')
    parser.add_argument('--hp-bnexp', default=0.993, type=float, help='Hyperparameter: exponential decay for batch norm moving averages.')
    parser.add_argument('--hp-lw1', default=1, type=float, help='Hyperparameter: loss weight LW1')
    parser.add_argument('--hp-lw2', default=1, type=float, help='Hyperparameter: loss weight LW2')
    parser.add_argument('--hp-lw3', default=1, type=float, help='Hyperparameter: loss weight LW3')
    parser.add_argument('--hp-grid-nn', default=16, type=int, help='Hyperparameter: size of YOLO grid: grid-nn x grid-nn')
    parser.add_argument('--hp-cell-n', default=2, type=int, help='Hyperparameter: number of ROIs detected per YOLO grid cell')
    parser.add_argument('--hp-cell-swarm', default=True, type=str2bool, help='Hyperparameter: ground truth ROIs selection algorithm. The better swarm algorithm is only implemented for cell_n=2')
    parser.add_argument('--hp-cell-grow', default=1.3, type=float, help='Hyperparameter: ROIs allowed to be cetered beyond grid cell by this factor')
    parser.add_argument('--hp-rnd-hue', default=True, type=str2bool, help='Hyperparameter: data augmentation with random hue on training images')
    parser.add_argument('--hp-rnd-distmax', default=2.0, type=float, help='Hyperparameter: training tiles selection max random distance from ground truth ROI (always 2.0 for eval tiles)')
    args = parser.parse_args()
    arguments = args.__dict__

    hparams = {k[3:]: v for k, v in arguments.items() if k.startswith('hp_')}
    otherargs = {k: v for k, v in arguments.items() if not k.startswith('hp_')}

    logging.log(logging.INFO, "Hyperparameters:" + str(sorted(hparams.items())))
    logging.log(logging.INFO, "Other parameters:" + str(sorted(otherargs.items())))

    output_dir = otherargs.pop('job_dir')
    start_training(output_dir, hparams, **otherargs)

if __name__ == '__main__':
    main(sys.argv)
