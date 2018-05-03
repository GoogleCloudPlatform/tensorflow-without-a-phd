"""Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
_______________________________________________________________________

Datageneration for YOLO (You Look Only Once) detection model."""

import os
import sys
import json
import argparse
import tempfile
import numpy as np
import tensorflow as tf
from tensorflow.python.lib.io import file_io as gcsfile
from tensorflow.python.platform import tf_logging as logging

from trainer_yolo import settings
from trainer_yolo import utils_box as box

tf.logging.set_verbosity(tf.logging.INFO)

def log_tensor(message, tensor):
    """Log the value of a tensor at graph execution time.

    Warning: this will only work if the tensor is evaluated in your graph.

    Args:
        message: Prefix message string
        tensor: The tensor to evaluate
    """
    tf.Print(tensor, [tensor], message)


def extract_filename_without_extension(filename):
    basename = os.path.basename(filename)
    barename, extension = os.path.splitext(basename)
    return (barename, filename)


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


def yolo_roi_attribution(tiles, rois, grid_nn, cell_n, cell_swarm, cell_grow):
    # Tile divided in grid_nn x grid_nn grid
    # Recognizing cell_n boxes per grid cell
    # For each tile, for each grid cell, determine the cell_n largest ROIs centered in that cell
    # Output shape [tiles_n, grid_nn, grid_nn, cell_n, 3] 3 for x, y, w

    # dynamic number of rois
    rois = tf.reshape(rois, [-1, 4])  # I know the shape but Tensorflow does not
    rois_n = tf.shape(rois)[0] # known shape [n, 4]

    if cell_n == 2 and cell_swarm:
        yolo_target_rois = tf.map_fn(lambda tile: box.n_experimental_roi_selection_strategy(tile, rois, rois_n, grid_nn, cell_n, cell_grow), tiles)
    elif not cell_swarm:
        yolo_target_rois = tf.map_fn(lambda tile: box.n_largest_rois_in_cell_relative(tile, rois, rois_n, grid_nn, cell_n), tiles)
    else:
        raise ValueError('Ground truth ROI selection strategy cell_swarm is only implemented for cell_n=2')

    yolo_target_rois = tf.reshape(yolo_target_rois, [-1, grid_nn, grid_nn, cell_n, 3])  # 3 for x, y, w

    return yolo_target_rois


def generate_slice(pixels, rois, fname, grid_nn, cell_n, cell_swarm, cell_grow, rnd_hue, rnd_distmax, idx):
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
    N_RANDOM_POSITIONS = 20  # 20 * max nb of planes in one input image = nb of tiles generated in RAM (watch out!)
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
    # "roi": a plane bounding box (gorund thruth)

    # Compute ground truth ROIs
    target_rois = box.rois_in_tile_relative(tiles, rois, settings.TILE_SIZE, settings.MAX_TARGET_ROIS_PER_TILE)  # shape [n_tiles, MAX_TARGET_ROIS_PER_TILE, 4]

    # Compute ground truth ROIs assigned to YOLO grid cells
    yolo_target_rois = yolo_roi_attribution(tiles, rois, grid_nn, cell_n, cell_swarm, cell_grow)

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

    if rnd_hue:  # random hue shift for all training images
        image_tiles = batch_random_hue(image_tiles)

    # filename containing the airport name for logging and debugging
    filenames = tf.tile([fname], [tiles_n])

    return tf.data.Dataset.from_tensor_slices((image_tiles, plane_counts, filenames, yolo_target_rois, target_rois))


def load_files(img_filename, roi_filename):
    log_tensor("Loading ", img_filename)
    img_bytes = tf.read_file(img_filename)
    pixels = tf.image.decode_image(img_bytes, channels=3)
    pixels = tf.cast(pixels, tf.uint8)
    json_bytes = tf.read_file(roi_filename)
    # parse json
    rois = tf.py_func(decode_json_py, [json_bytes], [tf.float32])
    rois = tf.reshape(rois[0], [-1, 4])
    return pixels, rois, img_filename


def load_tiles(img_filename, roi_filename, grid_nn, cell_n, cell_swarm, cell_grow):
    # TODO: refactor coordinate formats so that target_rois and yolo_target_rois use the same format
    pixels, rois, img_filename = load_files(img_filename, roi_filename)
    pixels = tf.reshape(pixels, [settings.TILE_SIZE, settings.TILE_SIZE, 3])  # 3 for r, g, b
    # the tile is already cut
    one_tile = tf.constant([[0, 0, settings.TILE_SIZE, settings.TILE_SIZE]], tf.float32)
    # Compute ground truth ROIs
    target_rois = box.rois_in_tile_relative(one_tile, rois, settings.TILE_SIZE, settings.MAX_TARGET_ROIS_PER_TILE)  # shape [n_tiles, MAX_TARGET_ROIS_PER_TILE, 4]
    target_rois = tf.reshape(target_rois, [settings.MAX_TARGET_ROIS_PER_TILE, 4])  # 4 for x1, y1, x2, y2
    # Compute ground truth ROIs assigned to YOLO grid cells
    yolo_target_rois = yolo_roi_attribution(one_tile, rois, grid_nn, cell_n, cell_swarm, cell_grow)
    yolo_target_rois = tf.reshape(yolo_target_rois, [grid_nn, grid_nn, cell_n, 3])  # 3 for x, y, w
    a = tf.constant(0, tf.float32)
    b = tf.constant("toto", tf.string)
    # TODO: remove plane_counts entirely from the model
    return pixels, a, b, yolo_target_rois, target_rois


def generate(pixels, rois, fname, repeat_slice, grid_nn, cell_n, cell_swarm, cell_grow, rnd_hue, rnd_distmax):
    # generate_slice generates random image tiles in memory from a large aerial shot
    # we call it multiple tiles to get more random tiles from the same image, without exceeding available memory.
    return tf.data.Dataset.range(repeat_slice).flat_map(lambda i: generate_slice(pixels, rois, fname,
                                                                                 grid_nn, cell_n,
                                                                                 cell_swarm, cell_grow,
                                                                                 rnd_hue, rnd_distmax, i))


def features_and_labels(dataset):
    it = dataset.make_one_shot_iterator()
    image_tiles, counts, filenames, yolo_target_rois, target_rois = it.get_next()
    features = {'image': image_tiles}
    labels = {'yolo_target_rois': yolo_target_rois, 'target_rois': target_rois, 'count': counts, 'fnames': filenames}
    return features, labels


def train_dataset(img_filelist, roi_filelist, from_tiles, grid_nn, cell_n, cell_swarm, cell_grow, rnd_hue, rnd_distmax):
    fileset = tf.data.Dataset.from_tensor_slices((tf.constant(img_filelist), tf.constant(roi_filelist)))
    fileset = fileset.shuffle(500000)  # shuffle filenames
    #fileset.repeat(6)
    # TODO: when loading from tiles, make sure ROIs are in the correct format
    # TODO: when loading from tiles, make sure YOLO-assigned ROIs are computed
    if from_tiles:
        dataset = fileset.map(lambda tilef, jsonf: load_tiles(tilef, jsonf, grid_nn, cell_n, cell_swarm, cell_grow))
    else:
        dataset = fileset.map(load_files)
        dataset = dataset.flat_map(lambda pix, rois, fname: generate(pix, rois, fname,
                                                                     repeat_slice=5,
                                                                     grid_nn=grid_nn,
                                                                     cell_n=cell_n,
                                                                     cell_swarm=cell_swarm,
                                                                     cell_grow=cell_grow,
                                                                     rnd_hue=rnd_hue,
                                                                     rnd_distmax=rnd_distmax))
    return dataset


def train_data_input_fn(img_filelist, roi_filelist, from_tiles, batch_size, grid_nn, cell_n, cell_swarm, cell_grow, shuffle_buf, rnd_hue, rnd_distmax):
    dataset = train_dataset(img_filelist, roi_filelist, from_tiles, grid_nn, cell_n, cell_swarm, cell_grow, rnd_hue, rnd_distmax)
    dataset = dataset.cache(tempfile.mkdtemp(prefix="datacache") + "/datacache")
    dataset = dataset.repeat()  # indefinitely
    dataset = dataset.shuffle(shuffle_buf)
    dataset = dataset.prefetch(batch_size)
    dataset = dataset.batch(batch_size)
    return features_and_labels(dataset)


def eval_dataset(img_filelist, roi_filelist, from_tiles, grid_nn, cell_n, cell_swarm):
    fileset = tf.data.Dataset.from_tensor_slices((tf.constant(img_filelist), tf.constant(roi_filelist)))
    if from_tiles:
        dataset = fileset.map(lambda tilef, jsonf: load_tiles(tilef, jsonf, grid_nn, cell_n, cell_swarm, cell_grow=1.0))
    else:
        dataset = fileset.map(load_files)
        dataset = dataset.flat_map(lambda pix, rois, fname: generate(pix, rois, fname,
                                                                     repeat_slice=1,
                                                                     grid_nn=grid_nn,
                                                                     cell_n=cell_n,
                                                                     cell_swarm=cell_swarm,
                                                                     cell_grow=1.0,
                                                                     rnd_hue=False,
                                                                     rnd_distmax=2.0))
    return dataset


def eval_data_input_fn(img_filelist, roi_filelist, from_tiles, eval_batch_size, grid_nn, cell_n, cell_swarm):
    dataset = eval_dataset(img_filelist, roi_filelist, from_tiles, grid_nn, cell_n, cell_swarm)
    dataset = dataset.cache(tempfile.mkdtemp(prefix="evaldatacache") + "/evaldatacache")
    dataset = dataset.repeat(1)
    # eval dataset was 3820 tiles (60 batches of 64). A larger batch will OOM.
    # eval dataset is 8380 tiles (131 batches of 64). A larger batch will OOM.
    # eval dataset is 8380 tiles (262 batches of 32). A larger batch will OOM.
    dataset = dataset.batch(eval_batch_size)
    return features_and_labels(dataset)


def run_data_generation(data, output_dir, rnd_hue, rnd_distmax, eval):

    img_filelist, roi_filelist = load_file_list(data)

    # sanity checks and log messages
    if len(img_filelist) > 0:
        logging.log(logging.INFO, "Generating {} data.".format("eval" if eval else "training"))
    else:
        logging.log(logging.INFO, "No image/json pairs found in folder {}. Skipping.".format(data))
        return

    # dummy args only used in YOLO box assignments, which will be discarded anyway
    # TODO: refactor these outside of the generate_slice function
    grid_nn = 16
    cell_n = 2
    cell_swarm = True
    cell_grow = 1.0
    from_tiles = False

    if eval:
        dataset = eval_dataset(img_filelist, roi_filelist, from_tiles, grid_nn, cell_n, cell_swarm)
    else:
        dataset = train_dataset(img_filelist, roi_filelist, from_tiles, grid_nn, cell_n, cell_swarm, cell_grow, rnd_hue, rnd_distmax)

    dataset = dataset.repeat(1)
    image_tiles, counts, fname, _, target_rois = dataset.make_one_shot_iterator().get_next()

    # TF graph for JPEG image encoding
    unencoded_image = tf.placeholder(tf.uint8, [settings.TILE_SIZE, settings.TILE_SIZE, 3])
    encoded_jpeg = tf.image.encode_jpeg(unencoded_image, optimize_size=True, chroma_downsampling=False)

    i = 0
    with tf.Session() as sess:
        while True:
            try:
                i += 1
                image_tile_r, target_rois_r, fname_r = sess.run([image_tiles, target_rois, fname])
                image_jpeg_r = sess.run(encoded_jpeg, feed_dict={unencoded_image: image_tile_r})
            except tf.errors.OutOfRangeError:
                break
            except tf.errors.NotFoundError:
                break
            # write ROIs
            markers = []
            basename = os.path.basename(fname_r.decode("utf-8"))
            basename, _ = os.path.splitext(basename)
            filename = os.path.join(output_dir, "tile{:08}_{}.json".format(i, basename))
            with gcsfile.FileIO(filename, "w") as file:
                for k, roi in enumerate(target_rois_r):
                    box_x1 = roi[0]
                    box_y1 = roi[1]
                    box_x2 = roi[2]
                    box_y2 = roi[3]
                    box_x = box_x1 * settings.TILE_SIZE
                    box_y = box_y2 * settings.TILE_SIZE
                    box_w = (box_x2 - box_x1) * settings.TILE_SIZE
                    if box_w > 0:
                        markers.append({"x": int(round(box_x)), "y": int(round(box_y)), "w": int(round(box_w))})
                json_rois = {"markers": markers}
                file.write(json.dumps(json_rois))
            # write image
            filename = os.path.join(output_dir, "tile{:08}_{}.jpeg".format(i, basename))
            with gcsfile.FileIO(filename, "w") as file:
                file.write(image_jpeg_r)

def datagen_main(argv):
    parser = argparse.ArgumentParser()
    def str2bool(v): return v=='True'
    parser.add_argument('--job-dir', default="checkpoints", help='Not used in datagen mode but required by ML engine')
    parser.add_argument('--data', default="sample_data/USGS_public_domain_airports", help='Path to data file (can be on Google cloud storage gs://...)')
    parser.add_argument('--output-dir', default="tilecache", help='Folder where generated training and eavl tiles will be stored (can be on Google cloud storage gs://...)')
    parser.add_argument('--hp-rnd-hue', default=True, type=str2bool, help='Hyperparameter: data augmentation with random hue on training images')
    parser.add_argument('--hp-rnd-distmax', default=2.0, type=float, help='Hyperparameter: training tiles selection max random distance from ground truth ROI (always 2.0 for eval tiles)')

    args = parser.parse_args()
    arguments = args.__dict__

    data = arguments["data"]
    output_dir = arguments["output_dir"]
    rnd_hue = arguments["hp_rnd_hue"]
    rnd_distmax = arguments["hp_rnd_distmax"]

    data_eval = data + "_eval"
    output_dir_eval = output_dir + "_eval"
    if not gcsfile.file_exists(output_dir) or not gcsfile.file_exists(output_dir_eval):
        logging.log(logging.ERROR, "Error: both the otput path \"{}\" and the eval "
                                   "output path \"{}\" must exist. Please create them "
                                   "before starting data generation.".format(output_dir, output_dir_eval))
        exit(-1)

    logging.log(logging.INFO, "Training data path: " + data)
    logging.log(logging.INFO, "Eval data path: " + data)
    logging.log(logging.INFO, "Command-line parameters only affect training data generation. "
                              "Eval data is generated with hard-coded parameters so as to offer "
                              "a consistent evaluation benchmark.")

    run_data_generation(data, output_dir, rnd_hue, rnd_distmax, eval=False)
    run_data_generation(data_eval, output_dir_eval, rnd_hue, rnd_distmax, eval=True)

if __name__ == '__main__':
    datagen_main(sys.argv)