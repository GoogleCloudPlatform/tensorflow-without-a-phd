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
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.python.lib.io import file_io as gcsfile
from tensorflow.python.platform import tf_logging as logging

from trainer_count import model
from trainer_count.boxutils import boxintersect

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
    rois = np.array([(roi['y'], roi['x'], roi['y']+roi['w'], roi['x']+roi['w']) for roi in obj["markers"]], dtype=np.float32)
    return rois

def gcsload(filename):
    # this function is here just to log loaded files
    logging.info("loaded: {}".format(filename))
    return gcsfile.read_file_to_string(filename, binary_mode=True)

def tf_log_(msg):
    logging.info(msg.decode("utf-8"))

def tf_logten_(msg, ten):
    logging.info(msg.decode("utf-8") + str(ten))

# log a string message from tensorflow code
def tf_log(msg):
    tf.py_func(tf_log_, [tf.constant(msg)], [])

# long a message with a tensor value from tnesorflow code
def tf_logten(msg, ten):
    tf.py_func(tf_logten_, [tf.constant(msg), ten], [])

def generate(pixels, json_bytes, eval=False):
    # parse json
    rois = tf.py_func(decode_json_py, [json_bytes], [tf.float32])

    # fighting with dynamic image shapes ...
    img_shape = tf.shape(pixels)  # known shape [height, width, 3]
    img_shape = tf.reshape(img_shape, [3])
    img_h, img_w, _ = tf.unstack(tf.cast(img_shape, dtype=tf.float32), axis=0)
    # KO!

    # fighting with dynamic number of rois
    rois = tf.reshape(rois, [-1, 4])  # I know the shape but Tensorflow does not
    rois_n = tf.shape(rois)  # known shape [n, 4]
    rois_n = tf.reshape(rois_n, [2])
    rois_n, _ = tf.unstack(rois_n, axis=0)
    #KO!

    TILE_SIZE = 256
    TILE_INTERSECT_FRACTION = 0.75

    # random displacements around each ROI
    N_MAX_OUTPUTS = 3000
    N_RANDOM_POSITIONS = 100
    RANDOM_MAX_DISTANCE = 1.4*TILE_SIZE  # adjusted so that tiles with 0, 1, 2 or many planes happen with roughly equal frequency
    if eval:
        N_RANDOM_POSITIONS = 10

    limit = tf.floordiv(N_MAX_OUTPUTS, rois_n)
    nrand = tf.where(tf.less(N_RANDOM_POSITIONS, limit), N_RANDOM_POSITIONS, limit)
    tf_logten("image slicing limit: ", nrand*rois_n)

    # increase sdtev to reach more zones without airplanes
    rnd_x = tf.truncated_normal([nrand], mean=0.0, stddev=RANDOM_MAX_DISTANCE/2.0)
    rnd_y = tf.truncated_normal([nrand], mean=0.0, stddev=RANDOM_MAX_DISTANCE/2.0)

    def roi_many_around_this_one(roi):
        roi_y1, roi_x1, roi_y2, roi_x2 = tf.unstack(roi, axis=0)
        # re-center the roi
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
        roisx = tf.stack([roi_y1, roi_x1, roi_y2, roi_x2], axis=1)
        return roisx

    # for each roi, generate N_RANDOM_POSITIONS translated ROIs
    rois = tf.reshape(rois, [-1, 4])  # I know the shape but Tensorflow does not
    roisx = tf.map_fn(roi_many_around_this_one, rois, dtype=tf.float32, name="jitter")
    roisx = tf.reshape(roisx, [-1, 4])  # flatten all generated random ROIs

    def count_planes(roi):
        inter = boxintersect(roi, rois, TILE_INTERSECT_FRACTION)
        return tf.reduce_sum(tf.cast(inter, dtype=tf.int32))

    # plane counting
    labels = tf.map_fn(count_planes, roisx, dtype=tf.int32)
    # count up to 1 max (planes/no planes)
    # or count up to 3 max (0, 1, 2, lots of planes)
    labels = tf.minimum(labels, 1)

    # debug
    labels3 = tf.count_nonzero(tf.floor_div(labels, 3))
    labels2 = tf.count_nonzero(tf.floor_div(labels, 2)) -labels3
    labels1 = tf.count_nonzero(tf.floor_div(labels, 1)) -labels3-labels2
    labels0 = tf.count_nonzero(tf.add(labels, 1)) - labels1 -labels2-labels3
    tf_logten("Labels 0: ", labels0)
    tf_logten("Labels 1: ", labels1)
    #tf_logten("Labels 2: ", labels2)
    #tf_logten("Labels 3: ", labels3)

    # fighting with dynamic number of rois
    roisx_n = tf.shape(roisx)  # known shape [n, 4]
    roisx_n = tf.reshape(roisx_n, [2])
    roisx_n, _ = tf.unstack(roisx_n, axis=0)
    #KO!

    # resize rois to units used by crop_and_resize
    rois_y1, rois_x1, rois_y2, rois_x2 = tf.unstack(roisx, axis=1)
    rois_y1 = rois_y1 / img_h
    rois_x1 = rois_x1 / img_w
    rois_y2 = rois_y2 / img_h
    rois_x2 = rois_x2 / img_w
    roisx = tf.stack([rois_y1, rois_x1, rois_y2, rois_x2], axis=1)

    indices = tf.zeros([roisx_n], dtype=tf.int32) # all the rois refer to image #0 (there is only one)
    # expand_dims needed because crop_and_resize expects a batch of images
    images = tf.image.crop_and_resize(tf.expand_dims(pixels, 0), roisx, indices, [TILE_SIZE, TILE_SIZE])
    # crop_and_resize does not output a proper pixel depth but the convolutional layers need it
    images = tf.reshape(images, [-1, TILE_SIZE, TILE_SIZE, 3])
    images = tf.cast(images, tf.uint8)
    return tf.contrib.data.Dataset.from_tensor_slices((images, labels))

def load_files(img_filename, roi_filename):
    tf_logten("Loading ", img_filename)
    img_bytes = tf.read_file(img_filename)
    # This works too bu I do not understand the syntax [toto] =
    #[img_bytes] = tf.py_func(gcsload, [img_filename], [tf.string])
    pixels = tf.image.decode_image(img_bytes, channels=3)
    pixels = tf.cast(pixels, tf.uint8)
    json_bytes = tf.read_file(roi_filename)
    return pixels, json_bytes

def generate_eval(pixels, json_bytes):
    return generate(pixels, json_bytes, eval=True)

def features_and_labels(dataset):
    it = dataset.make_one_shot_iterator()
    images, labels = it.get_next()
    features = {'image': images}
    return features, labels

def dataset_input_fn(img_filelist, roi_filelist):
    dataset = tf.contrib.data.Dataset.from_tensor_slices((tf.constant(img_filelist), tf.constant(roi_filelist)))
    dataset = dataset.map(load_files)
    dataset = dataset.flat_map(generate)
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(50)
    dataset = dataset.repeat()  # indefinitely
    return features_and_labels(dataset)

def dataset_eval_input_fn(img_filelist, roi_filelist):
    dataset = tf.contrib.data.Dataset.from_tensor_slices((tf.constant(img_filelist), tf.constant(roi_filelist)))
    dataset = dataset.map(load_files)
    dataset = dataset.flat_map(generate_eval)
    dataset = dataset.batch(191)
    return features_and_labels(dataset)


# input function for base64 encoded JPEG in JSON, with automatic scanning
# Called when the model is deployed for online predictions on Cloud ML Engine.
def serving_input_fn():
    # TODO: completely wrong...
    # input expects a list of jpeg images

    input_bytes = {'image_bytes': tf.placeholder(tf.string, [None, None]),
                   'square_size': tf.placeholder(tf.int32)}

    input_images = input_bytes['image_bytes'][0]  # accepting only one instance
    input_tilesz = input_bytes['square_size'][0]

    def jpeg_to_bytes(jpeg):
        pixels = tf.image.decode_jpeg(jpeg, channels=3)
        # image format uint8
        # pixels = tf.cast(pixels, tf.float32) / 255.0
        pixels = tf.image.crop_and_resize(tf.expand_dims(pixels,0), boxes, box_ind, [trained_tile_size, trained_tile_size])
        pixels = tf.cast(pixels, dtype=tf.uint8)
        return pixels

    # TODO: this is just a placeholder for one dummy roi
    boxes = tf.constant([input_tilesz, input_tilesz], dtype=tf.float32)

    images = tf.map_fn(jpeg_to_bytes, input_images, dtype=tf.uint8)
    feature_dic = {'image': images, 'roi': boxes}
    return tf.estimator.export.ServingInputReceiver(feature_dic, input_bytes)


def main(argv):
    training_config = tf.contrib.learn.RunConfig(save_checkpoints_secs=None, save_checkpoints_steps=500)
    # Bug, exports_to_keep=None is necessary, otherwise this crashes under Python 3
    export_strategy = tf.contrib.learn.utils.saved_model_export_utils.make_export_strategy(serving_input_fn=serving_input_fn, exports_to_keep=None)

    # The Experiment is an Estimator with data loading functions and other parameters
    def experiment_fn_with_params(output_dir, hparams, data, **kwargs):
        # load data
        img_filelist, roi_filelist = load_file_list(data)
        img_filelist_eval, roi_filelist_eval = load_file_list(data + "_eval")
        ITERATIONS = hparams["iterations"]
        # Compatibility warning: Experiment will move out of conttrib in 1.4
        return tf.contrib.learn.Experiment(
            estimator=tf.estimator.Estimator(model_fn=model.model_fn_squeeze, model_dir=output_dir, config=training_config, params=hparams),
            train_input_fn=lambda: dataset_input_fn(img_filelist, roi_filelist),
            eval_input_fn=lambda: dataset_eval_input_fn(img_filelist_eval, roi_filelist_eval),
            train_steps=ITERATIONS,
            eval_steps=10,
            min_eval_frequency=100,
            export_strategies=export_strategy
        )

    parser = argparse.ArgumentParser()
    # mandatory arguments format for ML Engine:
    # gcloud ml-engine jobs submit training jobXXX --job-dir=... --ml-engine-args -- --user-args

    parser.add_argument('--job-dir', default="checkpoints", help='GCS or local path where to store training checkpoints')
    parser.add_argument('--data', default="planesnet32K.pklz", help='Path to data file (can be on Google cloud storage gs://...)')
    parser.add_argument('--hp-iterations', default=50000, type=int, help='Hyperparameter: number of training iterations')
    parser.add_argument('--hp-lr0', default=0.01, type=float, help='Hyperparameter: initial (max) learning rate')
    parser.add_argument('--hp-lr1', default=0.0001, type=float, help='Hyperparameter: target (min) learning rate')
    parser.add_argument('--hp-lr2', default=3000, type=float, help='Hyperparameter: learning rate decay speed in steps. Learning rate decays by exp(-1) every N steps.')
    parser.add_argument('--hp-dropout', default=0.3, type=float, help='Hyperparameter: dropout rate on dense layers.')
    parser.add_argument('--hp-bnexp', default=0.993, type=float, help='Hyperparameter: exponential decay for batch norm moving averages.')
    parser.add_argument('--hp-dense', default=80, type=int, help='Hyperparameter: size of the dense layer')
    args = parser.parse_args()
    arguments = args.__dict__

    hparams = {k[3:]: v for k, v in arguments.items() if k.startswith('hp_')}
    otherargs = {k: v for k, v in arguments.items() if not k.startswith('hp_')}

    logging.log(logging.INFO, "Hyperparameters:" + str(sorted(hparams.items())))
    logging.log(logging.INFO, "Other parameters:" + str(sorted(otherargs.items())))

    output_dir = otherargs.pop('job_dir')
    experiment_fn = lambda output_dir: experiment_fn_with_params(output_dir, hparams, **otherargs)
    tf.contrib.learn.learn_runner.run(experiment_fn, output_dir)

if __name__ == '__main__':
    main(sys.argv)
