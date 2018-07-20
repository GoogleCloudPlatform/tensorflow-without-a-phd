# Copyright 2018 Google LLC
#
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
import gzip
#import png
import pickle
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.python.lib.io import file_io as gcsfile
from tensorflow.python.platform import tf_logging as logging

from trainer import model
from trainer import boxscan

cnt = 0

logging.set_verbosity(logging.INFO)
logging.log(logging.INFO, "Tensorflow version " + tf.__version__)


def train_data_input_fn(images, labels):
    features, labels = tf.train.shuffle_batch([tf.constant(images), tf.constant(labels)],
                                              batch_size=100, capacity=5000, min_after_dequeue=2000, enqueue_many=True)
    boxes = tf.zeros(shape=[tf.shape(features)[0],4])
    features = {'image': features, 'boxes':boxes}
    return features, labels


# Eval data is an in-memory constant here.
def eval_data_input_fn(images, labels):
    features, labels = tf.constant(images), tf.constant(labels)
    boxes = tf.zeros(shape=[tf.shape(features)[0],4])
    features = {'image': features, 'boxes':boxes}
    return features, labels

def load_dataset(directory):
    files = gcsfile.get_matching_files(directory + "/*")
    labels = list(map(lambda filename: int(os.path.basename(filename)[0:1] == '1'), files))
    boxes = tf.zeros(shape=[len(files),4])
    return tf.contrib.data.Dataset.from_tensor_slices((tf.constant(files), tf.constant(labels), boxes)), len(files)

def gcsload(filename, label, box):
    global cnt
    logging.info("{}:{}".format(cnt,filename))
    cnt +=1
    return gcsfile.read_file_to_string(filename, binary_mode=True), label, box

def load(filename, label, box):
    return tf.read_file(filename), label, box
    #return tf.py_func(gcsload, [filename, label, box], [tf.string, tf.int32, tf.float32])

def decode(img_bytes, label, box):
    img_decoded = tf.image.decode_image(img_bytes, channels=3)
    return img_decoded, label, box

def features_and_labels(dataset):
    it = dataset.make_one_shot_iterator()
    images, labels, boxes = it.get_next()
    features = {'image': images, 'boxes': boxes}
    return features, labels

def dataset_input_fn(dataset):
    dataset = dataset.map(load)
    dataset = dataset.map(decode)
    dataset = dataset.shuffle(20)
    dataset = dataset.batch(1)
    dataset = dataset.repeat()  # indefinitely
    return features_and_labels(dataset)

def dataset_eval_input_fn(dataset, n):
    dataset = dataset.map(load)
    dataset = dataset.map(decode)
    dataset = dataset.batch(n)  # single batch with everything
    return features_and_labels(dataset)


# input function for raw JSON bitmap (uint8)
# Called when the model is deployed for online predictions on Cloud ML Engine.
# def serving_input_fn():
#     inputs = {'image': tf.placeholder(tf.float32, [None, 20, 20, 3])}  # format [batch, x, y, rgb]
#     features = inputs['image'] / 255.0  # from int to to float
#     feature_dic = {'image': features}  # current TF implementation forces features to be a dict (bug?)
#     return tf.estimator.export.ServingInputReceiver(feature_dic, inputs)


# input function for base64 encoded JPEG in JSON
# Called when the model is deployed for online predictions on Cloud ML Engine.
# def serving_input_fn():
#     # input expects a list of jpeg images
#
#     # This works for local predictions
#     # input_bytes = {'image_bytes': tf.placeholder(tf.string, [1, None])}  # format [1, nb_images] why the initial "1"? Mystery!
#
#     input_bytes = {'image_bytes': tf.placeholder(tf.string, [None, None])}  # format [1, nb_images] why the initial "1"? Mystery!
#     input_images = input_bytes['image_bytes'][0]
#
#     def jpeg_to_bytes(jpeg):
#         pixels = tf.image.decode_jpeg(jpeg, channels=3)
#         pixels = tf.cast(pixels, tf.float32) / 255.0
#         return pixels
#
#     images = tf.map_fn(jpeg_to_bytes, input_images, dtype=tf.float32)
#     feature_dic = {'image': images}  # current TF implementation forces features to be a dict (bug?)
#     return tf.estimator.export.ServingInputReceiver(feature_dic, input_bytes)


# input function for base64 encoded JPEG in JSON, with automatic scanning
# Called when the model is deployed for online predictions on Cloud ML Engine.
def serving_input_fn():
    # input expects a list of jpeg images

    input_bytes = {'image_bytes': tf.placeholder(tf.string),
                   'square_size': tf.placeholder(tf.int32)}

    # TODO: get inage instances from instances. Currently getting multiple images in each
    # instance (input_bytes['image_bytes'][0]) because misunderstanding on format.

    input_images = input_bytes['image_bytes']
    input_tilesz = input_bytes['square_size'][0]

    trained_tile_size = 20
    tile_step = 5
    zoom_step = 1.3
    boxes100x100 = np.stack(list(boxscan.genBox(100, 100, trained_tile_size, tile_step, zoom_step)), axis=0)/100.0  # 479 tiles
    boxes200x200 = np.stack(list(boxscan.genBox(200, 200, trained_tile_size, tile_step, zoom_step)), axis=0)/200.0  # 2473 tiles (x5)
    boxes256x256 = np.stack(list(boxscan.genBox(256, 256, trained_tile_size, tile_step, zoom_step)), axis=0)/256.0
    boxes300x300 = np.stack(list(boxscan.genBox(300, 300, trained_tile_size, tile_step, zoom_step)), axis=0)/300.0  # 6052 tiles
    boxes400x400 = np.stack(list(boxscan.genBox(400, 400, trained_tile_size, tile_step, zoom_step)), axis=0)/400.0  # 11369 tiles (x5)
    boxes600x600 = np.stack(list(boxscan.genBox(600, 600, trained_tile_size, tile_step, zoom_step)), axis=0)/600.0  # 26760 tiles
    boxes900x900 = np.stack(list(boxscan.genBox(900, 900, trained_tile_size, tile_step, zoom_step)), axis=0)/900.0  # 62003 tiles (x5)

    def tile100x100(): return tf.constant(boxes100x100, dtype=tf.float32), tf.constant(np.zeros(len(boxes100x100)), dtype=tf.int32)
    def tile200x200(): return tf.constant(boxes200x200, dtype=tf.float32), tf.constant(np.zeros(len(boxes200x200)), dtype=tf.int32)
    def tile256x256(): return tf.constant(boxes256x256, dtype=tf.float32), tf.constant(np.zeros(len(boxes256x256)), dtype=tf.int32)
    def tile300x300(): return tf.constant(boxes300x300, dtype=tf.float32), tf.constant(np.zeros(len(boxes300x300)), dtype=tf.int32)
    def tile400x400(): return tf.constant(boxes400x400, dtype=tf.float32), tf.constant(np.zeros(len(boxes400x400)), dtype=tf.int32)
    def tile600x600(): return tf.constant(boxes600x600, dtype=tf.float32), tf.constant(np.zeros(len(boxes600x600)), dtype=tf.int32)
    def tile900x900(): return tf.constant(boxes900x900, dtype=tf.float32), tf.constant(np.zeros(len(boxes900x900)), dtype=tf.int32)

    boxes, box_ind = tf.case([(tf.equal(input_tilesz, 100), tile100x100),
                              (tf.equal(input_tilesz, 200), tile200x200),
                              (tf.equal(input_tilesz, 256), tile256x256),
                              (tf.equal(input_tilesz, 300), tile300x300),
                              (tf.equal(input_tilesz, 400), tile400x400),
                              (tf.equal(input_tilesz, 600), tile600x600),
                              (tf.equal(input_tilesz, 900), tile900x900)], default=tile100x100, exclusive=True)

    def jpeg_to_bytes(jpeg):
        pixels = tf.image.decode_jpeg(jpeg, channels=3)
        # image format uint8
        # pixels = tf.cast(pixels, tf.float32) / 255.0
        pixels = tf.image.crop_and_resize(tf.expand_dims(pixels,0), boxes, box_ind, [trained_tile_size, trained_tile_size])
        pixels = tf.cast(pixels, dtype=tf.uint8)
        return pixels

    mapped_boxes = tf.tile(boxes, [tf.shape(input_images)[0], 1])

    images = tf.map_fn(jpeg_to_bytes, input_images, dtype=tf.uint8)
    feature_dic = {'image': images, 'boxes': mapped_boxes}
    return tf.estimator.export.ServingInputReceiver(feature_dic, input_bytes)


# def image_dump(data_image, data_label, data_latlon, data_scnid):
#     with open('sample_data/images3/{}__{}__{}_{}.png'.format(data_label, data_scnid, data_latlon[0], data_latlon[1]), 'wb') as imfile:
#         imdata = data_image
#         imdata = np.reshape(imdata, (-1, 20*3)) # [y, [(r,g,b),(r,g,b),(r,g,b),(r,g,b),...]]
#         w = png.Writer(20, 20) # expects a list of rows of pixels in (r,g,b) format
#         w.write(imfile, imdata)

def load_data(path):

    # loads from GCS if gs:// path,
    # loads locally otherwise

    with gcsfile.FileIO(path, 'rb') as zf:
        with gzip.GzipFile(fileobj=zf, mode='rb') as f:
            planesnet = pickle.load(f)
            # unpack dictionary
            data_images = planesnet['data']
            data_labels = np.array(planesnet['labels'])
            #data_latlon = np.array(planesnet['locations'])
            #data_scnids = np.array(planesnet['scene_ids'])
            assert len(data_images) == len(data_labels)
            #log message
            logging.log(logging.INFO, "Loaded data file " + path)

            # images are provided, as a single array of ints, by color planes first
            # and in each color plane, first row first. Reshaping to [batch, 3, 20, 20]
            # will give indexing as [batch, rgb, y, x]. Then swap axes -> [batch, y, x, rgb]
            data_images = np.reshape(data_images, (-1, 3, 20, 20), order="C")
            data_images = np.swapaxes(data_images, 1, 2)
            data_images = np.swapaxes(data_images, 2, 3)

            # image dump for debugging
            #for i in range(24000, 32000):
            #    image_dump(data_images[i], data_labels[i], data_latlon[i], data_scnids[i])

            # shuffle the data
            np.random.seed(0)
            n = len(data_images)
            p = np.random.permutation(n)
            data_images = data_images[p]
            data_labels = data_labels[p]

            # convert images to float
            #data_images = (data_images / 255.0).astype(np.float32)
            # image format uint8

            # partition training and test data
            TEST_SIZE = n // 10
            TEST_SIZE = 5000 if TEST_SIZE<5000 else 10000 if TEST_SIZE > 10000 else TEST_SIZE
            test_images = data_images[:TEST_SIZE]
            test_labels = data_labels[:TEST_SIZE]
            train_images = data_images[TEST_SIZE:]
            train_labels = data_labels[TEST_SIZE:]
            return test_images, test_labels, train_images, train_labels

def main(argv):
    training_config = tf.contrib.learn.RunConfig(save_checkpoints_secs=None, save_checkpoints_steps=500)
    # Bug, exports_to_keep=None is necessary, otherwise this crashes under Python 3
    export_strategy = tf.contrib.learn.utils.saved_model_export_utils.make_export_strategy(serving_input_fn=serving_input_fn, exports_to_keep=None)

    # The Experiment is an Estimator with data loading functions and other parameters
    def experiment_fn_with_params(output_dir, hparams, data, **kwargs):
        # load data
        test_images, test_labels, train_images, train_labels = load_data(data)
        #dataset, nb = load_dataset(data)
        #dataset_eval, nb_eval_files = load_dataset(data + "_eval")
        ITERATIONS = hparams["iterations"]
        # Compatibility warning: Experiment will move out of contrib in 1.4
        return tf.contrib.learn.Experiment(
            estimator=tf.estimator.Estimator(model_fn=model.model_fn, model_dir=output_dir, config=training_config, params=hparams),
            train_input_fn=lambda: train_data_input_fn(train_images, train_labels),
            eval_input_fn=lambda: eval_data_input_fn(test_images, test_labels),
            #train_input_fn=lambda: dataset_input_fn(dataset),
            #eval_input_fn=lambda: dataset_eval_input_fn(dataset_eval, nb_eval_files),
            train_steps=ITERATIONS,
            eval_steps=1,
            min_eval_frequency=100,
            export_strategies=export_strategy
        )

    parser = argparse.ArgumentParser()
    # mandatory arguments format for ML Engine:
    # gcloud ml-engine jobs submit training jobXXX --job-dir=... --ml-engine-args -- --user-args

    parser.add_argument('--job-dir', default="checkpoints", help='GCS or local path where to store training checkpoints')
    parser.add_argument('--data', default="planesnet32K.pklz", help='Path to data file (can be on Google cloud storage gs://...)')
    parser.add_argument('--hp-iterations', default=80000, type=int, help='Hyperparameter: number of training iterations')
    parser.add_argument('--hp-lr0', default=0.01, type=float, help='Hyperparameter: initial (max) learning rate')
    parser.add_argument('--hp-lr1', default=0.0001, type=float, help='Hyperparameter: target (min) learning rate')
    parser.add_argument('--hp-lr2', default=800, type=float, help='Hyperparameter: learning rate decay speed in steps. Learning rate decays by exp(-1) every N steps.')
    parser.add_argument('--hp-dropout', default=0.3, type=float, help='Hyperparameter: dropout rate on dense layers.')
    parser.add_argument('--hp-filter-sizes', default='S' , help='Hyperparameter: convolutional filter sizes S, M, L.')
    parser.add_argument('--hp-conv1', default=16, type=int, help='Hyperparameter: depth of first convolutional layer. Depth then doubles at each layer.')
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
