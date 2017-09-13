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
import math
import time
import gzip
#import png
import pickle
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.python.lib.io import file_io as gcsfile
from tensorflow.python.platform import tf_logging as logging
logging.set_verbosity(logging.INFO)
logging.log(logging.INFO, "Tensorflow version " + tf.__version__)

def train_data_input_fn(images, labels):
    features, labels = tf.train.shuffle_batch([tf.constant(images), tf.constant(labels)],
                                              batch_size=100, capacity=5000, min_after_dequeue=2000, enqueue_many=True)
    features = {'image': features}
    return features, labels


# Eval data is an in-memory constant here.
def eval_data_input_fn(images, labels):
    features, labels = tf.constant(images), tf.constant(labels)
    features = {'image': features}
    return features, labels


# Model
def conv_model(features, labels, mode, params):

    def learn_rate(lr, step):
        return 0.0001 + tf.train.exponential_decay(lr, step, 2000, 1/math.e)

    X = features["image"]
    Y_ = labels
    XX = tf.reshape(X, [-1, 20, 20, 3])
    biasInit = tf.constant_initializer(0.1, dtype=tf.float32)
    Y1 = tf.layers.conv2d(XX,  filters=6,  kernel_size=[5, 5], padding="same", activation=tf.nn.relu, bias_initializer=biasInit)
    Y2 = tf.layers.conv2d(Y1, filters=12, kernel_size=[4, 4], padding="same", strides=2, activation=tf.nn.relu, bias_initializer=biasInit)
    Y3 = tf.layers.conv2d(Y2, filters=24, kernel_size=[3, 3], padding="same", strides=2, activation=tf.nn.relu, bias_initializer=biasInit)
    Y4 = tf.reshape(Y3, [-1, 24*5*5])
    Y5 = tf.layers.dense(Y4, 100, activation=tf.nn.relu, bias_initializer=biasInit)
    Y5d = tf.layers.dropout(Y5, rate=0.3, training=(mode == tf.estimator.ModeKeys.TRAIN))
    Ylogits = tf.layers.dense(Y5d, 2)
    predict = tf.nn.softmax(Ylogits)
    classes = tf.cast(tf.argmax(predict, 1), tf.uint8)

    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(tf.one_hot(Y_,2), Ylogits)) * 100
    train_op = tf.contrib.layers.optimize_loss(loss, tf.train.get_global_step(), learning_rate=0.003, optimizer="Adam", learning_rate_decay_fn=learn_rate)
    eval_metrics = {'accuracy': tf.metrics.accuracy(classes, Y_)}

    # Compatibility warning: this will move to tf.estimator.EstimatorSpec in TF 1.2
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={"predictions": predict, "classes": classes}, # name these fields as you like
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metrics,
        export_outputs={'classes': tf.estimator.export.PredictOutput({"predictions": predict, "classes": classes})}
    )

def image_dump(data_image, data_label, data_latlon, data_scnid):
    with open('{}__{}__{}_{}.png'.format(data_label, data_scnid, data_latlon[0], data_latlon[1]), 'wb') as imfile:
        imdata = data_image
        imdata = np.swapaxes(imdata, 0, 1) # [y, x, rgb]
        imdata = np.reshape(imdata, (-1, 20*3)) # [y, [(r,g,b),(r,g,b),(r,g,b),(r,g,b),...]]
        w = png.Writer(20, 20) # expects a list of rows of pixels in (r,g,b) format
        w.write(imfile, imdata)

def load_data(path):

    # loads from GCS if gs:// path,
    # loads locally otherwise

    with gcsfile.FileIO(path, 'rb') as zf:
        with gzip.GzipFile(fileobj=zf, mode='rb') as f:
            planesnet = pickle.load(f)
            # unpack dictionary
            data_images = planesnet['data']
            data_labels = np.array(planesnet['labels'])
            data_latlon = np.array(planesnet['locations'])
            data_scnids = np.array(planesnet['scene_ids'])
            assert len(data_images) == len(data_labels)
            #log message
            logging.log(logging.INFO, "Loaded data file " + path)
            # shuffle the data
            p = np.random.permutation(len(data_images))
            data_images = data_images[p]
            data_labels = data_labels[p]
            data_latlon = data_latlon[p]
            data_scnids = data_scnids[p]
            # images are provided, as a single array of ints, by color planes first
            # and in each color plane, first row first. Reshaping to [batch, 3, 20, 20]
            # will give indexing as [batch, rgb, y, x]. Then swap axes -> [batch, x, y, rgb]
            data_images = np.reshape(data_images, (-1, 3, 20, 20), order="C")
            data_images = np.swapaxes(data_images, 1, 3)

            # image dump for debugging
            #for i in range(64):
            #    image_dump(data_images[i], data_labels[i], data_latlon[i], data_scnids[i])

            # convert images to float
            data_images = (data_images / 255.0).astype(np.float32)

            # partition training and test data
            TEST_SIZE = 5000
            test_images = data_images[:TEST_SIZE]
            test_labels = data_labels[:TEST_SIZE]
            train_images = data_images[TEST_SIZE:]
            train_labels = data_labels[TEST_SIZE:]
            return test_images, test_labels, train_images, train_labels

def main(argv):
    training_config = tf.contrib.learn.RunConfig(save_checkpoints_secs=None, save_checkpoints_steps=1000)

    # The Experiment is an Estimator with data loading functions and other parameters
    def experiment_fn_with_params(output_dir, hparams, data, **kwargs):
        # load data
        test_images, test_labels, train_images, train_labels = load_data(data)
        ITERATIONS = hparams["iterations"]
        # Compatibility warning: Experiment will move out of contrib in 1.4
        return tf.contrib.learn.Experiment(
            estimator=tf.estimator.Estimator(model_fn=conv_model, model_dir=output_dir, config=training_config, params=hparams),
            train_input_fn=lambda: train_data_input_fn(train_images, train_labels),
            eval_input_fn=lambda: eval_data_input_fn(test_images, test_labels),
            train_steps=ITERATIONS,
            eval_steps=1,
            min_eval_frequency=100
        )

    parser = argparse.ArgumentParser()
    # mandatory arguments format for ML Engine:
    # gcloud ml-engine jobs submit training jobXXX --job-dir=... --ml-engine-args -- --user-args

    parser.add_argument('--job-dir', default="checkpoints", help='GCS or local path where to store training checkpoints')
    parser.add_argument('--data', default="planesnet001.pklz", help='Path to data file (can be on Google cloud storage gs://...)')
    parser.add_argument('--hp-iterations', default=10000, type=int, help='Hyperparameter: number of training iterations')
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
