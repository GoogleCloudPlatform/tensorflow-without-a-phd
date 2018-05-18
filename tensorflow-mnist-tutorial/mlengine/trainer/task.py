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
from tensorflow.python.platform import tf_logging as logging


import numpy as np
import argparse
import math
import sys
from trainer.utils import maybe_download_and_ungzip
logging.set_verbosity(logging.INFO)
logging.log(logging.INFO, "Tensorflow version " + tf.__version__)

#
# To run this: see README.md
#

def read_label(tf_bytestring):
    label = tf.decode_raw(tf_bytestring, tf.uint8)
    return tf.reshape(label, [])

def read_image(tf_bytestring):
    image = tf.decode_raw(tf_bytestring, tf.uint8)
    return tf.cast(image, tf.float32)/256.0

def load_mnist_data(data_dir):
    SOURCE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
    train_images_file = 'train-images-idx3-ubyte.gz'
    local_train_images_file = maybe_download_and_ungzip(train_images_file, data_dir, SOURCE_URL + train_images_file)
    train_labels_file = 'train-labels-idx1-ubyte.gz'
    local_train_labels_file = maybe_download_and_ungzip(train_labels_file, data_dir, SOURCE_URL + train_labels_file)
    test_images_file = 't10k-images-idx3-ubyte.gz'
    local_test_images_file = maybe_download_and_ungzip(test_images_file, data_dir, SOURCE_URL + test_images_file)
    test_labels_file = 't10k-labels-idx1-ubyte.gz'
    local_test_labels_file = maybe_download_and_ungzip(test_labels_file, data_dir, SOURCE_URL + test_labels_file)
    return local_train_images_file, local_train_labels_file, local_test_images_file, local_test_labels_file

# Called when the model is deployed for online predictions on Cloud ML Engine.
def serving_input_fn():
    inputs = {'image': tf.placeholder(tf.float32, [None, 28, 28])}
    # Here, you can transform the data received from the API call
    features = inputs
    return tf.estimator.export.ServingInputReceiver(features, inputs)


# Load a tf.data.Dataset made of interleaved images and labels
# from an image file and a labels file.
def load_dataset(image_file, label_file):
    imagedataset = tf.data.FixedLengthRecordDataset(image_file, 28*28,
                                                    header_bytes=16, buffer_size=1024*16).map(read_image)
    labelsdataset = tf.data.FixedLengthRecordDataset(label_file, 1,
                                                     header_bytes=8, buffer_size=1024*16).map(read_label)
    dataset = tf.data.Dataset.zip((imagedataset, labelsdataset))
    return dataset


# Returns the iterator nodes that will fedd the model with data
# The outpt format of this function must match the input format of your model_fn
def nodes_for_model(dataset):
    features, labels = dataset.make_one_shot_iterator().get_next()
    return {'image': features}, labels


def train_data_input_fn(image_file, label_file):
    dataset = load_dataset(image_file, label_file)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(60000)
    dataset = dataset.batch(100)
    return nodes_for_model(dataset)


# Eval data
def eval_data_input_fn(image_file, label_file):
    dataset = load_dataset(image_file, label_file)
    dataset = dataset.batch(10000)  # a single batch with all the test data
    dataset = dataset.repeat(1)
    return nodes_for_model(dataset)


# Learning rate with exponential decay and min value
def learn_rate(params, step):
    return params['lr1'] + tf.train.exponential_decay(params['lr0'], step, params['lr2'], 1/math.e)


# Model loss (not needed in INFER mode)
def conv_model_loss(Ylogits, Y_, mode):
    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(tf.one_hot(Y_,10), Ylogits)) * 100
        # Tensorboard training curves
        #tf.summary.scalar("loss", loss)
        return loss
    else:
        return None


# Model optimiser (only needed in TRAIN mode)
def conv_model_train_op(loss, mode, params):
    if mode == tf.estimator.ModeKeys.TRAIN:
        lr = learn_rate(params, tf.train.get_or_create_global_step())
        tf.summary.scalar("learn_rate", lr)
        optimizer = tf.train.AdamOptimizer(lr)
        return tf.contrib.training.create_train_op(loss, optimizer)
    else:
        return None


# Model evaluation metric (not needed in INFER mode)
def conv_model_eval_metrics(classes, Y_, mode):
    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        accuracy = tf.metrics.accuracy(classes, Y_)
        #tf.summary.scalar("accuracy", tf.reduce_mean(accuracy))
        # You can name the fields of your metrics dictionary as you like.
        return {'accuracy': accuracy}
    else:
        return None


# Model
def conv_model(features, labels, mode, params):
    X = features['image']
    Y_ = labels

    #bias_init = tf.constant_initializer(0.1, dtype=tf.float32)
    weights_init = tf.truncated_normal_initializer(stddev=0.1)

    def batch_norm_cnv(inputs):
        return tf.layers.batch_normalization(inputs, axis=3, momentum=params['bnexp'], epsilon=1e-5, scale=False, training=(mode == tf.estimator.ModeKeys.TRAIN))

    def batch_norm(inputs):
        return tf.layers.batch_normalization(inputs, axis=1, momentum=params['bnexp'], epsilon=1e-5, scale=False, training=(mode == tf.estimator.ModeKeys.TRAIN))

    XX = tf.reshape(X, [-1, 28, 28, 1])
    Y1 = tf.layers.conv2d(XX,  filters=params['conv1'],  kernel_size=[6, 6], padding="same", kernel_initializer=weights_init)
    Y1bn = tf.nn.relu(batch_norm_cnv(Y1))

    Y2 = tf.layers.conv2d(Y1bn, filters=params['conv2'], kernel_size=[5, 5], padding="same", strides=2, kernel_initializer=weights_init)
    Y2bn = tf.nn.relu(batch_norm_cnv(Y2))

    Y3 = tf.layers.conv2d(Y2bn, filters=params['conv3'], kernel_size=[4, 4], padding="same", strides=2, kernel_initializer=weights_init)
    Y3bn = tf.nn.relu(batch_norm_cnv(Y3))


    Y4 = tf.reshape(Y3bn, [-1, params['conv3']*7*7])
    Y5 = tf.layers.dense(Y4, 200, kernel_initializer=weights_init)
    Y5bn = tf.nn.relu(batch_norm(Y5))


    # to deactivate dropout on the dense layer, set rate=1. The rate is the % of dropped neurons.
    Y5d = tf.layers.dropout(Y5bn, rate=params['dropout'], training=(mode == tf.estimator.ModeKeys.TRAIN))
    Ylogits = tf.layers.dense(Y5d, 10)
    predict = tf.nn.softmax(Ylogits)
    classes = tf.cast(tf.argmax(predict, 1), tf.uint8)

    loss = conv_model_loss(Ylogits, Y_, mode)
    train_op = conv_model_train_op(loss, mode, params)
    eval_metrics = conv_model_eval_metrics(classes, Y_, mode)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={"predictions": predict, "classes": classes},  # name these fields as you like
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metrics,
        export_outputs={'classes': tf.estimator.export.PredictOutput({"predictions": predict, "classes": classes})}
    )

def main(argv):
    parser = argparse.ArgumentParser()
    # You must accept a --job-dir argument when running on Cloud ML Engine. It specifies where checkpoints
    # should be saved. You can define additional user arguments which will have to be specified after
    # an empty arg -- on the command line:
    # gcloud ml-engine jobs submit training jobXXX --job-dir=... --ml-engine-args -- --user-args

    # no batch norm: lr 0.002-0.0002-2000 is ok, over 10000 iterations (final accuracy 0.9937 loss 2.39 job156)
    # batch norm: lr 0.02-0.0001-600 conv 16-32-64 trains in 3000 iteration (final accuracy 0.9949 loss 1.466 job 159)
    parser.add_argument('--job-dir', default="checkpoints", help='GCS or local path where to store training checkpoints')
    parser.add_argument('--data-dir', default="data", help='Where training data will be loaded and unzipped')
    parser.add_argument('--hp-lr0', default=0.02, type=float, help='Hyperparameter: initial (max) learning rate')
    parser.add_argument('--hp-lr1', default=0.0001, type=float, help='Hyperparameter: target (min) learning rate')
    parser.add_argument('--hp-lr2', default=600, type=float, help='Hyperparameter: learning rate decay speed in steps. Learning rate decays by exp(-1) every N steps.')
    parser.add_argument('--hp-dropout', default=0.3, type=float, help='Hyperparameter: dropout rate on dense layers.')
    parser.add_argument('--hp-conv1', default=6, type=int, help='Hyperparameter: depth of first convolutional layer.')
    parser.add_argument('--hp-conv2', default=12, type=int, help='Hyperparameter: depth of second convolutional layer.')
    parser.add_argument('--hp-conv3', default=24, type=int, help='Hyperparameter: depth of third convolutional layer.')
    parser.add_argument('--hp-bnexp', default=0.993, type=float, help='Hyperparameter: exponential decay for batch norm moving averages.')
    parser.add_argument('--hp-iterations', default=3000, type=int, help='Hyperparameter: number of training iterations.')
    args = parser.parse_args()
    arguments = args.__dict__

    hparams = {k[3:]: v for k, v in arguments.items() if k.startswith('hp_')}
    otherargs = {k: v for k, v in arguments.items() if not k.startswith('hp_')}

    logging.log(logging.INFO, "Hyperparameters:" + str(sorted(hparams.items())))

    data_dir = otherargs['data_dir']
    job_dir = otherargs.pop('job_dir')

    train_images_file, train_labels_file, test_images_file, test_labels_file = load_mnist_data(data_dir)
    def train_input_fn(): return train_data_input_fn(train_images_file, train_labels_file)
    def eval_input_fn(): return eval_data_input_fn(test_images_file, test_labels_file)

    training_config = tf.estimator.RunConfig(model_dir=job_dir, save_summary_steps=10, save_checkpoints_steps=200)
    estimator = tf.estimator.Estimator(model_fn=conv_model, model_dir=job_dir, params=hparams, config=training_config)
    train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=hparams['iterations'])
    export_latest = tf.estimator.LatestExporter("mnist-model",serving_input_receiver_fn=serving_input_fn)
    eval_spec = tf.estimator.EvalSpec(eval_input_fn, steps=1, exporters=export_latest, throttle_secs=60)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

if __name__ == '__main__':
    main(sys.argv)
