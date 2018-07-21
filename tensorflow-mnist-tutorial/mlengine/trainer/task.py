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
import os
from trainer.utils import maybe_download_and_ungzip
logging.set_verbosity(logging.INFO)
logging.log(logging.INFO, "Tensorflow version " + tf.__version__)

#
# To run this: see README.md
#

def read_label(tf_bytestring):
    label = tf.decode_raw(tf_bytestring, tf.uint8)
    label = tf.reshape(label, [])
    return tf.to_int32(label)

def read_image(tf_bytestring):
    image = tf.decode_raw(tf_bytestring, tf.uint8)
    image = tf.cast(image, tf.float32)/256.0
    image = tf.reshape(image, [28*28])
    return image

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
    #return {'image': features}, labels  # the dictionary here does not seem to be necessary anymore ...
    return features, labels


def train_data_input_fn(image_file, label_file, params):
    dataset = load_dataset(image_file, label_file)
    dataset = dataset.cache()  ## super-important on TPU
    dataset = dataset.repeat(1)
    dataset = dataset.shuffle(60000)
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(params['batch_size']))
    return nodes_for_model(dataset)


# Eval data
def eval_data_input_fn(image_file, label_file, params):
    dataset = load_dataset(image_file, label_file)
    #dataset = dataset.repeat()
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(params['batch_size']))
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
        #tf.summary.scalar("learn_rate", lr)
        #optimizer = tf.train.AdamOptimizer(lr)
        optimizer = tf.train.GradientDescentOptimizer(lr)
        if params['use_tpu']:
            optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
        #return tf.contrib.training.create_train_op(loss, optimizer)
        return optimizer.minimize(loss, tf.train.get_or_create_global_step())
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
    Y1 = tf.layers.conv2d(XX,  filters=params['conv1'], kernel_size=[6, 6], padding="same", kernel_initializer=weights_init)
    #Y1 = tf.nn.relu(batch_norm_cnv(Y1))
    Y1 = tf.nn.relu(Y1)

    Y2 = tf.layers.conv2d(Y1, filters=params['conv2'], kernel_size=[5, 5], padding="same", strides=2, kernel_initializer=weights_init)
    #Y2 = tf.nn.relu(batch_norm_cnv(Y2))
    Y2 = tf.nn.relu(Y2)

    Y3 = tf.layers.conv2d(Y2, filters=params['conv3'], kernel_size=[4, 4], padding="same", strides=2, kernel_initializer=weights_init)
    #Y3 = tf.nn.relu(batch_norm_cnv(Y3))
    Y3 = tf.nn.relu(Y3)

    Y4 = tf.reshape(Y3, [-1, params['conv3'] * 7*7])
    Y5 = tf.layers.dense(Y4, 200, kernel_initializer=weights_init)
    #Y5 = tf.nn.relu(batch_norm(Y5))
    Y5 = tf.nn.relu(Y5)

    # to deactivate dropout on the dense layer, set rate=1. The rate is the % of dropped neurons.
    Y5d = tf.layers.dropout(Y5, rate=params['dropout'], training=(mode == tf.estimator.ModeKeys.TRAIN))
    Ylogits = tf.layers.dense(Y5d, 10)
    predict = tf.nn.softmax(Ylogits)
    #classes = tf.cast(tf.argmax(predict, 1), tf.uint8)
    classes = tf.argmax(predict, 1)

    loss = conv_model_loss(Ylogits, Y_, mode)
    train_op = conv_model_train_op(loss, mode, params)
    eval_metrics = conv_model_eval_metrics(classes, Y_, mode)

    if params['use_tpu']:
        def metric_fn(predictions, targets): return conv_model_eval_metrics(predictions, targets, mode)

        return tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            predictions={"predictions": predict, "classes": classes},  # name these fields as you like
            loss=loss,
            train_op=train_op,
            eval_metrics=(metric_fn, [classes, Y_]),
            export_outputs={'out': tf.estimator.export.PredictOutput({"predictions": predict, "classes": classes})}
        )
    else:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"predictions": predict, "classes": classes},  # name these fields as you like
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metrics,
            export_outputs={'out': tf.estimator.export.PredictOutput({"predictions": predict, "classes": classes})}
        )

def create_keras_model():
    l = tf.keras.layers
    return tf.keras.Sequential(
        [
            l.Reshape(target_shape=[28, 28, 1], input_shape=(28 * 28,)),
            l.Conv2D(6, 6, padding='same', activation=tf.nn.relu),
            l.Conv2D(12, 5, padding='same', activation=tf.nn.relu),
            l.Conv2D(24, 4, padding='same', activation=tf.nn.relu),
            l.Flatten(),
            l.Dense(200, activation=tf.nn.relu),
            l.Dropout(0.3),
            l.Dense(10)
        ])

# Model
def conv_model2(features, labels, mode, params):
    #X = features['image']
    X = features
    Y_ = labels

    model = create_keras_model()

    Ylogits = model(X, training=(mode == tf.estimator.ModeKeys.TRAIN))

    predict = tf.nn.softmax(Ylogits)
    #classes = tf.cast(tf.argmax(predict, 1), tf.uint8)
    classes = tf.argmax(predict, 1)

    loss = conv_model_loss(Ylogits, Y_, mode)
    train_op = conv_model_train_op(loss, mode, params)
    eval_metrics = conv_model_eval_metrics(classes, Y_, mode)

    if params['use_tpu']:
        def metric_fn(predictions, targets): return conv_model_eval_metrics(predictions, targets, mode)

        return tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            predictions={"predictions": predict, "classes": classes},  # name these fields as you like
            loss=loss,
            train_op=train_op,
            eval_metrics=(metric_fn, [classes, Y_]),
            export_outputs={'out': tf.estimator.export.PredictOutput({"predictions": predict, "classes": classes})}
        )
    else:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"predictions": predict, "classes": classes},  # name these fields as you like
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metrics,
            export_outputs={'out': tf.estimator.export.PredictOutput({"predictions": predict, "classes": classes})}
        )

def main(argv):
    parser = argparse.ArgumentParser()
    # You must accept a --job-dir argument when running on Cloud ML Engine. It specifies where checkpoints
    # should be saved. You can define additional user arguments which will have to be specified after
    # an empty arg -- on the command line:
    # gcloud ml-engine jobs submit training jobXXX --job-dir=... --ml-engine-args -- --user-args

    # no batch norm: lr 0.002-0.0002-2000 is ok, over 10000 iterations (final accuracy 0.9937 loss 2.39 job156)
    # batch norm: lr 0.02-0.0001-600 conv 16-32-64 trains in 3000 iteration (final accuracy 0.9949 loss 1.466 job 159)
    def str2bool(v): return v=='True'
    parser.add_argument('--job-dir', default="checkpoints", help='GCS or local path where to store training checkpoints')
    parser.add_argument('--data-dir', default="data", help='Where training data will be loaded and unzipped')
    parser.add_argument('--lr0', default=0.02, type=float, help='Hyperparameter: initial (max) learning rate')
    parser.add_argument('--lr1', default=0.0001, type=float, help='Hyperparameter: target (min) learning rate')
    parser.add_argument('--lr2', default=600, type=float, help='Hyperparameter: learning rate decay speed in steps. Learning rate decays by exp(-1) every N steps.')
    parser.add_argument('--dropout', default=0.3, type=float, help='Hyperparameter: dropout rate on dense layers.')
    parser.add_argument('--conv1', default=6, type=int, help='Hyperparameter: depth of first convolutional layer.')
    parser.add_argument('--conv2', default=12, type=int, help='Hyperparameter: depth of second convolutional layer.')
    parser.add_argument('--conv3', default=24, type=int, help='Hyperparameter: depth of third convolutional layer.')
    parser.add_argument('--bnexp', default=0.993, type=float, help='Hyperparameter: exponential decay for batch norm moving averages.')
    parser.add_argument('--iterations', default=5000, type=int, help='Hyperparameter: number of training iterations.')
    parser.add_argument('--eval-iterations', default=10, type=int, help='Hyperparameter: number of evaluation iterations.')
    parser.add_argument('--batch', default=1024, type=int, help='Global batch size (1/8th of this is the real batch size on one TPU)')
    parser.add_argument('--use-tpu', default=False, type=str2bool, help='Using a TPU or not')
    parser.add_argument('--tpu-iterations', default=100, type=int, help='Iterations per call to the TPU')
    # TPUEstimator also adds the following parameters internally - do not use them
    parser.add_argument('--tpu', default=None, help='(internal) ML Engine uses this argument to apps the IP address of the TPU')
    parser.add_argument('--tpu-zone', default=None, help='(internal) GCP zone where to provision the TPUs')
    parser.add_argument('--gcp-project', default=None, help='(internal) GCP project where to provision the TPUs')
    #parser.add_argument('--batch-size', default=None, help='(internal) Global batch size on TPUs')
    args = parser.parse_args()

    logging.log(logging.INFO, "Parameters:" + str(args))

    train_images_file, train_labels_file, test_images_file, test_labels_file = load_mnist_data(args.data_dir)
    def train_input_fn(params): return train_data_input_fn(train_images_file, train_labels_file, params)
    def eval_input_fn(params): return eval_data_input_fn(test_images_file, test_labels_file, params)

    # training_config = tf.contrib.tpu.RunConfig(
    #     cluster=tf.contrib.cluster_resolver.TPUClusterResolver(args.tpu, args.tpu_zone,args.gcp_project) \
    #         if args.use_tpu else None,
    #     model_dir=args.job_dir,
    #     session_config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True),
    #     tpu_config=tf.contrib.tpu.TPUConfig(args.tpu_iterations, 8)
    # )

    training_config = tf.estimator.RunConfig(model_dir=args.job_dir, save_summary_steps=100, save_checkpoints_steps=500, keep_checkpoint_max=1)
    # estimator = tf.contrib.tpu.TPUEstimator(model_fn=conv_model2, model_dir=args.job_dir, params=args.__dict__,
    #                                         train_batch_size=args.batch,
    #                                         eval_batch_size=args.batch,
    #                                         config=training_config, use_tpu=args.use_tpu)

    params = args.__dict__
    params["batch_size"] = args.batch
    estimator = tf.estimator.Estimator(model_fn=conv_model2, model_dir=args.job_dir, params=params, config=training_config)

    #train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=args.iterations)
    train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=None)
    export_latest = tf.estimator.LatestExporter("mnist-model",serving_input_receiver_fn=serving_input_fn)
    eval_spec = tf.estimator.EvalSpec(eval_input_fn, steps=10, exporters=export_latest, throttle_secs=2)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    #estimator.train(train_input_fn, max_steps=args.iterations)
    #estimator.evaluate(input_fn=eval_input_fn, steps=args.eval_iterations)
    #estimator.export_savedmodel(os.path.join(args.job_dir, "savedmodel"), serving_input_fn)

if __name__ == '__main__':
    main(sys.argv)

# ML Engine command-line:
# gcloud ml-engine jobs submit training tpu007 --job-dir gs://ml1-demo-martin/jobs/tpu007 --module-name trainer.task --package-path trainer --runtime-version 1.8 --scale-tier BASIC_TPU -- --data-dir gs://ml1-demo-martin/data/mnist-data --use-tpu=True
