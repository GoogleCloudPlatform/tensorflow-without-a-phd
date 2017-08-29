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

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
import argparse
import math
import sys
logging.set_verbosity(logging.INFO)
logging.log(logging.INFO, "Tensorflow version " + tf.__version__)

#
# To run this: see README.md
#

# Called when the model is deployed for online predictions on Cloud ML Engine.
def serving_input_fn():
    inputs = {'image': tf.placeholder(tf.float32, [None, 28, 28])}
    # Here, you can transform the data received from the API call
    features = inputs
    return tf.estimator.export.ServingInputReceiver(features, inputs)


# In memory training data for this simple case.
# When data is too large to fit in memory, use Tensorflow queues.
def train_data_input_fn(mnist):
    features, labels = tf.train.shuffle_batch([tf.constant(mnist.train.images), tf.constant(mnist.train.labels)],
                                            batch_size=100, capacity=5000, min_after_dequeue=2000, enqueue_many=True)
    features = {'image': features}
    return features, labels


# Eval data is an in-memory constant here.
def eval_data_input_fn(mnist):
    features, labels = tf.constant(mnist.test.images), tf.constant(mnist.test.labels)
    features = {'image': features}
    return features, labels


# Model loss (not needed in INFER mode)
def conv_model_loss(Ylogits, Y_, mode):
    return tf.reduce_mean(tf.losses.softmax_cross_entropy(tf.one_hot(Y_,10), Ylogits)) * 100 \
        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL else None


# Model optimiser (only needed in TRAIN mode)
def conv_model_train_op(loss, mode, params):
    # Compatibility warning: optimize_loss is still in contrib. This will change in Tensorflow 1.4
    return tf.contrib.layers.optimize_loss(loss, tf.train.get_global_step(), learning_rate=params['lr0'], optimizer="Adam",
        # to remove learning rate decay, comment the next line
        learning_rate_decay_fn=lambda lr, step: params['lr1'] + tf.train.exponential_decay(lr, step, -params['lr2'], math.e)
        ) if mode == tf.estimator.ModeKeys.TRAIN else None


# Model evaluation metric (not needed in INFER mode)
def conv_model_eval_metrics(classes, Y_, mode):
    # You can name the fields of your metrics dictionary as you like.
    return {'accuracy': tf.metrics.accuracy(classes, Y_)} \
        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL else None


# Model
def conv_model(features, labels, mode, params):
    X = features['image']
    Y_ = labels

    bias_init = tf.constant_initializer(0.1, dtype=tf.float32)
    weights_init = tf.truncated_normal_initializer(stddev=0.1)

    XX = tf.reshape(X, [-1, 28, 28, 1])
    Y1 = tf.layers.conv2d(XX,  filters=params['conv1'],  kernel_size=[6, 6], padding="same", activation=tf.nn.relu, bias_initializer=bias_init, kernel_initializer=weights_init)
    Y2 = tf.layers.conv2d(Y1, filters=params['conv2'], kernel_size=[5, 5], padding="same", strides=2, activation=tf.nn.relu, bias_initializer=bias_init, kernel_initializer=weights_init)
    Y3 = tf.layers.conv2d(Y2, filters=params['conv3'], kernel_size=[4, 4], padding="same", strides=2, activation=tf.nn.relu, bias_initializer=bias_init, kernel_initializer=weights_init)
    Y4 = tf.reshape(Y3, [-1, params['conv3']*7*7])
    Y5 = tf.layers.dense(Y4, 200, activation=tf.nn.relu, bias_initializer=bias_init, kernel_initializer=weights_init)
    # to deactivate dropout on the dense layer, set rate=1. The rate is the % of dropped neurons.
    Y5d = tf.layers.dropout(Y5, rate=params['dropout'], training=mode==tf.estimator.ModeKeys.TRAIN)
    Ylogits = tf.layers.dense(Y5d, 10)
    predict = tf.nn.softmax(Ylogits)
    classes = tf.cast(tf.argmax(predict, 1), tf.uint8)

    loss = conv_model_loss(Ylogits, Y_, mode)
    train_op = conv_model_train_op(loss, mode, params)
    eval_metrics = conv_model_eval_metrics(classes, Y_, mode)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={"predictions": predict, "classes": classes}, # name these fields as you like
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metrics,
        # ???
        export_outputs={'classes': tf.estimator.export.PredictOutput({"predictions": predict, "classes": classes})}
    )

# Compatibility warning: this will move to tf.estimator.run_config.RunConfing in TF 1.4
training_config = tf.contrib.learn.RunConfig(save_checkpoints_secs=None, save_checkpoints_steps=1000)

# This will export a model at every checkpoint, including the transformations needed for online predictions.
# Bug: exports_to_keep=None is mandatory otherwise training crashes.
# Compatibility warning: make_export_strategy is currently in contrib. It will move in TF 1.4
export_strategy = tf.contrib.learn.utils.saved_model_export_utils.make_export_strategy(serving_input_fn=serving_input_fn)


# The Experiment is an Estimator with data loading functions and other parameters
def experiment_fn_with_params(output_dir, hparams, data_dir, **kwargs):
    ITERATIONS = 10000
    mnist = mnist_data.read_data_sets(data_dir, reshape=True, one_hot=False, validation_size=0) # loads training and eval data in memory
    # Compatibility warning: Experiment will move out of contrib in 1.4
    return tf.contrib.learn.Experiment(
    estimator=tf.estimator.Estimator(model_fn=conv_model, model_dir=output_dir, config=training_config, params=hparams),
    train_input_fn=lambda: train_data_input_fn(mnist),
    eval_input_fn=lambda: eval_data_input_fn(mnist),
    train_steps=ITERATIONS,
    eval_steps=1,
    min_eval_frequency=100,
    export_strategies=export_strategy
)


def main(argv):
    parser = argparse.ArgumentParser()
    # You must accept a --job-dir argument when running on Cloud ML Engine. It specifies where checkpoints
    # should be saved. You can define additional user arguments which will have to be specified after
    # an empty arg -- on the command line:
    # gcloud ml-engine jobs submit training jobXXX --job-dir=... --ml-engine-args -- --user-args
    parser.add_argument('--job-dir', default="checkpoints", help='GCS or local path where to store training checkpoints')
    parser.add_argument('--data-dir', default="data", help='Where training data will be loaded and unzipped')
    parser.add_argument('--hp-lr0', default=0.005, type=float, help='Hyperparameter: initial (max) learning rate')
    parser.add_argument('--hp-lr1', default=0.0002, type=float, help='Hyperparameter: target (min) learning rate')
    parser.add_argument('--hp-lr2', default=2000, type=float, help='Hyperparameter: learning rate decay speed in steps. Learning rate decays by exp(-1) every N steps.')
    parser.add_argument('--hp-dropout', default=0.3, type=float, help='Hyperparameter: Dropout rate on dense layers.')
    parser.add_argument('--hp-conv1', default=6, type=int, help='Hyperparameter: Depth of first convolutional layer.')
    parser.add_argument('--hp-conv2', default=12, type=int, help='Hyperparameter: Depth of second convolutional layer.')
    parser.add_argument('--hp-conv3', default=24, type=int, help='Hyperparameter: Depth of third convolutional layer.')
    args = parser.parse_args()
    arguments = args.__dict__

    hparams = {k[3:]: v for k, v in arguments.items() if k.startswith('hp_')}
    otherargs = {k: v for k, v in arguments.items() if not k.startswith('hp_')}

    logging.log(logging.INFO, "Hyperparameters:" + str(sorted(hparams.items())))

    output_dir = otherargs.pop('job_dir')

    # learn_runner needs an experiment function with a single parameter: the output directory.
    # Here we pass additional command line arguments through a closure.
    experiment_fn = lambda output_dir: experiment_fn_with_params(output_dir, hparams, **otherargs)
    # Compatibility warning: learn_runner is currently in contrib. It will move in TF 1.2
    tf.contrib.learn.learn_runner.run(experiment_fn, output_dir)


if __name__ == '__main__':
    main(sys.argv)
