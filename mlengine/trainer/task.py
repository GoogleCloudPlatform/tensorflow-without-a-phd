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
from tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.python.platform import tf_logging as logging
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
import argparse
import math
import sys
print("Tensorflow version " + tf.__version__)
logging.set_verbosity(logging.INFO)

#
# To run this: see README.md
#

# Called when the model is deployed for online predictions on Cloud ML Engine.
def serving_input_fn():
    inputs = {'image': tf.placeholder(tf.float32, [None, 28, 28])}
    # Here, you can transform the data received from the API call
    features = [inputs['image']]
    # Compatibility warning: this will move to tf.estimator.export.ServingInputReceiver in TF 1.2
    return tf.contrib.learn.utils.InputFnOps(features, None, inputs)


# In memory training data for this simple case.
# When data is too large to fit in memory, use Tensorflow queues.
def train_data_input_fn(mnist):
    return tf.train.shuffle_batch([tf.constant(mnist.train.images), tf.constant(mnist.train.labels)],
                                  batch_size=100, capacity=1100, min_after_dequeue=1000, enqueue_many=True)


# Eval data is an in-memory constant here.
def eval_data_input_fn(mnist):
    return tf.constant(mnist.test.images), tf.constant(mnist.test.labels)


# Model loss (not needed in INFER mode)
def conv_model_loss(Ylogits, Y_, mode):
    return tf.reduce_mean(tf.losses.softmax_cross_entropy(tf.one_hot(Y_,10), Ylogits)) * 100 \
        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL else None


# Model optimiser (only needed in TRAIN mode)
def conv_model_train_op(loss, mode):
    # Compatibility warning: optimize_loss is still in contrib. This will change in Tensorflow 1.2
    return tf.contrib.layers.optimize_loss(loss, tf.train.get_global_step(), learning_rate=0.003, optimizer="Adam",
        # to remove learning rate decay, comment the next line
        learning_rate_decay_fn=lambda lr, step: 0.0001 + tf.train.exponential_decay(lr, step, -2000, math.e)
        ) if mode == tf.estimator.ModeKeys.TRAIN else None


# Model evaluation metric (not needed in INFER mode)
def conv_model_eval_metrics(classes, Y_, mode):
    # You can name the fields of your metrics dictionary as you like.
    return {'accuracy': tf.metrics.accuracy(classes, Y_)} \
        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL else None


# Model
def conv_model(features, labels, mode):
    X = features
    Y_ = labels
    XX = tf.reshape(X, [-1, 28, 28, 1])
    biasInit = tf.constant_initializer(0.1, dtype=tf.float32)
    Y1 = tf.layers.conv2d(XX,  filters=6,  kernel_size=[6, 6], padding="same", activation=tf.nn.relu, bias_initializer=biasInit)
    Y2 = tf.layers.conv2d(Y1, filters=12, kernel_size=[5, 5], padding="same", strides=2, activation=tf.nn.relu, bias_initializer=biasInit)
    Y3 = tf.layers.conv2d(Y2, filters=24, kernel_size=[4, 4], padding="same", strides=2, activation=tf.nn.relu, bias_initializer=biasInit)
    Y4 = tf.reshape(Y3, [-1, 24*7*7])
    Y5 = tf.layers.dense(Y4, 200, activation=tf.nn.relu, bias_initializer=biasInit)
    # to deactivate dropout on the dense layer, set rate=1. The rate is the % of dropped neurons.
    Y5d = tf.layers.dropout(Y5, rate=0.25, training=mode==tf.estimator.ModeKeys.TRAIN)
    Ylogits = tf.layers.dense(Y5d, 10)
    predict = tf.nn.softmax(Ylogits)
    classes = tf.cast(tf.argmax(predict, 1), tf.uint8)

    loss = conv_model_loss(Ylogits, Y_, mode)
    train_op = conv_model_train_op(loss, mode)
    eval_metrics = conv_model_eval_metrics(classes, Y_, mode)

    # Compatibility warning: this will move to tf.estimator.EstimatorSpec in TF 1.2
    return tf.contrib.learn.ModelFnOps(
        mode=mode,
        predictions={"predictions": predict, "classes": classes}, # name these fields as you like
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metrics
    )

# Compatibility warning: this will move to tf.estimator.run_config.RunConfing in TF 1.2
training_config = tf.contrib.learn.RunConfig(save_checkpoints_secs=None, save_checkpoints_steps=1000)

# This will export a model at every checkpoint, including the transformations needed for online predictions.
# Bug: exports_to_keep=None is mandatory otherwise training crashes.
# Compatibility warning: make_export_strategy is currently in contrib. It will move in TF 1.2
export_strategy=saved_model_export_utils.make_export_strategy(serving_input_fn=serving_input_fn, exports_to_keep=None)


# The Experiment is an Estimator with data loading functions and other parameters
def experiment_fn_with_params(output_dir, data_dir, **kwargs):
    ITERATIONS = 10000
    mnist = mnist_data.read_data_sets(data_dir, reshape=True, one_hot=False, validation_size=0) # loads training and eval data in memory
    # Compatibility warning: Experiment will move
    return tf.contrib.learn.Experiment(
    # Compatibility warning: this will move to tf.estimator.Estimator
    estimator=tf.contrib.learn.Estimator(model_fn=conv_model, model_dir=output_dir, config=training_config),
    train_input_fn=lambda: train_data_input_fn(mnist),
    eval_input_fn=lambda: eval_data_input_fn(mnist),
    train_steps=ITERATIONS,
    eval_steps=1,
    min_eval_frequency=1000,
    export_strategies=export_strategy
)


def main(argv):
    parser = argparse.ArgumentParser()
    # You must accept a --job-dir argument when running on Cloud ML Engine. It specifies where checkpoints
    # should be saved. You can define additional user arguments which will have to be specified after
    # an empty arg -- on the command line:
    # gcloud ml-engine jobs submit training jobXXX --job-dir=... --ml-engine-args -- --user-args
    parser.add_argument('--job-dir', default="checkpoints", help='GCS or local path where to store training checkpoints')
    args = parser.parse_args()
    arguments = args.__dict__
    arguments['data_dir'] = "data" # Hard-coded here: training data will be downloaded to folder 'data'.

    # learn_runner needs an experiment function with a single parameter: the output directory.
    # Here we pass additional command line arguments through a closure.
    output_dir = arguments.pop('job_dir')
    experiment_fn = lambda output_dir: experiment_fn_with_params(output_dir, **arguments)
    # Compatibility warning: learn_runner is currently in contrib. It will move in TF 1.2
    learn_runner.run(experiment_fn, output_dir)


if __name__ == '__main__':
    main(sys.argv)
