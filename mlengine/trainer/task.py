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
from tensorflow.contrib import learn
from tensorflow.contrib import layers
from tensorflow.contrib import metrics
from tensorflow.contrib import framework
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils
from tensorflow.contrib.learn.python.learn.utils import input_fn_utils
from tensorflow.examples.tutorials.mnist import input_data

import argparse
import sys

#
# To run this: see README.md
#

logging.set_verbosity(logging.INFO)

def serving_input_fn():
    inputs_123 = {'image': tf.placeholder(tf.uint8, [None, 28, 28])}
    features = [tf.cast(inputs_123['image'], tf.float32)]
    return input_fn_utils.InputFnOps(features, None, inputs_123)

def train_data_input_fn(mnist):
    EPOCHS=100
    images = tf.train.limit_epochs(tf.constant(mnist.train.images), EPOCHS)
    labels = tf.train.limit_epochs(tf.constant(mnist.train.labels), EPOCHS)
    return tf.train.shuffle_batch([images, labels], 100, 10000, 1000, enqueue_many=True)

def eval_data_input_fn(mnist):
    images_feed, labels_feed = (mnist.test.images, mnist.test.labels)
    return tf.constant(images_feed), tf.constant(labels_feed)


def conv_model_loss(Ylogits, Y_, mode):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=tf.one_hot(Y_,10))) * 100 \
        if mode == learn.ModeKeys.TRAIN or mode == learn.ModeKeys.EVAL else None


def conv_model_train_op(loss, mode):
    return layers.optimize_loss(loss, framework.get_global_step(), 0.001, "Adam") \
        if mode == learn.ModeKeys.TRAIN else None


def conv_model_eval_metrics(classes, Y_, mode):
    return {'accuracy': metrics.accuracy(classes, Y_)} \
        if mode == learn.ModeKeys.TRAIN or mode == learn.ModeKeys.EVAL else None


def conv_model(X, Y_, mode):
    XX = tf.reshape(X, [-1, 28, 28, 1])
    biasInit = tf.constant_initializer(0.1, dtype=tf.float32)
    Y1 = layers.conv2d(XX,  num_outputs=6,  kernel_size=[6, 6], biases_initializer=biasInit)
    Y2 = layers.conv2d(Y1, num_outputs=12, kernel_size=[5, 5], stride=2, biases_initializer=biasInit)
    Y3 = layers.conv2d(Y2, num_outputs=24, kernel_size=[4, 4], stride=2, biases_initializer=biasInit)
    Y4 = layers.flatten(Y3)
    Y5 = layers.relu(Y4, 200, biases_initializer=biasInit)
    Ylogits = layers.linear(Y5, 10)
    predict = tf.nn.softmax(Ylogits)
    classes = tf.cast(tf.argmax(predict, 1), tf.uint8)

    loss = conv_model_loss(Ylogits, Y_, mode)
    train_op = conv_model_train_op(loss, mode)
    eval_metrics = conv_model_eval_metrics(classes, Y_, mode)

    return learn.ModelFnOps(
        mode=mode,
        predictions={"predictions": predict, "classes": classes},
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metrics
    )

training_config = tf.contrib.learn.RunConfig(save_checkpoints_secs=None, save_checkpoints_steps=1000, gpu_memory_fraction=0.9)
export_strategy=saved_model_export_utils.make_export_strategy(export_input_fn=serving_input_fn)

def experiment_fn(output_dir, data, **kwargs):
    ITERATIONS = 10001
    mnist = input_data.read_data_sets(data)
    return learn.Experiment(
    estimator=learn.Estimator(model_fn=conv_model, model_dir=output_dir, config=training_config),
    train_input_fn=lambda: train_data_input_fn(mnist),
    eval_input_fn=lambda: eval_data_input_fn(mnist),
    train_steps=ITERATIONS,
    eval_steps=1,
    export_strategies=export_strategy
)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', default="checkpoints", help='GCS or local path where to store training checkpoints')
    args = parser.parse_args()
    arguments = args.__dict__
    output_dir = arguments.pop('job_dir')
    arguments['data'] = "data"

    # learn_runner needs an experiment function that takes the output directory
    # (where trained checkpoints will be saved) as its first parameter.
    # All other command line parameters will be passed to it as additional arguments.
    experiment = lambda output_dir: experiment_fn(output_dir, **arguments)
    learn_runner.run(experiment, output_dir)


if __name__ == '__main__':
    main(sys.argv)
