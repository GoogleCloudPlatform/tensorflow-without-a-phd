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
from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec

import tempfile
from tensorflow.examples.tutorials.mnist import input_data

# To run this:
# cd to the directory containing the "trainer" directory and thre "config.yaml" file
# gcloud beta ml jobs submit training job22 --package-path=trainer --module-name=trainer.task --staging-bucket=gs://ml1-demo-martin --config=config.yaml -- --train_dir=gs://ml1-demo-martin/jobs/train22

logging.set_verbosity(logging.INFO)
# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')

def train_data_input_fn(mnist):
    EPOCHS=100
    images = tf.train.limit_epochs(tf.constant(mnist.train.images), EPOCHS)
    labels = tf.train.limit_epochs(tf.constant(mnist.train.labels), EPOCHS)
    return tf.train.shuffle_batch([images, labels], 100, 10000, 1000, enqueue_many=True)

def eval_data_input_fn(mnist):
    images_feed, labels_feed = (mnist.test.images, mnist.test.labels)
    return tf.constant(images_feed), tf.constant(labels_feed)

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
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Ylogits, tf.one_hot(Y_, 10)))*100
    train_op = layers.optimize_loss(loss, framework.get_global_step(), 0.001, "Adam")
    return {"predictions":predict, "classes": classes}, loss, train_op

def softmax_model(X, Y_, mode):
    Ylogits = layers.linear(X, 10)
    predict = tf.nn.softmax(Ylogits)
    classes = tf.cast(tf.argmax(predict, 1), tf.uint8)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Ylogits, tf.one_hot(Y_, 10)))*100
    train_op = layers.optimize_loss(loss, framework.get_global_step(), 0.003, "Adam")
    return {"predictions":predict, "classes": classes}, loss, train_op

evaluationMetrics = {
    'my_accuracy': MetricSpec(
        metric_fn=metrics.accuracy,
        prediction_key='classes')
}

trainingConfig = tf.contrib.learn.RunConfig(save_checkpoints_secs=60)

def experiment_fn(output_dir):
    ITERATIONS = 10000
    mnist = input_data.read_data_sets(tempfile.mkdtemp())
    return learn.Experiment(
    estimator=learn.Estimator(model_fn=conv_model, model_dir=output_dir, config=trainingConfig),
    train_input_fn=lambda: train_data_input_fn(mnist),
    eval_input_fn=lambda: eval_data_input_fn(mnist),
    train_steps=ITERATIONS,
    eval_steps=1,
    local_eval_frequency=30, #secs between evals (?) - deprecated but learn_runner needs updating...
    eval_metrics=evaluationMetrics
)

def main(argv=None):
    learn_runner.run(experiment_fn, FLAGS.train_dir)

if __name__ == '__main__':
    main()
