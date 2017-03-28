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
from tensorflow.contrib.learn import MetricSpec
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import math
from mlengine.digits import test_digits
logging.set_verbosity(logging.INFO)

# This sample shows how to write Tensorflow models using the high-level layers API
# in Tensorflow. Using high-level APIs, you do not have to define placeholders and
# variables yourself. Also, you will not need to write your own training loop by
# using the Estimator interface instead.
#
# WARNING: tensorflow.contrib.learn.* APIs are still experimental and can change in breaking ways
# as they mature. API stability will be ensured when tensorflow.contrib.learn becomes tensorflow.learn

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = read_data_sets("data", one_hot=False, reshape=True, validation_size=0)

# In memory training data for this simple case.
# When data is too large to fit in memory, use Tensorflow queues.
def train_data_input_fn():
    return tf.train.shuffle_batch([tf.constant(mnist.train.images), tf.constant(mnist.train.labels)],
                                  batch_size=100, capacity=1100, min_after_dequeue=1000, enqueue_many=True)

# Eval data is an in-memory constant here.
def eval_data_input_fn():
    return tf.constant(mnist.test.images), tf.constant(mnist.test.labels)


# Test data for a predictions run
def predict_input_fn():
    return tf.cast(tf.constant(test_digits), tf.float32)


# Model loss (not needed in INFER mode)
def conv_model_loss(Ylogits, Y_, mode):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=tf.one_hot(Y_,10))) * 100 \
        if mode == learn.ModeKeys.TRAIN or mode == learn.ModeKeys.EVAL else None


# Model optimiser (only needed in TRAIN mode)
def conv_model_train_op(loss, mode):
    return layers.optimize_loss(loss, framework.get_global_step(), learning_rate=0.003, optimizer="Adam",
        # to remove learning rate decay, comment the next line
        learning_rate_decay_fn=lambda lr, step: 0.0001 + tf.train.exponential_decay(lr, step, -2000, math.e)
        ) if mode == learn.ModeKeys.TRAIN else None


# Model evaluation metric (not needed in INFER mode)
def conv_model_eval_metrics(classes, Y_, mode):
    # You can name the fields of your metrics dictionary as you like.
    return {'accuracy': metrics.accuracy(classes, Y_)} \
        if mode == learn.ModeKeys.TRAIN or mode == learn.ModeKeys.EVAL else None

# Model
def conv_model(X, Y_, mode):
    XX = tf.reshape(X, [-1, 28, 28, 1])
    biasInit = tf.constant_initializer(0.1, dtype=tf.float32)
    Y1 = layers.conv2d(XX,  num_outputs=6,  kernel_size=[6, 6], biases_initializer=biasInit)
    Y2 = layers.conv2d(Y1, num_outputs=12, kernel_size=[5, 5], stride=2, biases_initializer=biasInit)
    Y3 = layers.conv2d(Y2, num_outputs=24, kernel_size=[4, 4], stride=2, biases_initializer=biasInit)
    Y4 = layers.flatten(Y3)
    Y5 = layers.relu(Y4, 200, biases_initializer=biasInit)
    # to deactivate dropout on the dense layer, set keep_prob=1
    Y5d = layers.dropout(Y5, keep_prob=0.75, noise_shape=None, is_training=mode==learn.ModeKeys.TRAIN)
    Ylogits = layers.linear(Y5d, 10)
    predict = tf.nn.softmax(Ylogits)
    classes = tf.cast(tf.argmax(predict, 1), tf.uint8)

    loss = conv_model_loss(Ylogits, Y_, mode)
    train_op = conv_model_train_op(loss, mode)
    eval_metrics = conv_model_eval_metrics(classes, Y_, mode)

    return learn.ModelFnOps(
        mode=mode,
        # You can name the fields of your predictions dictionary as you like.
        predictions={"predictions": predict, "classes": classes},
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metrics
    )

# Configuration to save a checkpoint every 1000 steps.
training_config = tf.contrib.learn.RunConfig(save_checkpoints_secs=None, save_checkpoints_steps=1000, gpu_memory_fraction=0.9)

estimator=learn.Estimator(model_fn=conv_model, model_dir="checkpoints", config=training_config)

# Trains for 10000 additional steps saving checkpoints on a regular basis. The next
# training will resume from the checkpoint unless you delete the "checkpoints" folder.
estimator.fit(input_fn=train_data_input_fn, steps=10000)
estimator.evaluate(input_fn=eval_data_input_fn, steps=1)
digits = estimator.predict(input_fn=predict_input_fn)
for digit in digits:
    print(str(digit['classes']), str(digit['predictions']))