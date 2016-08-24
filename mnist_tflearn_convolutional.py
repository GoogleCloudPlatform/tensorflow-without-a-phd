from tensorflow.contrib.learn.python.learn.estimators._sklearn import accuracy_score
from tensorflow.contrib import learn
from tensorflow.contrib import layers
#from tensorflow.contrib import metrics
from tensorflow.contrib import framework
from tensorflow.contrib.learn import monitors
from tensorflow.python.platform import tf_logging as logging
import tensorflow as tf
import datetime
import numpy as np

tf.set_random_seed(0)

def learning_rate(lr, i):
    max_learning_rate = tf.constant(0.003, dtype=tf.float32)
    min_learning_rate = tf.constant(0.0001, dtype=tf.float32)
    decay_speed = tf.constant(2000.0, dtype=tf.float32)
    lr = tf.add(min_learning_rate, tf.mul(tf.sub(max_learning_rate, min_learning_rate), tf.exp(tf.div(tf.to_float(-i),decay_speed))))
    return lr

### Download and load MNIST data.

mnist = learn.datasets.load_dataset('mnist')


### Convolutional network
def conv_model(X, Y_, mode):
    XX = tf.reshape(X, [-1, 28, 28, 1])
    Y1 = layers.conv2d(XX,  num_outputs=6,  kernel_size=[6, 6], normalizer_fn=layers.batch_norm)
    Y2 = layers.conv2d(Y1, num_outputs=12, kernel_size=[5, 5], stride=2, normalizer_fn=layers.batch_norm)
    Y3 = layers.conv2d(Y2, num_outputs=24, kernel_size=[4, 4], stride=2, normalizer_fn=layers.batch_norm)
    Y4 = layers.flatten(Y3)
    Y5 = layers.relu(Y4, 200, normalizer_fn=layers.batch_norm)
    #Y6 = layers.dropout(Y5, keep_prob=0.75, is_training=(mode ==learn.ModeKeys.TRAIN))
    Ylogits = layers.linear(Y5, 10)
    #prediction = tf.arg_max(tf.nn.softmax(Ylogits), 1)
    ## dense to one-hot
    Y__ = tf.one_hot(tf.cast(Y_, tf.int32), 10, dtype=tf.float32)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Ylogits, Y__))
    train_op = layers.optimize_loss(loss, framework.get_global_step(), 0.003, "Adam", learning_rate_decay_fn=learning_rate)
    #train_op = layers.optimize_loss(loss, framework.get_global_step(), 0.003, "Adam")

    # 10000 iterations with batch normalization on all layers, no dropout : final accuracy 99.42%
    # 10000 iterations with batch normalization on all layers, dropout on fully connected layer : final accuracy 99.28%
    return Ylogits, loss, train_op

def conv_model2(X, Y_, mode):
    XX = tf.reshape(X, [-1, 28, 28, 1])
    Y1 = layers.conv2d(XX,  num_outputs=6,  kernel_size=[6, 6], normalizer_fn=layers.batch_norm)
    Y2 = layers.conv2d(Y1, num_outputs=12, kernel_size=[5, 5], stride=2, normalizer_fn=layers.batch_norm)
    Y3 = layers.conv2d(Y2, num_outputs=24, kernel_size=[4, 4], stride=2, normalizer_fn=layers.batch_norm)
    Y4 = layers.flatten(Y3)
    Y5 = layers.relu(Y4, 200, normalizer_fn=layers.batch_norm)
    Ylogits = layers.linear(Y5, 10)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Ylogits, tf.one_hot(tf.cast(Y_, tf.int32), 10, dtype=tf.float32)))
    train_op = layers.optimize_loss(loss, framework.get_global_step(), 0.003, "Adam", learning_rate_decay_fn=learning_rate)
    # 10000 iterations with batch normalization on all layers, 4 conv layers, 2 normal layers, no dropout : final accuracy 99.38%
    # (layer parameters 6x6x8s1, 5x5x16s2, 4x4x24s2, 3x3x32s1, 400, 200)
    # 10000 iterations with batch normalization on all layers, 3 conv layers, 2 normal layers, no dropout : final accuracy 99.27%
    # (layer parameters 6x6x8s1, 5x5x16s2, 4x4x32s2, 400, 200)
    # 10000 iterations with batch normalization on all layers, 3 conv layers, 2 normal layers, no dropout : final accuracy 99.33%
    # (layer parameters 6x6x6s1, 5x5x12s2, 4x4x24s2, 200, 50)
    return Ylogits, loss, train_op

def softmax_model(X, Y_, mode):
    Ylogits = layers.linear(X, 10)
    #prediction = tf.nn.softmax(Ylogits)
    Y__ = tf.one_hot(tf.cast(Y_, tf.int32), 10, dtype=tf.float32)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Ylogits, Y__))
    train_op = layers.optimize_loss(loss, framework.get_global_step(), 0.003, "Adam")
    return Ylogits, loss, train_op

def dense_model(X, Y_, mode):
    Y1 = layers.fully_connected(X, 200, normalizer_fn=layers.batch_norm)
    Ylogits = layers.linear(Y1, 10)
    #prediction = tf.nn.softmax(Ylogits)
    Y__ = tf.one_hot(tf.cast(Y_, tf.int32), 10, dtype=tf.float32)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Ylogits, Y__))
    train_op = layers.optimize_loss(loss, framework.get_global_step(), 0.003, "Adam")
    return Ylogits, loss, train_op

logging.set_verbosity(logging.INFO)

# Training and predicting
dense_train_labels = mnist.train.labels.astype(np.int64)
dense_test_labels = mnist.test.labels.astype(np.int64)
#mon = monitors.ValidationMonitor(mnist.test.images, dense_test_labels, every_n_steps=100)
classifier = learn.Classifier(model_fn=conv_model, n_classes=10, model_dir="logs/run"+str(datetime.datetime.now().timestamp()))
classifier.fit(mnist.train.images, dense_train_labels, batch_size=100, steps=10000)
score = accuracy_score(mnist.test.labels, classifier.predict(mnist.test.images))
print('Accuracy: {0:f}'.format(score))

## code graveyard

#classifier = learn.TensorFlowEstimator(model_fn=conv_model, n_classes=10, batch_size=100, optimizer="Adam", learning_rate=learning_rate, verbose=2)
#one_hot_train_labels = learn.datasets.mnist.dense_to_one_hot(mnist.train.labels, 10).astype(np.float32)
#one_hot_test_labels = learn.datasets.mnist.dense_to_one_hot(mnist.test.labels, 10).astype(np.float32)