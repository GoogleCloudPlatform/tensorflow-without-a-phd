#from sklearn import metrics
from tensorflow.contrib.learn.python.learn.estimators._sklearn import accuracy_score
from tensorflow.contrib import learn
import tensorflow as tf
import datetime
import math

def learning_rate(i):
    max_learning_rate = tf.constant(0.003, dtype=tf.float32)
    min_learning_rate = tf.constant(0.0001, dtype=tf.float32)
    decay_speed = tf.constant(2000.0, dtype=tf.float32)
    lr = tf.add(min_learning_rate, tf.mul(tf.sub(max_learning_rate, min_learning_rate), tf.exp(tf.div(tf.to_float(-i),decay_speed))))
    return lr

### Download and load MNIST data.

mnist = learn.datasets.load_dataset('mnist')

### Convolutional network

def conv_model(X, y):
    X = tf.reshape(X, [-1, 28, 28, 1])
    with tf.variable_scope('conv_layer1'):
        h_conv1 = learn.ops.conv2d(X, n_filters=6, filter_shape=[6, 6], bias=True, activation=tf.nn.relu)
    with tf.variable_scope('conv_layer2'):
        h_conv2 = learn.ops.conv2d(h_conv1, n_filters=12, filter_shape=[5, 5], bias=True, activation=tf.nn.relu, strides=[1,2,2,1])
    with tf.variable_scope('conv_layer3'):
        h_conv3 = learn.ops.conv2d(h_conv2, n_filters=24, filter_shape=[4, 4], bias=True, activation=tf.nn.relu, strides=[1,2,2,1])
        h_conv3_flat = tf.reshape(h_conv3, [-1, 7 * 7 * 24])
    h_fc1 = learn.ops.dnn(h_conv3_flat, [200], activation=tf.nn.relu, dropout=0.75)
    return learn.models.logistic_regression(h_fc1, y)

# Training and predicting
classifier = learn.TensorFlowEstimator(model_fn=conv_model, n_classes=10, batch_size=100, steps=10000, learning_rate=learning_rate, verbose=2)
timestamp = datetime.datetime.now().timestamp()
classifier.fit(mnist.train.images, mnist.train.labels, logdir="logs/run"+str(timestamp))
score = accuracy_score(mnist.test.labels, classifier.predict(mnist.test.images))
print('Accuracy: {0:f}'.format(score))