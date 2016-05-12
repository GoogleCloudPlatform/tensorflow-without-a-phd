# encoding: UTF-8
# Copyright 2016 Google.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Import data
import mnist_data as input_data
import tensorflow as tf
import tensorflowvisu
import math
tf.set_random_seed(0)

mnist = input_data.read_data_sets("data")

# neural network structure for this sample:
#
# · · · · · · · · · ·      (input data, 1-deep)                 X [batch, 28, 28, 1]
# @ @ @ @ @ @ @ @ @ @   -- conv. layer 5x5x1=>24 stride 1       W1 [5, 5, 1, 24]       B1 [24]
# ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                           Y1 [batch, 28, 28, 24]
#   @ @ @ @ @ @ @ @     -- conv. layer 5x5x24=>48 stride 2      W2 [5, 5, 24, 48]      B2 [48]
#   ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                             Y2 [batch, 14, 14, 48]
#     @ @ @ @ @ @       -- conv. layer 5x5x48=>24 stride 1      W3 [5, 5, 48, 24]      B3 [24]
#     ∶∶∶∶∶∶∶∶∶∶∶                                               Y3 [batch, 14, 14, 24]
#       @ @ @ @         -- conv. layer 5x5x24=>48 stride 2      W4 [5, 5, 24, 48]      B4 [48]
#       ∶∶∶∶∶∶∶                                                 Y4 [batch, 7, 7, 48] reshaped to Z1 [batch, 7*7*48]
#       \x/x\x/ -       -- fully connected layer (relu+dropout) Wf1 [7*7*48, 1024]     Bf1 [1024]
#        · · ·                                                  T1 [batch, 1024]
#        \x/x/ -        -- fully connected layer (relu+dropout) Wf2 [1024, 512]        Bf2 [512]
#         · ·                                                   T2 [batch, 512]
#         \x/           -- fully connected layer (softmax)      Ws [512, 10]           Bs [10]
#          ·                                                    V [batch, 10]

# input: 28 x 28 images pixel depth is 1
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
pkeep = tf.placeholder(tf.float32)
lr = tf.placeholder(tf.float32)

K = 6  # first convolutional layer output depth
L = 12  # second convolutional layer output depth
M = 24  # third convolutional layer
N = 200  # fully connected layer

W1 = tf.Variable(tf.truncated_normal([6, 6, 1, K], stddev=0.1))
B1 = tf.Variable(tf.constant(0.1, tf.float32, [K]))
W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
B2 = tf.Variable(tf.constant(0.1, tf.float32, [L]))
W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
B3 = tf.Variable(tf.constant(0.1, tf.float32, [M]))

W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1))
B4 = tf.Variable(tf.constant(0.1, tf.float32, [N]))
W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
B5 = tf.Variable(tf.constant(0.1, tf.float32, [10]))

stride = 1
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
stride = 2
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
stride = 2
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

YY = tf.reshape(Y3, shape=[-1, 7 * 7 * M])

Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
YY4 = tf.nn.dropout(Y4, pkeep)
Ylogits = tf.matmul(YY4, W5) + B5
Y = tf.nn.softmax(Ylogits)
Y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(Ylogits, Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100 # norlmalised for batches of 100 training images

# training rule
train = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# accuracy for display
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1)), tf.float32))

# all weights and biases, images for visualisation
allweights = tf.concat(0, [tf.reshape(W1, [-1]), tf.reshape(W2, [-1]), tf.reshape(W3, [-1]), tf.reshape(W4, [-1]), tf.reshape(W5, [-1])])
allbiases  = tf.concat(0, [tf.reshape(B1, [-1]), tf.reshape(B2, [-1]), tf.reshape(B3, [-1]), tf.reshape(B4, [-1]), tf.reshape(B5, [-1])])
I = tensorflowvisu.tf_format_mnist_images(X, Y, Y_)
It = tensorflowvisu.tf_format_mnist_images(X, Y, Y_, 1000, lines=25)

datavis = tensorflowvisu.MnistDataVis()

with tf.Session() as sess:
    tf.initialize_all_variables().run()

    def training_step(i, update_test_data, update_train_data):

        batch_X, batch_Y = mnist.train.next_batch(100)

        # learning rate decay
        max_learning_rate = 0.003
        min_learning_rate = 0.0001
        decay_speed = 2000
        learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)

        if update_train_data:
            feed = {X: batch_X, Y_: batch_Y, pkeep: 1.0, lr: learning_rate}
            a, c, weights, biases, im = sess.run([accuracy, cross_entropy, allweights, allbiases, I], feed_dict=feed)
            print(str(i) + ":training accuracy: " + str(a) + " (lr:" + str(learning_rate) + ")" + " (ce:" + str(c) + ")")
            datavis.append_training_curves_data(i, a, c)
            datavis.append_data_histograms(i, weights, biases)
            datavis.update_image1(im)

        if update_test_data:
            feed = {X: mnist.test.images, Y_: mnist.test.labels, pkeep: 1.0, lr: learning_rate}
            a, c, im = sess.run([accuracy, cross_entropy, It], feed_dict=feed)
            print(str(i) + ":test accuracy: " + str(a) + " (lr:" + str(learning_rate) + ")" + " (ce:" + str(c) + ")")
            datavis.append_test_curves_data(i, a, c)
            datavis.update_image2(im)

        train.run(feed_dict={X: batch_X, Y_: batch_Y, pkeep: 0.75, lr: learning_rate})

    datavis.animate(training_step, 10001, train_data_update_freq=20, test_data_update_freq=100)

    #for i in range (2000):
    #    training_step(i, i%100==0 and i>0, i%10==0)

    print("max test accuracy: " + str(datavis.get_max_test_accuracy()))

    # accuracy 0.9920 at 2600 iterations
    # accuracy 0.9932 at 2900 iterations
    # accuracy 0.9936 at 3800 iterations
    # accuracy 0.9937 at 8900 iterations
    # accuracy 0.9931 at 10K iterations
    # wow!
    # stable above 0.9930 for most of the run
    # test cross-entropy goes deep under 3.0
    # accuracy and cross entropy curves still a bit jittery. I will try a lower learning rate.

    # on another run, I do only 0.9931 max - where is the randomness coming from ???
    # yet another run, 0.9933 max - where is the randomness coming from ???
    # yet another run, 0.9936 max - beautiful cruise at 0.993
    # yet another run, 0.9940 at 4300 iterations wow!
