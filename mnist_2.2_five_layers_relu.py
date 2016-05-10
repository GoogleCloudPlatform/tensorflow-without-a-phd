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

import mnist_data
import tensorflow as tf
import tensorflowvisu
tf.set_random_seed(0)

# neural network with 5 layers
#
# · · · · · · · · · ·       (input data, flattened pixels)    X [batch, 784]   # 784 = 28*28
# \x/x\x/x\x/x\x/x\x/    -- fully connected layer (relu)      W1 [784, 200]      B1[200]
#  · · · · · · · · ·                                          Y1 [batch, 200]
#   \x/x\x/x\x/x\x/      -- fully connected layer (relu)      W2 [200, 100]      B2[100]
#    · · · · · · ·                                            Y2 [batch, 100]
#    \x/x\x/x\x/         -- fully connected layer (relu)      W3 [100, 60]       B3[60]
#     · · · · ·                                               Y3 [batch, 60]
#     \x/x\x/            -- fully connected layer (relu)      W4 [60, 30]        B4[30]
#      · · ·                                                  Y4 [batch, 30]
#      \x/               -- fully connected layer (softmax)   W5 [30, 10]        B5[10]
#       ·                                                     Y5 [batch, 10]

# Download images and labels
mnist = mnist_data.read_data_sets("data")

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 10])

# five layers and their number of neurons (tha last layer has 10 softmax neurons)
L = 200
M = 100
N = 60
O = 30
# Weights and biases initialised with small random values.
# When using RELUs, make sure biases are initialised with small *positive* values
W1 = tf.Variable(tf.truncated_normal([784, L], stddev=0.1))  # 784 = 28 * 28
B1 = tf.Variable(tf.truncated_normal([L], stddev=0.1, mean=0.2))
W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
B2 = tf.Variable(tf.truncated_normal([M], stddev=0.1, mean=0.2))
W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
B3 = tf.Variable(tf.truncated_normal([N], stddev=0.1, mean=0.2))
W4 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1))
B4 = tf.Variable(tf.truncated_normal([O], stddev=0.1, mean=0.2))
W5 = tf.Variable(tf.truncated_normal([O, 10], stddev=0.1))
B5 = tf.Variable(tf.zeros([10]))

#model
XX = tf.reshape(X, [-1, 28*28])
Y1l = tf.matmul(XX, W1) + B1
Y1 = tf.nn.relu(Y1l)
Y2l = tf.matmul(Y1, W2) + B2
Y2 = tf.nn.relu(Y2l)
Y3l = tf.matmul(Y2, W3) + B3
Y3 = tf.nn.relu(Y3l)
Y4l = tf.matmul(Y3, W4) + B4
Y4 = tf.nn.relu(Y4l)
Ylogits = tf.matmul(Y4, W5) + B5
Y = tf.nn.softmax(Ylogits)

# cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
# TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
# problems with log(0) which is NaN
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(Ylogits, Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# matplotlib visualisation
allweights = tf.concat(0, [tf.reshape(W1, [-1]), tf.reshape(W2, [-1]), tf.reshape(W3, [-1]), tf.reshape(W4, [-1]), tf.reshape(W5, [-1])])
allbiases  = tf.concat(0, [tf.reshape(B1, [-1]), tf.reshape(B2, [-1]), tf.reshape(B3, [-1]), tf.reshape(B4, [-1]), tf.reshape(B5, [-1])])
allactivations = tf.concat(0, [tf.reshape(Y1, [-1]), tf.reshape(Y2, [-1]), tf.reshape(Y3, [-1]), tf.reshape(Y4, [-1])])
alllogits = tf.concat(0, [tf.reshape(Y1l, [-1]), tf.reshape(Y2l, [-1]), tf.reshape(Y3l, [-1]), tf.reshape(Y4l, [-1]), tf.reshape(Ylogits, [-1])])
I = tensorflowvisu.tf_format_mnist_images(X, Y, Y_)
It = tensorflowvisu.tf_format_mnist_images(X, Y, Y_, 1000, lines=25)
datavis = tensorflowvisu.MnistDataVis(title4="Logits", title5="Activations", histogram4colornum=2, histogram5colornum=2)

# training step, learning rate = 0.003
train_step = tf.train.AdamOptimizer(0.003).minimize(cross_entropy)

# init
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# You can call this function in a loop to train the model, 100 images at a time
def training_step(i, update_test_data, update_train_data):

    # training on batches of 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(100)

    # compute training values for visualisation
    if update_train_data:
        a, c, im, al, aa = sess.run([accuracy, cross_entropy, I, alllogits, allactivations], {X: batch_X, Y_: batch_Y})
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c))
        datavis.append_training_curves_data(i, a, c)
        datavis.update_image1(im)
        datavis.append_data_histograms(i, al, aa)

    # compute test values for visualisation
    if update_test_data:
        a, c, im = sess.run([accuracy, cross_entropy, It], {X: mnist.test.images, Y_: mnist.test.labels})
        print(str(i) + ": ********* epoch " + str(i*100//mnist.train.images.shape[0]+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))
        datavis.append_test_curves_data(i, a, c)
        datavis.update_image2(im)

    # the backpropagation training step
    sess.run(train_step, {X: batch_X, Y_: batch_Y})

datavis.animate(training_step, iterations=10000+1, train_data_update_freq=10, test_data_update_freq=100, more_tests_at_start=True)

# to save the animation as a movie, add save_movie=True as an argument to datavis.animate
# to disable the visualisation use the following line instead of the datavis.animate line
# for i in range(10000+1): training_step(i, i % 100 == 0, i % 10 == 0)

print("max test accuracy: " + str(datavis.get_max_test_accuracy()))

# final test accuracy = 0.9801 (sigmoid, 20K iterations - really painful start...)
# final test accuracy = 0.9829 (relu, 20K iterations - normal quick start ...)
# with RELUs, accuracy should get above 0.97 in the first 2000 iterations
