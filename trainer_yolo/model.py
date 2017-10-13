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

import math
import tensorflow as tf
import trainer_yolo.digits as dd
from trainer_yolo import boxutils

def get_left_digits(classes):
    digits = tf.image.grayscale_to_rgb(dd.digits_left())
    digits = tf.image.resize_bilinear(digits, [256, 256])
    return tf.gather(digits, tf.minimum(classes,9)) # correct digits to be printed on the images

def get_right_digits(classes):
    digits = tf.image.grayscale_to_rgb(dd.digits_right())
    digits = tf.image.resize_bilinear(digits, [256, 256])
    return tf.gather(digits, tf.minimum(classes,9)) # correct digits to be printed on the images

def image_compose(img1, img2):
    # img1 has the real image
    # img2 has markings on a black background
    pix_r, pix_g, pix_b = tf.split(img2, 3, axis=3)
    alpha = tf.maximum(pix_r, pix_g)
    alpha = tf.maximum(pix_b, alpha)
    alpha = tf.concat([alpha, alpha, alpha], axis=3)
    return (img1*(1-alpha)+img2*alpha)

def draw_color_boxes(img, boxes, r, g, b):
    pix_r, _, _ = tf.split(img, 3, axis=3)
    black = tf.zeros(tf.shape(pix_r))
    box_r = tf.image.draw_bounding_boxes(black, boxes) * r
    box_g = tf.image.draw_bounding_boxes(black, boxes) * g
    box_b = tf.image.draw_bounding_boxes(black, boxes) * b
    box_img = tf.concat([box_r, box_g, box_b], axis=3)
    return image_compose(img, box_img)

# Model
def model_fn_squeeze(features, labels, mode, params):

    def learn_rate(lr, step):
        return params['lr1'] + tf.train.exponential_decay(lr, step, params['lr2'], 1/math.e)

    def batch_normalization(x):  # axis=-1 will work for both dense and convolutional layers
        return tf.layers.batch_normalization(x, axis=-1, momentum=params['bnexp'], epsilon=1e-5, center=True, scale=False, training=(mode == tf.estimator.ModeKeys.TRAIN))

    def layer_conv2d_batch_norm_relu(x, filters, kernel_size, strides=1):
        y = tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size, strides=strides, padding="same", activation=None, use_bias=False)
        return tf.nn.relu(batch_normalization(y))

    def layer_conv1x1_batch_norm(x, depth):
        y = tf.layers.conv2d(x, filters=depth, kernel_size=1, strides=1, padding="same", activation=None, use_bias=False)
        return batch_normalization(y)

    def layer_dense_batch_norm_relu_dropout(x, size, dropout_rate):
        y = tf.layers.dense(x, size, activation=None, use_bias=False)
        z = tf.nn.relu(batch_normalization(y))
        return tf.layers.dropout(z, rate=dropout_rate, training=(mode == tf.estimator.ModeKeys.TRAIN))

    # model inputs
    X = features["image"]
    X = tf.to_float(X) / 255.0 # input image format is uint8

    N_CLASSES = 2
    GRID_N = 16  # must be the same as in train.py
    CELL_B = 1   # must be the same as in train.py
    TILE_SIZE = 256  # must be the same as in train.py

    # simplified small squeezenet architecture
    Y1 = layer_conv2d_batch_norm_relu(X, filters=32, kernel_size=6, strides=2)  # output 128x128x32
    # maxpool
    Y2 = tf.layers.max_pooling2d(Y1, pool_size=3, strides=2, padding="same")    # output 64x64x32
    Y3 = layer_conv2d_batch_norm_relu(Y2, filters=16, kernel_size=1, strides=1) #squeeze
    Y4l = layer_conv2d_batch_norm_relu(Y3, filters=32, kernel_size=1, strides=1) #expand 1x1
    Y4t = layer_conv2d_batch_norm_relu(Y3, filters=32, kernel_size=3, strides=1) #expand 3x3
    Y4 = tf.concat([Y4l, Y4t], 3)                                                # output 64x64x64
    Y5 = layer_conv2d_batch_norm_relu(Y4, filters=32, kernel_size=1, strides=1) #squeeze
    Y6l = layer_conv2d_batch_norm_relu(Y5, filters=64, kernel_size=1, strides=1) #expand 1x1
    Y6t = layer_conv2d_batch_norm_relu(Y5, filters=64, kernel_size=3, strides=1) #expand 3x3
    Y6 = tf.concat([Y6l, Y6t], 3)                                                # output 64x64x128
    f=2
    g=2
    # maxpool
    Y9 = tf.layers.max_pooling2d(Y6, pool_size=3, strides=2, padding="same")   # output 32x32x128
    Y10 = layer_conv2d_batch_norm_relu(Y9, filters=32*f, kernel_size=1, strides=1) #squeeze
    Y11l = layer_conv2d_batch_norm_relu(Y10, filters=32*f, kernel_size=1, strides=1) #expand 1x1
    Y11t = layer_conv2d_batch_norm_relu(Y10, filters=32*f, kernel_size=3, strides=1) #expand 3x3
    Y11 = tf.concat([Y11l, Y11t], 3)                                              # output 32x32x64
    Y12 = layer_conv2d_batch_norm_relu(Y11, filters=16*2*g*f, kernel_size=1, strides=1) #squeeze
    Y12l = layer_conv2d_batch_norm_relu(Y12, filters=16*2*g*f, kernel_size=1, strides=1) #expand 1x1
    Y12t = layer_conv2d_batch_norm_relu(Y12, filters=16*2*g*f, kernel_size=3, strides=1) #expand 3x3
    Y13 = tf.concat([Y12l, Y12t], 3)                                              # output 32x32x32
    #maxpool
    Y16 = tf.layers.max_pooling2d(Y13, pool_size=3, strides=2, padding="same")    # output 16x16x32
    Y17 = layer_conv2d_batch_norm_relu(Y16, filters=16*g*f, kernel_size=1, strides=1) #squeeze
    Y17l = layer_conv2d_batch_norm_relu(Y17, filters=16*g*f, kernel_size=1, strides=1) #expand 1x1
    Y17t = layer_conv2d_batch_norm_relu(Y17, filters=16*g*f, kernel_size=3, strides=1) #expand 3x3
    Y18 = tf.concat([Y17l, Y17t], 3)                                              # output 16x16x32

    # counting head
    C19 = layer_conv2d_batch_norm_relu(Y18, filters=4, kernel_size=1, strides=1) # output 16*16*4
    C20 = tf.layers.average_pooling2d(C19, pool_size=4, strides=4, padding="valid") # 4x4x4
    C21 = tf.reshape(C20, [-1, 4*4*4])
    Clogits = tf.layers.dense(C21, N_CLASSES)
    predict = tf.nn.softmax(Clogits)
    classes = tf.cast(tf.argmax(predict, 1), tf.int32)

    # old bounding box head
    #T19 = layer_conv2d_batch_norm_relu(Y18, filters=CELL_B*4, kernel_size=1, strides=1) # output 16*16*12
    #T20 = tf.layers.average_pooling2d(T19, pool_size=4, strides=4, padding="valid") # 4x4x12 shape [batch, 4,4,12]
    # for each cell, this has CELL_B predictions of bounding box (x,y,w,c)
    # apply tanh for x, y and sigmoid for w, c
    #TX, TY, TW, TC = tf.split(T20, 4, axis=3)  # shape 4 x [batch, 4,4,3] = 4 x [batch, GRID_N, GRID_N, CELL_B]

    # bounding box head
    T19 = layer_conv2d_batch_norm_relu(Y18, filters=CELL_B*8*f*g, kernel_size=1, strides=1) # output 16*16*24 if CELL_B=3
    T20=T19
    # not needed at GRID_N=16
    #T20 = tf.layers.average_pooling2d(T19, pool_size=4, strides=4, padding="valid") # 4x4x12 shape [batch, 4,4,12] = [batch, GRID_N, GRID_N, CELL_B*8]
    # for each cell, this has CELL_B predictions of bounding box (x,y,w,c)
    # apply tanh for x, y and sigmoid for w, c
    TX0, TY0, TW0, TC0 = tf.split(T20, 4, axis=3)  # shape 4 x [batch, GRID_N, GRID_N, CELL_B*2]
    # TODO: idea: batch norm may be bad on this layer
    # TODO: try with a deeper layer as well
    # TODO: try a filtered convolution instead of pooling2d, maybe info from cell sides should be weighted differently
    TX = tf.nn.tanh(layer_conv1x1_batch_norm(TX0, depth=CELL_B))  # shape [batch, 4,4,CELL_B]
    TY = tf.nn.tanh(layer_conv1x1_batch_norm(TY0, depth=CELL_B))  # shape [batch, 4,4,CELL_B]
    TW = tf.nn.sigmoid(layer_conv1x1_batch_norm(TW0, depth=CELL_B))  # shape [batch, 4,4,CELL_B]
    TC = tf.nn.sigmoid(layer_conv1x1_batch_norm(TC0, depth=CELL_B))  # shape [batch, 4,4,CELL_B]

    # testing different options for W
    # Woption0
    #real_TW = TW
    # Woption1
    #real_TW = TW*TW
    # Woption2
    #real_TW = TW
    # Woption3
    #real_TW = tf.sqrt(TW)
    # Woption4
    real_TW = tf.sqrt(TW)
    # Woption5
    #real_TW = TW*TW

    rois = tf.stack([TX,TY,real_TW], axis=-1)  # shape [batch, GRID_N, GRID_N, CEL_B, 3]
    rois = boxutils.grid_cell_to_tile_coords(rois, GRID_N, TILE_SIZE)/TILE_SIZE # shape [batch, GRID_N, GRID_N, CELL_B, 4]
    rois = tf.reshape(rois, [-1, GRID_N*GRID_N*CELL_B, 4])
    roiC = tf.reshape(TC, [-1, GRID_N*GRID_N*CELL_B])

    loss = train_op = eval_metrics = None
    if mode != tf.estimator.ModeKeys.PREDICT:
        C_ = labels["count"]
        T_ = labels["target"]  # shape [4,4,3,3] = [batch, GRID_N, GRID_N, CEL_B, xyw]
        TX_, TY_, TW_ = tf.unstack(T_, 3, axis=-1) # shape 3 x [batch, 4,4,3] = [batch, GRID_N, GRID_N, CELL_B]
        # target probability is 1 if there is a corresponding target box, 0 otherwise
        TC_ = tf.cast(tf.greater(TW_, 0), tf.float32) # shape [batch, 4,4,3] = [batch, GRID_N, GRID_N, CELL_B]

        # debug: expected and predicted counts
        debug_img = X
        debug_img = image_compose(debug_img, get_left_digits(C_))
        debug_img = image_compose(debug_img, get_right_digits(classes))
        # debug: ground truth boxes in grey
        target_rois = boxutils.grid_cell_to_tile_coords(T_, GRID_N, TILE_SIZE)/TILE_SIZE
        target_rois = tf.reshape(target_rois, [-1, GRID_N*GRID_N*CELL_B, 4])
        debug_img = draw_color_boxes(debug_img, target_rois, 0.7, 0.7, 0.7)
        # debug: computed ROIs boxes in shades of yellow
        no_box = tf.zeros(tf.shape(rois))
        select = tf.stack([roiC, roiC, roiC, roiC], axis=-1)
        for i in range(9):
            debug_rois = tf.where(tf.greater(select, 0.1*(i+1)), rois, no_box)
            debug_img = draw_color_boxes(debug_img, debug_rois, 0.1*(i+2), 0.1*(i+2), 0)
        tf.summary.image("input_image", debug_img, max_outputs=10)


        # model outputs
        count_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(tf.one_hot(C_,N_CLASSES), Clogits))
        position_loss = tf.reduce_mean(TC_ * (tf.square(TX-TX_)+tf.square(TY-TY_)))
        # YOLO trick: take square root of predicted size for loss so as not to drown errors on small boxes

        # testing different options for W
        # Woption0
        #size_loss = tf.reduce_mean(TC_ * tf.square(tf.sqrt(TW)-tf.sqrt(TW_)) * 2)
        # Woption1
        #size_loss = tf.reduce_mean(TC_ * tf.square(TW-tf.sqrt(TW_)) * 2)
        # Woption2
        #size_loss = tf.reduce_mean(TC_ * tf.square(TW-TW_) * 2)
        # Woption3
        #size_loss = tf.reduce_mean(TC_ * tf.square(tf.sqrt(TW)-TW_) * 2)
        # Woption4
        size_loss = tf.reduce_mean(TC_ * tf.square(TW-TW_*TW_) * 2)
        # Woption5
        #size_loss = tf.reduce_mean(TC_ * tf.square(TW*TW-TW_) * 2)

        obj_loss = tf.reduce_mean(TC_ * tf.square(TC - 1))
        noobj_loss = tf.reduce_mean((1-TC_) * tf.square(TC - 0))
        # TODO: idea, add a per-cell plane/no plane detection head. Maybe it can force better gradients (?)
        # because current split of detections "per responsible bounding box" might be hard for a neural network
        # TODO: similar idea: if only one plane in cell, teach all CELL_B detectors to detect it
        # if multiple planes, then each one its own detector. This could avoid detectors on areas with planes to be trained to detect nothing.

        # YOLO trick: weights the different losses differently
        Lc = 5
        Lo = 0.5
        loss = count_loss + Lc*(position_loss + size_loss) + (obj_loss + Lo*noobj_loss)

        train_op = tf.contrib.layers.optimize_loss(loss, tf.train.get_global_step(), learning_rate=params['lr0'], optimizer="Adam", learning_rate_decay_fn=learn_rate)
        eval_metrics = {"counting_accuracy": tf.metrics.accuracy(classes, C_),
                        "counting_error": tf.metrics.mean(count_loss),
                        "position_error": tf.metrics.mean(Lc*position_loss),
                        "size_error": tf.metrics.mean(Lc*size_loss),
                        "plane_confidence_error": tf.metrics.mean(obj_loss),
                        "no_plane_confidence_error": tf.metrics.mean(Lo*noobj_loss)}
        #debug
        tf.summary.scalar("counting_error", count_loss)
        tf.summary.scalar("position_error", Lc*position_loss)
        tf.summary.scalar("size_error", Lc*size_loss)
        tf.summary.scalar("plane_confidence_error", obj_loss)
        tf.summary.scalar("no_plane_confidence_error", noobj_loss)
        tf.summary.scalar("loss", loss)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={"predictions": predict, "classes": classes, "rois":rois, "rois_confidence": roiC},  # name these fields as you like
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metrics,
        export_outputs={'classes': tf.estimator.export.PredictOutput({"predictions": predict, "classes": classes, "rois":rois, "rois_confidence": roiC})}
    )