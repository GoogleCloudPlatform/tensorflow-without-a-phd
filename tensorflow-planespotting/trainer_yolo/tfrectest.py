"""
Copyright 2018 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
_______________________________________________________________________

Debug script for data in TF Record format. Reads data files and outputs
JPEG tiles with ROIs (Regions of Interes) superimposed. Allows to check
visusally that the ROIs are correct."""

import sys
import time
import argparse
import tensorflow as tf
from tensorflow.python.lib.io import file_io as gcsfile
from tensorflow.python.platform import tf_logging as logging

from trainer_yolo import model
from trainer_yolo import datagen
from trainer_yolo import utils_imgdbg

logging.set_verbosity(logging.INFO)
logging.log(logging.INFO, "Tensorflow version " + tf.__version__)


# Test script for reading tiles out of TF record files.
# This can be deleted before release.

def main():
    tfrec_filelist = gcsfile.get_matching_files("sample_data/tilecache2" + "/*.tfrecord")
    yolo_cfg = datagen.YOLOConfig(16, 2, True, 1.3)
    dataset = datagen.train_dataset_from_tfrecords(tfrec_filelist, 10, 200, yolo_cfg, True, False)
    features, labels = dataset.make_one_shot_iterator().get_next()

    target_rois = labels["target_rois"]
    image = tf.to_float(features["image"]) / 255.0
    image = utils_imgdbg.draw_color_boxes(image, target_rois, 0.0, 1.0, 1.0)
    jpegs = tf.map_fn(tf.image.encode_jpeg, tf.cast(image*255, tf.uint8), dtype=tf.string)

    with tf.Session() as sess:
        n = 0
        start_time = time.time()
        while True:
            if n==50:
                start_time = time.time()
                n=0
            n += 1
            try:
                features_, target_rois_, jpegs_ = sess.run([features, target_rois, jpegs])

                # write the image out
                for k, jpeg in enumerate(jpegs_):
                    with open("sample_data/tilecache2/extractedIMG_" + str(n*100+k) + ".jpg", "wb") as f:
                        f.write(jpeg)

                # print(target_rois_)

                batch_size = features_['image'].shape[0]
                duration = time.time() - start_time
                time_per_image = duration / (n * batch_size)
                time_per_batch = duration
                # print(str(n) + ": " + str(1.0/time_per_batch) + " batches/s")
                print(str(n) + ": " + str(1.0/time_per_image) + " images/s")
            except tf.errors.OutOfRangeError:
                break
            except tf.errors.NotFoundError:
                break

if __name__ == '__main__':
    main()