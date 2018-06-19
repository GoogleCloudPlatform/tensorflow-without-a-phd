import sys
import time
import argparse
import tensorflow as tf
from tensorflow.python.lib.io import file_io as gcsfile
from tensorflow.python.platform import tf_logging as logging

from trainer_yolo import model
from trainer_yolo import datagen

logging.set_verbosity(logging.INFO)
logging.log(logging.INFO, "Tensorflow version " + tf.__version__)


# Test script for reading tiles out of TF record files.
# This can be deleted before release.

def main():
    tfrec_filelist = gcsfile.get_matching_files("sample_data/tiledata2" + "/*.tfrecord")
    yolo_cfg = datagen.YOLOConfig(16, 2, True, 1.3)
    features, labels = datagen.train_data_input_fn_from_tfrecords(tfrec_filelist, 10, 200, yolo_cfg)

    jpegs = tf.map_fn(tf.image.encode_jpeg, features['image'] , dtype=tf.string)


    with tf.Session() as sess:
        n = 0
        start_time = time.time()
        while True:
            n += 1
            try:
                features_, labels_, jpegs_ = sess.run([features, labels, jpegs])

                # write the image out
                for k, jpeg in enumerate(jpegs_):
                    with open("sample_data/tiledata2/extractedIMG_" + str(n*100+k) + ".jpg", "wb") as f:
                        f.write(jpeg)

                batch_size = features_['image'].shape[0]
                duration = time.time() - start_time
                time_per_image = duration / (n * batch_size)
                print(str(n) + ": " + str(1.0/time_per_image) + " tiles/s")
            except tf.errors.OutOfRangeError:
                break
            except tf.errors.NotFoundError:
                break

if __name__ == '__main__':
    main()