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

import sys
import argparse
import tensorflow as tf
from tensorflow.python.lib.io import file_io as gcsfile
from tensorflow.python.platform import tf_logging as logging

from trainer_yolo import model
from trainer_yolo import datagen

logging.set_verbosity(logging.INFO)
logging.log(logging.INFO, "Tensorflow version " + tf.__version__)


# input function for base64 encoded JPEG in JSON
# Called when the model is deployed for online predictions on Cloud ML Engine.
def serving_input_fn():

    # input expects a list of jpeg images

    input_bytes = {'image_bytes': tf.placeholder(tf.string),  # this should be shape [None] TODO: test and fix
                   'square_size': tf.placeholder(tf.int32)}

    input_images = input_bytes['image_bytes']

    def jpeg_to_bytes(jpeg):
        pixels = tf.image.decode_jpeg(jpeg, channels=3)
        pixels = tf.cast(pixels, dtype=tf.uint8)
        return pixels

    images = tf.map_fn(jpeg_to_bytes, input_images, dtype=tf.uint8)
    feature_dic = {'image': images}
    return tf.estimator.export.ServingInputReceiver(feature_dic, input_bytes)


def start_training(output_dir, hparams, data, tiledata, **kwargs):

    # YOLO configuration for ROI assignments
    yolo_cfg = datagen.YOLOConfig(hparams["grid_nn"], hparams["cell_n"], hparams["cell_swarm"], hparams["cell_grow"])
    eval_yolo_cfg = datagen.YOLOConfig(hparams["grid_nn"], hparams["cell_n"], hparams["cell_swarm"], 1.0)

    # data source selection: full aerial imagery of TFRecords containing individual 256x256 tiles
    if tiledata != "" and  data == "":  # training from tfrecords
        tfrec_filelist = gcsfile.get_matching_files(tiledata + "/*.tfrecord")
        train_data_input_fn = lambda: datagen.train_data_input_fn_from_tfrecords(tfrec_filelist,
                                                                                 hparams["batch_size"],
                                                                                 hparams["shuffle_buf"],
                                                                                 yolo_cfg)
        tfrec_filelist_eval = gcsfile.get_matching_files(tiledata + "_eval" + "/*.tfrecord")
        eval_data_input_fn = lambda: datagen.eval_data_input_fn_from_tfrecords(tfrec_filelist_eval,
                                                                               hparams["eval_batch_size"],
                                                                               eval_yolo_cfg)
    elif data != "" and  tiledata == "":  # training from aerial imagery directly
        img_filelist, roi_filelist = datagen.load_file_list(data)
        train_data_input_fn = lambda: datagen.train_data_input_fn_from_images(img_filelist, roi_filelist,
                                                                              hparams["batch_size"],
                                                                              hparams["shuffle_buf"],
                                                                              yolo_cfg,
                                                                              hparams["rnd_hue"],
                                                                              hparams["rnd_distmax"])
        img_filelist_eval, roi_filelist_eval = datagen.load_file_list(data + "_eval")
        eval_data_input_fn = lambda: datagen.eval_data_input_fn_from_images(img_filelist_eval, roi_filelist_eval,
                                                                            hparams["eval_batch_size"],
                                                                            eval_yolo_cfg)
    else:
        logging.log(logging.ERROR, "One and only one of parameters 'data' and 'tiledata' must be supplied.")
        return

    # Estimator configuration
    export_latest = tf.estimator.LatestExporter(name="planesnet",
                                                serving_input_receiver_fn=serving_input_fn,
                                                exports_to_keep=1)

    train_spec = tf.estimator.TrainSpec(input_fn=train_data_input_fn,
                                        max_steps=hparams["iterations"])

    eval_spec = tf.estimator.EvalSpec(input_fn=eval_data_input_fn,
                                      steps=99999, # evals until Dataset is exhausted (bug: steps=None works but disables evaluation logs)
                                      exporters=export_latest,
                                      start_delay_secs=1,  # Confirmed: this does not work (plane533 for ex.)
                                      throttle_secs=1)

    training_config = tf.estimator.RunConfig(model_dir=output_dir,
                                             save_summary_steps=100,
                                             save_checkpoints_steps=2000,
                                             keep_checkpoint_max=1)

    estimator=tf.estimator.Estimator(model_fn=model.model_fn,
                                     model_dir=output_dir,
                                     config=training_config,
                                     params=hparams)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

def main(argv):
    parser = argparse.ArgumentParser()
    # mandatory arguments format for ML Engine:
    # gcloud ml-engine jobs submit training jobXXX --job-dir=... --ml-engine-args -- --user-args
    def str2bool(v): return v=='True'
    parser.add_argument('--job-dir', default="checkpoints", help='GCS or local path where to store training checkpoints')
    parser.add_argument('--data', default="", help='Path to training data folder containing full-scale aerial imagery (can be on Google cloud storage gs://...). Eval data should be in a folder with the same name and and _eval suffix.')
    parser.add_argument('--tiledata', default="", help='Path to training data folder containing image tiles (can be on Google cloud storage gs://...). Eval data should be in a folder with the same name and and _eval suffix.')
    parser.add_argument('--hp-iterations', default=25000, type=int, help='Hyperparameter: number of training iterations')
    parser.add_argument('--hp-batch-size', default=10, type=int, help='Hyperparameter: training batch size')
    parser.add_argument('--hp-eval-batch-size', default=32, type=int, help='Hyperparameter: evaluation batch size')
    parser.add_argument('--hp-shuffle-buf', default=50000, type=int, help='Hyperparameter: data shuffle buffer size')
    parser.add_argument('--hp-layers', default=11, type=int, help='Hyperparameter: number of layers')
    parser.add_argument('--hp-first-layer-filter-size', default=3, type=int, help='Hyperparameter: filter size in first layer')
    parser.add_argument('--hp-first-layer-filter-stride', default=1, type=int, help='Hyperparameter: filter stride in first layer')
    parser.add_argument('--hp-first-layer-filter-depth', default=50, type=int, help='Hyperparameter: the number of filters in the first and last layers')
    parser.add_argument('--hp-depth-increment', default=5, type=int, help='Hyperparameter: increment the decrement filter depth by this amount between first and last layer')
    parser.add_argument('--hp-grid-nn', default=16, type=int, help='Hyperparameter: size of YOLO grid: grid-nn x grid-nn')
    parser.add_argument('--hp-cell-n', default=2, type=int, help='Hyperparameter: number of ROIs detected per YOLO grid cell')
    parser.add_argument('--hp-cell-swarm', default=True, type=str2bool, help='Hyperparameter: ground truth ROIs selection algorithm. The better swarm algorithm is only implemented for cell_n=2')
    parser.add_argument('--hp-cell-grow', default=1.3, type=float, help='Hyperparameter: ROIs allowed to be cetered beyond grid cell by this factor')
    parser.add_argument('--hp-lr0', default=0.01, type=float, help='Hyperparameter: initial (max) learning rate')
    parser.add_argument('--hp-lr1', default=0.0001, type=float, help='Hyperparameter: target (min) learning rate')
    parser.add_argument('--hp-lr2', default=3000, type=float, help='Hyperparameter: learning rate decay speed in steps. Learning rate decays by exp(-1) every N steps.')
    parser.add_argument('--hp-dropout', default=0.0, type=float, help='Hyperparameter: dropout rate. It should be between 0.0 and 0.5. 0.0 for no dropout.')
    parser.add_argument('--hp-spatial-dropout', default=False, type=str2bool, help='Hyperparameter: dropout type, spatial or ordinary. Spatial works better.')
    parser.add_argument('--hp-bnexp', default=0.993, type=float, help='Hyperparameter: exponential decay for batch norm moving averages.')
    parser.add_argument('--hp-lw1', default=1, type=float, help='Hyperparameter: loss weight LW1')
    parser.add_argument('--hp-lw2', default=3, type=float, help='Hyperparameter: loss weight LW2')
    parser.add_argument('--hp-lw3', default=30, type=float, help='Hyperparameter: loss weight LW3')
    # hyperparameters for training data generation. They do not affect test data.
    parser.add_argument('--hp-rnd-hue', default=True, type=str2bool, help='Hyperparameter: data augmentation with random hue on training images')
    parser.add_argument('--hp-rnd-distmax', default=2.0, type=float, help='Hyperparameter: training tiles selection max random distance from ground truth ROI (always 2.0 for eval tiles)')

    args = parser.parse_args()
    arguments = args.__dict__

    # TODO: spatial dropout should be true by default
    # TODO: split data generation of images and ROIs on one side, assignment of ROIs into YOLO grid cells
    # to the other. Keep YOLO assignments during training but put the rest into a data generation script or option.

    hparams = {k[3:]: v for k, v in arguments.items() if k.startswith('hp_')}
    otherargs = {k: v for k, v in arguments.items() if not k.startswith('hp_')}

    logging.log(logging.INFO, "Hyperparameters:" + str(sorted(hparams.items())))
    logging.log(logging.INFO, "Other parameters:" + str(sorted(otherargs.items())))

    output_dir = otherargs.pop('job_dir')
    start_training(output_dir, hparams, **otherargs)

if __name__ == '__main__':
    main(sys.argv)
