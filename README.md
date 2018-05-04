# Reinforcement Learning Pong on Cloud Machine Learning Engine

## Introduction


## Prerequisites

- Python 2.7
- [Google Cloud SDK](https://cloud.google.com/sdk/)

## Installation

```
pip install -r requirements.txt
```

To verify that the packages are installed successfully, run the following command to see a round of Pong gameplay rendered on screen:

```
python -m trainer.task --render
```

You can also run the following command for training locally, which defaults to write the output (TensorBoard summary and checkpoints) to `/tmp/pong_output`:

```
python -m trainer.task
```

## Run Training Job

To submit the training job to Cloud Machine Learning Engine:

```
GCS_BUCKET=<gs://YOUR-BUCKET/> bash submit.sh
```


## Monitor Training Job

1. You can monitor the training job and find the logs on the [console](https://console.cloud.google.com/mlengine/jobs).

1. To monitor the progress of training, use TensorBoard:

    ```
    tensorboard --logdir gs://YOUR-BUCKET/JOB_NAME
    ```

    Here `JOB_NAME` is of the format of `"pong_[timestamp]"`, which you can find on the console.

1. To watch Pong gameplay with a trained model, run:

    ```
    python -m trainer.task --output-dir gs://YOUR-BUCKET/JOB_NAME --restore --render
    ```

    It might take about 10 minutes before the first checkpoint is saved, and it might take a few hours before the trained model can consistently win a some points.  It might take about 20 hours of training before the trained model can consistently win games.


