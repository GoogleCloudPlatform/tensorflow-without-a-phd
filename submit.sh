BUCKET="gs://sandbox-cmle/"

TRAINER_PACKAGE_PATH="./trainer"
MAIN_TRAINER_MODULE="trainer.task"

now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="pong_$now"

JOB_DIR=$BUCKET$JOB_NAME

gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir $JOB_DIR  \
    --package-path $TRAINER_PACKAGE_PATH \
    --module-name $MAIN_TRAINER_MODULE \
    --region us-central1 \
    --config config.yaml \
    -- \
    --output-dir "gs://pong-demo/pong_200_noop_lazy" \
    --learning-rate 0.0005 \
    --allow-noop \
    --beta 0.02 \
    --gamma 0.99 \
    --decay 0.99 \

    
# python trainer/task.py --render --dry-run --restore --allow-noop --output-dir "gs://pong-demo/rl-pong/pong_200_noop_smooth_new" --n-batch 1 --batch-size 1

# tensorboard --logdir="gs://pong-demo/rl-pong/pong_200_noop_smooth_new"