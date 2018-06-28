#!/usr/bin/env bash

CONFIG="config.yaml"
#CONFIG="config-distributed.yaml"
#CONFIG="config-master-worker.yaml"
#CONFIG="config-hptune-yolo1.yaml"
BUCKET="gs://ml1-demo-martin"
DATA="gs://ml1-demo-martin/data/USGS_public_domain_airports"
#TILEDATA="gs://ml1-demo-martin/data/USGS_public_domain_tiles100_airports_tfrecords"
TILEDATA="gs://ml1-demo-martin/data/USGS_public_domain_tiles100_x166_rnd_orient_airports_tfrecords"
#DATA="gs://ml1-demo-martin/data/USGS_public_domainTINY_airports"
PROJECT="cloudml-demo-martin"
REGION="us-central1"

# auto-incrementing run number padded with zeros to 3 digits
NFILE="cloudrunN.txt"
if [ ! -f $NFILE ]; then echo "0" > $NFILE; fi
read -r line < $NFILE
N=$((line+1))
echo $N > $NFILE;
printf -v N "%03d" $N

set -x
gcloud ml-engine jobs submit training airplane$N \
    --job-dir "${BUCKET}/jobs/airplane$N" \
    --config ${CONFIG} \
    --project ${PROJECT} \
    --region ${REGION} \
    --module-name trainer_yolo.main \
    --package-path trainer_yolo \
    --runtime-version 1.8 \
    -- \
    --tiledata "${TILEDATA}" \
    --hp-shuffle-buf 5000 \
    --hp-iterations 120000 \
    --hp-lr2 15000 \
    --hp-layers 17 \
    --hp-first-layer-filter-depth 128 \
    --hp-first-layer-filter-size 3 \
    --hp-first-layer-filter-stride 1 \
    --hp-depth-increment 8 \
    --hp-dropout 0.0 \
    --hp-data-cache-n-epochs 2

#    --hp-decay-type cosine-restarts \
#    --hp-decay-restarts 5 \
#    --hp-decay-restart-height 0.99 \


# Model with fewest false positives: airplane806 (v806b). Training time: 24h, inference time: 2.8s
#gcloud ml-engine jobs submit training airplane$N \
#    --job-dir "${BUCKET}/jobs/airplane$N" \
#    --scale-tier BASIC_GPU \
#    --project ${PROJECT} \
#    --region ${REGION} \
#    --module-name trainer_yolo.main \
#    --package-path trainer_yolo \
#    --runtime-version 1.8 \
#    -- \
#    --tiledata gs://ml1-demo-martin/data/USGS_public_domain_tiles100_airports_tfrecords \
#    --hp-shuffle-buf 5000 \
#    --hp-iterations 120000 \
#    --hp-lr2 15000,
#    --hp-layers 17 \
#    --hp-first-layer-filter-depth 128 \
#    --hp-first-layer-filter-size 3 \
#    --hp-first-layer-filter-stride 1 \
#    --hp-depth-increment 8 \
#    --hp-dropout 0.0

# Best compromise between detection and false positives: airplane814 (v814). Training time: 24h, inference time: 2.8s
#gcloud ml-engine jobs submit training airplane$N \
#    --job-dir "${BUCKET}/jobs/airplane$N" \
#    --scale-tier BASIC_GPU \
#    --project ${PROJECT} \
#    --region ${REGION} \
#    --module-name trainer_yolo.main \
#    --package-path trainer_yolo \
#    --runtime-version 1.8 \
#    -- \
#    --tiledata gs://ml1-demo-martin/data/USGS_public_domain_tiles100_airports_tfrecords \
#    --hp-shuffle-buf 5000 \
#    --hp-iterations 120000 \
#    --hp-lr2 15000,
#    --hp-layers 17 \
#    --hp-first-layer-filter-depth 128 \
#    --hp-first-layer-filter-size 3 \
#    --hp-first-layer-filter-stride 1 \
#    --hp-depth-increment 8 \
#    --hp-dropout 0.15

# Best fast model: airplane795 (v795). Training time: 9h, inference time: 0.2s
#gcloud ml-engine jobs submit training airplane$N \
#    --job-dir "${BUCKET}/jobs/airplane$N" \
#    --scale-tier BASIC_GPU \
#    --project ${PROJECT} \
#    --region ${REGION} \
#    --module-name trainer_yolo.main \
#    --package-path trainer_yolo \
#    --runtime-version 1.8 \
#    -- \
#    --tiledata gs://ml1-demo-martin/data/USGS_public_domain_tiles100_airports_tfrecords \
#    --hp-shuffle-buf 5000 \
#    --hp-iterations 120000 \
#    --hp-lr2 5000,
#    --hp-layers 12 \
#    --hp-first-layer-filter-depth 32 \
#    --hp-first-layer-filter-size 6 \
#    --hp-first-layer-filter-stride 2 \
#    --hp-depth-increment 5 \
#    --hp-dropout 0.0

