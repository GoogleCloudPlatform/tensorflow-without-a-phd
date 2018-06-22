#!/usr/bin/env bash

#CONFIG="config.yaml"
#CONFIG="config-distributed.yaml"
#CONFIG="config-master-worker.yaml"
#CONFIG="config-hptune-yolo1.yaml"
BUCKET="gs://ml1-demo-martin"
DATA="gs://ml1-demo-martin/data/USGS_public_domain_airports"
TILEDATA="gs://ml1-demo-martin/data/USGS_public_domain_tiles100_airports_tfrecords"
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
    --scale-tier BASIC_GPU \
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
    --hp-dropout 0.0
# The parameters above were used on the best training run: airplane806

