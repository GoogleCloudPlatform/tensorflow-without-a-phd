#!/usr/bin/env bash

#CONFIG="config-distributed.yaml"
#CONFIG="config-master-worker.yaml"
CONFIG="config.yaml"
#CONFIG="config-hptune-yolo1.yaml"
BUCKET="gs://ml1-demo-martin"
DATA="gs://ml1-demo-martin/data/USGS_public_domain_airports"
#DATA="gs://ml1-demo-martin/data/USGS_public_domainTINY_airports"
PROJECT="cloudml-demo-martin"
REGION="us-central1"
#REGION="europe-west1"

# auto-incrementing run number padded with zeros to 3 digits
NFILE="cloudrunN.txt"
if [ ! -f $NFILE ]; then echo "0" > $NFILE; fi
read -r line < $NFILE
N=$((line+1))
echo $N > $NFILE;
printf -v N "%03d" $N

set -x
gcloud ml-engine jobs submit training plane$N \
    --job-dir "${BUCKET}/jobs/plane$N" \
    --config ${CONFIG} \
    --project ${PROJECT} \
    --region ${REGION} \
    --module-name trainer_yolo.train \
    --package-path trainer_yolo \
    --runtime-version 1.4 \
    -- \
    --data "${DATA}" \
    --hp-iterations 25000 \
    --hp-lw1 1 \
    --hp-lw2 1 \
    --hp-lw3 1 \
    --hp-lr2 5000 \
    --hp-rnd-distmax 2.0 \
    --hp-grid-nn 16 \
    --hp-cell-n 2 \
    --hp-cell-swarm True \
    --hp-cell-grow 1.0 \
    --hp-rnd-hue True \
    --hp-dropout 0.0 \
    --hp-spatial-dropout True \
    --hp-shuffle-buf 50000 \
    --hp-layers 15 \
    --hp-first-layer-filter-size 3 \
    --hp-first-layer-filter-stride 1 \
    --hp-first-layer-filter-depth 128 \
    --hp-depth-increment 8
