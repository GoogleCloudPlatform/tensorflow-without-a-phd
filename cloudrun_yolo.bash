#!/usr/bin/env bash

#CONFIG="config-distributed.yaml"
#CONFIG="config-master-worker.yaml"
CONFIG="config.yaml"
#CONFIG="config-hptune-yolo1.yaml"
BUCKET="gs://ml1-demo-martin"
DATA="gs://ml1-demo-martin/data/USGS_public_domain_airports"
TILEDATA="gs://ml1-demo-martin/data/USGS_public_domain_tiled_airports_tfrecords"
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
gcloud ml-engine jobs submit training airplane$N \
    --job-dir "${BUCKET}/jobs/airplane$N" \
    --config ${CONFIG} \
    --project ${PROJECT} \
    --region ${REGION} \
    --module-name trainer_yolo.main \
    --package-path trainer_yolo \
    --runtime-version 1.4 \
    -- \
    --tiledata "${TILEDATA}" \
    --hp-shuffle-buf 50000 \
    --hp-iterations 25000 \
    --hp-lr2 5000 \
    --hp-layers 12 \
    --hp-first-layer-filter-depth 32 \
    --hp-first-layer-filter-size 6 \
    --hp-first-layer-filter-stride 2 \
    --hp-depth-increment 5
