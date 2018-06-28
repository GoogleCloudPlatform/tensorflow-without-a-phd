#!/usr/bin/env bash

# this is a data generation job. No training will be performed.

CONFIG="config.yaml"
BUCKET="gs://ml1-demo-martin"
DATA="gs://ml1-demo-martin/data/USGS_public_domain_airports"
# Destination directory of the data. The folder as well as the <samename>_eval folder must exist.
TILEDATA="gs://ml1-demo-martin/data/USGS_public_domain_tiled_airports_tfrecords5"
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
gcloud ml-engine jobs submit training airplane_datagen$N \
    --job-dir "${BUCKET}/jobs/airplane_datagen$N" \
    --scale-tier BASIC \
    --project ${PROJECT} \
    --region ${REGION} \
    --module-name trainer_yolo.datagen \
    --package-path trainer_yolo \
    --runtime-version 1.8 \
    -- \
    --data "${DATA}" \
    --output-dir "${TILEDATA}" \
    --hp-data-rnd-orientation True \
    --hp-data-tiles-per-gt-roi 100
