#!/usr/bin/env bash

CONFIG="config.yaml"
BUCKET="gs://ml1-demo-martin"
#BUCKET="gs://ml-eurowest1-martin"
DATA="gs://ml1-demo-martin/data/planesnet32K.bkg82K.pklz2"
PROJECT="cloudml-demo-martin"
REGION="us-central1"
#REGION="euro-west1"

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
    --module-name trainer.train \
    --package-path trainer \
    --runtime-version 1.2 \
    -- \
    --data "${DATA}"

