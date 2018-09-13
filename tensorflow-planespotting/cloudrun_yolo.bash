#!/usr/bin/env bash

# Copyright 2018 Google LLC
#
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

# Use your own bucket and project here
BUCKET="gs://ml1-demo-martin"
PROJECT="cloudml-demo-martin"
REGION="us-central1"

CONFIG="config.yaml"
#CONFIG="config-distributed.yaml"
#CONFIG="config-hptune-yolo.yaml"
#DATA="gs://planespotting-data-public/USGS_public_domain_photos"
TILEDATA="gs://planespotting-data-public/tiles_from_USGS_photos"

# auto-incrementing run number padded with zeros to 3 digits
NFILE="cloudrunN.txt"
if [ ! -f $NFILE ]; then echo "0" > $NFILE; fi
read -r line < $NFILE
N=$((line+1))
echo $N > $NFILE;
printf -v N "%04d" $N

set -x
gcloud ml-engine jobs submit training airplane$N \
    --job-dir "${BUCKET}/jobs/airplane$N" \
    --config ${CONFIG} \
    --project ${PROJECT} \
    --region ${REGION} \
    --module-name trainer_yolo.main \
    --package-path trainer_yolo \
    --runtime-version 1.10 \
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

# Model with fewest false positives: airplane806 (v806b). Training time: 24h, inference time: 2.8s
#gcloud ml-engine jobs submit training airplane$N \
#    --job-dir "${BUCKET}/jobs/airplane$N" \
#    --scale-tier BASIC_GPU \
#    --project ${PROJECT} \
#    --region ${REGION} \
#    --module-name trainer_yolo.main \
#    --package-path trainer_yolo \
#    --runtime-version 1.9 \
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
#    --runtime-version 1.9 \
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
#    --runtime-version 1.9 \
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

# If you want to experiment with cosine-restart learning rate decay
#    --hp-dropout 0.0 \
#    --hp-decay-type cosine-restarts \
#    --hp-decay-restarts 4 \
#    --hp-decay-restart-height 0.99


