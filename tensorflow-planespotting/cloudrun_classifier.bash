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

# Use your own bucket and project here, source data must be downloaded from Kaggle
BUCKET="gs://ml1-demo-martin"
PROJECT="cloudml-demo-martin"
REGION="us-central1"
#DATA="gs://ml1-demo-martin/data/planesnet32K.bkg257K.pln88K.ppln88K.pklz2"
DATA="gs://ml1-demo-martin/data/planesnet32K.pklz2"

CONFIG="config.yaml"
#CONFIG="config-hptune-classifier.yaml"

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
    --data "${DATA}" \
    --hp-iterations 5000 \
    --hp-dense 43 \
    --hp-conv1 16 \
    --hp-dropout 0.3 \
    --hp-lr0 0.0086 \
    --hp-lr2 888 \
    --hp-filter-sizes S


