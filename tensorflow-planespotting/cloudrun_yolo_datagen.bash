#!/usr/bin/env bash

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

# This is a data generation job. No training will be performed.

# Set your own bucket and project here, as well as the destination folder
BUCKET="gs://ml1-demo-martin"
PROJECT="cloudml-demo-martin"
REGION="us-central1"
# Destination directory of the data. The folder as well as the <samename>_eval folder must exist.
TILEDATA="gs://ml1-demo-martin/data/USGS_public_domain_tiles100_x166_rnd_orient_airports_tfrecords2"

# Source data
DATA="gs://ml1-demo-martin/data/USGS_public_domain_airports"

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
    --runtime-version 1.9 \
    -- \
    --data "${DATA}" \
    --output-dir "${TILEDATA}" \
    --hp-data-rnd-orientation True \
    --hp-data-tiles-per-gt-roi 166
