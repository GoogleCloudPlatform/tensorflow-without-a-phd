"""
Copyright 2018 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
_______________________________________________________________________

Global settings for YOLO (You Look Only Once) detection model"""

# ROI = Region of Interest

TILE_SIZE = 256  # size of training and inference images
MAX_DETECTED_ROIS_PER_TILE = 60  # max number of ROIs detected in images. The max possible is GRID_N * GRID_N * CELL_B.
MAX_TARGET_ROIS_PER_TILE = 50  # max number of ground truth ROIs in training or test images
