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

def genBox(image_width, image_height, size, step, zoom_step, skip=1):
    # generates boxes that scan the image
    # image_width, image_height: input image size
    # size: smallest tile size (real tile size is size*zoom)
    # step: smallest step (real step is step*zoom)
    # zoom_step: reapeatedly applied zoom factor
    # skip: return 1 out of skip results only
    zoom = 1.0
    cnt = 0
    boxes = []
    while zoom <= min(image_width, image_height):
        s = size*zoom
        x = 0.0
        while x+s <= image_width:
            y = 0.0
            while y+s <= image_height:
                if (cnt % skip) == 0:
                    yield [x, y, x+s, y+s]
                cnt += 1
                y += step*zoom
            x += step*zoom
        zoom *= zoom_step