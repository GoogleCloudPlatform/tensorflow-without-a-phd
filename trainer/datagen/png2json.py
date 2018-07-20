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

# Convert from pickle format 3 to pickle format 2 for use with Python 2.7
# Run this under Python 3

import sys
import png
import json

def main(argv):
    if len(argv)<2:
        print("usage: python pickle32.py file1.png [file2.png file3.png ...]")
        return -1

    images = []
    for i in range(1,len(argv)):
        filename = argv[i]

        pngdata = png.Reader(filename).asRGB8()
        rows = []
        for row in pngdata[2]:
            rows.append(row.tolist())
        images.append(rows)
    pngformat = {'image': images}
    # pngformat = {'width': pngdata[0],
    #              'height': pngdata[1],
    #              'image': rows,
    #              'metadata': pngdata[3]}
    # image = np.vstack(rows)
    #data = map(np.array, pngdata)
    #image = np.vstack(data)
    print(json.dumps(pngformat))

    # with gzip.open(filename, mode='rb') as f:
    #     unpickled = pickle.load(f)
    #     print(json.dumps(unpickled))

if __name__ == '__main__':
    main(sys.argv)