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
import gzip
import pickle

def main(argv):
    if len(argv)<2:
        print("usage: python pickle32.py file")
        return -1

    filename = argv[1]

    with gzip.open(filename, mode='rb') as f:
        unpickled = pickle.load(f)
        with gzip.open(filename + '2', mode='wb') as d:
            pickle.dump(unpickled, d, protocol=2)

if __name__ == '__main__':
    main(sys.argv)