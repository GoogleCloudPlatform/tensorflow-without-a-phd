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

# Concatenates two or more planesnet data files. The result has the same name as the first file with
#  ".concat" inserted before the extension. Result picke format is the same as the format of the first file.

import os
import sys
import gzip
import pickle
import itertools
import numpy as np

def main(argv):
    if len(argv)<2:
        print("usage: python dataconcat.py file1 file2 file3 ... ")
        return -1

    data = []
    labels = []
    outfilename = ""
    pickle_protocol = None
    for filename in argv[1:]:
        # first filename is the base for output name
        # if first file has .pklz2 extension, result is pickled into python 2 format
        if outfilename == "":
            outfilename, ext = os.path.splitext(filename)
            if ext == ".pklz2":
                pickle_protocol = 2
            outfilename += ".concat" + ext

        with gzip.open(filename, mode='rb') as f:
            unpickled = pickle.load(f)
            data.append(unpickled["data"])
            labels.append(unpickled["labels"])

    with gzip.open(outfilename, mode='wb') as d:
        data = np.concatenate(data)
        labels = list(itertools.chain(*labels))
        pickle.dump({"data": data, "labels": labels}, d, protocol=pickle_protocol)

if __name__ == '__main__':
    main(sys.argv)