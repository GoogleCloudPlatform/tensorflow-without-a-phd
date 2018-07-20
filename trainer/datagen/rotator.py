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

# rotates plane tiles in order to generate more plane tiles.

import png
import gzip
import pickle
import numpy as np
from PIL import Image
from PIL import ImageChops

def image_dump(data_image, alpha, is_original, n):
    numchannels = 4 if alpha else 3
    qualifier = "o" if is_original else 'r'
    with open("debugdump_{}_{}.png".format(qualifier, n), 'wb') as imfile:
        imdata = data_image
        imdata = np.reshape(imdata, [numchannels, 20, 20])  # input format [rgb, y, x]
        imdata = np.moveaxis(imdata, 0, 2)  # output format [y, x, rgb]
        imdata = np.reshape(imdata, [-1, 20*numchannels])  # png lib expects a list of rows of pixels in (r,g,b) format
        w = png.Writer(20, 20, alpha=alpha)
        w.write(imfile, imdata)


def genRotated(infilename, outfilename):
    with gzip.open(infilename, mode='rb') as srcf:
        with gzip.open(outfilename, mode='wb') as destf:
            imcount = 0
            planesnet = pickle.load(srcf)
            # unpack dictionary
            data_images = planesnet['data']
            data_labels = np.array(planesnet['labels'])
            assert len(data_images) == len(data_labels)
            result = {"data":[], "labels": []}

            plane_images = data_images[0:8000]  # just planes
            for imdata in plane_images:
                # image_dump(imdata, False, True, imcount)  # debug
                imdata = np.reshape(imdata, (3, 20, 20), order="C")
                imdata = np.moveaxis(imdata, 0, 2)
                imdata = np.reshape(imdata, (-1))
                # original image
                im = Image.frombytes('RGB', (20, 20), imdata)
                im = im.convert("RGBA")
                for angle in range(30, 360, 30):  # start at 30°, we already have the original 0° image
                    # rotated image with black transparent corners
                    rot_im = im.rotate(angle, expand=False, resample=Image.BILINEAR)
                    # composite image
                    im2 = im.copy()
                    im2.alpha_composite(rot_im)
                    data = np.asarray([im2.getdata(band=0), im2.getdata(band=1), im2.getdata(band=2)], dtype=np.uint8)
                    data = np.reshape(data, [-1])  # to get it in exactly the same format as planesnet data
                    # image_dump(data, False, False, imcount)  # debug
                    result["data"].append(data)
                    result["labels"].append(1)  # this is an airplane
                    imcount += 1

            result["data"] = np.asarray(result["data"])  # to get it in exactly the same format as planesnet data
            pickle.dump(result, destf)
            print("Saved {} rotated airplane images to file {}".format(imcount, outfilename))

if __name__ == '__main__':
    genRotated("planesnet32K.pklz", "rotatedplanes.pklz")