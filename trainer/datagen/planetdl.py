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

import os
import sys
import gzip
import logging
import numpy as np
from pathlib import Path
import pickle
from tensorflow.python.lib.io import file_io as gcsfile
import requests

def list_planet_IDs(filename):
    with gcsfile.FileIO(filename, 'rb') as zf:
        with gzip.GzipFile(fileobj=zf, mode='rb') as f:
            planesnet = pickle.load(f)
            # unpack dictionary
            data_images = planesnet['data']
            data_labels = np.array(planesnet['labels'])
            data_latlon = np.array(planesnet['locations'])
            data_scnids = np.array(planesnet['scene_ids'])
            scenes_dict = {}
            for scnid in data_scnids:
                scenes_dict[scnid] = None
            scenes_list = list(scenes_dict)
            return scenes_list

def ping_scenes(scenelist, action):
    # Possible actions
    # status: list status
    # activate: activate inactive assets

    #logging.basicConfig(level=logging.DEBUG)
    session = requests.Session()
    session.auth = (os.environ['PL_API_KEY'], '')
    item_id = scenelist[0]
    item_type = "PSScene3Band"
    asset_type = "visual"

    status = []
    for sceneid in scenelist:
        item_filename = "{}.tiff".format(sceneid)
        item_exists = Path(item_filename).is_file()
        url = "https://api.planet.com/data/v1/item-types/{}/items/{}/assets/".format(item_type, sceneid)

        if (action == "download" and item_exists):
            print("skip {}".format(sceneid))
            continue

        print("ping {}".format(sceneid), end=" ")
        item = session.get(url)
        print(item.status_code, end=" ")
        if item.status_code == 429:
            raise Exception("rate limit error")

        data = item.json()
        item_activation_url = None
        item_status = None
        item_download_url = None
        item_activation_request_result = None
        item_to_download = None

        if asset_type in data:
            if "_links" in data[asset_type]:
                if "activate" in data[asset_type]["_links"]["activate"]:
                    item_activation_url = item.json()[asset_type]["_links"]["activate"]
            if "status" in data[asset_type]:
                item_status = data[asset_type]["status"]
            if "location" in data[asset_type]:
                item_download_url = data[asset_type]["location"]

        if item_status is not None:
            print(item_status, end=" ")

        if action=="activate" and item_status == "inactive" and item_activation_url is not None:
            print("act_request", end=" ")
            item_activation_request_result = session.post(item_activation_url).status_code
            print(item_activation_request_result, end=" ")

        if action=="download" and item_download_url is not None:
            item_to_download = item_filename
            r = session.get(item_download_url, stream=True)
            print("download " + str(r.status_code), end=" ")
            if r.status_code == 200:
                sz = 0
                with open(item_to_download, 'wb') as f:
                    print()
                    print("downloading: {:5d} MB".format(0), end="")
                    for chunk in r:
                        f.write(chunk)
                        sz += len(chunk)
                        if (sz % (1024*1024) == 0):
                            print("\r".format(sz//1024//1024), end="")
                            print("downloading: {:5d} MB".format(sz//1024//1024), end="")
            else:
                item_to_download = "error " + str(r.status_code)

        print()

        status.append({"status": item_status,
                       "id": sceneid,
                       "act_url": item_activation_url,
                       "dl_url": item_download_url,
                       "act_ok": item_activation_request_result,
                       "dl_ok": item_to_download})

    if action == "download":
        for stat in status:
            if stat["dl_url"] is None:
                print("{} {}".format(stat["id"], stat["status"]))
            else:
                print("{} downloaded: {}".format(stat["id"], stat["dl_ok"]))

    if action == "status":
        for stat in status:
            print("{} {}".format(stat["id"], stat["status"]))

    if action == "activate":
        for stat in status:
            if stat["act_ok"] is not None:
                print("{} {} act_request: {}".format(stat["id"], stat["status"], stat["act_ok"]))
            else:
                print("{} {}".format(stat["id"], stat["status"]))

    return status


if __name__ == '__main__':
    if (len(sys.argv) < 5) and not (sys.argv[0] == "activate" or sys.argv[0] == "download" or sys.argv[0] == "status"):
        print("Usage:")
        print("planetdl.py [status|activate|download] filename start_record end_record")
        print("example: planetdl.py status planesnet32K.pklz 0 100")
    action = sys.argv[1]
    filename = sys.argv[2]
    start = int(sys.argv[3])
    stop = int(sys.argv[4])
    ids = list_planet_IDs(filename)
    ping_scenes(ids[start:stop], action)