/**
 * Copyright 2018 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

var auth2  // The Sign-In object.
var googlemap = null  // Google maps API map instance
var mlengine = null  // ML engine API handle
var googlemapevtlistener = null

var payload_tiles = []  // extracted images. format {image_bytes: ,pos:{x: ,y: ,sz: }}

var zone_width = 1550
var zone_height = 1030
//var zone_width = 1000
//var zone_height = 760
var tile_size = 256
// acceptable tile sizes depend on ML Engine model used. Currently:
// plane_jpeg_scan_100_200_300_400_600_900 supports square tiles of 100, 200, 300, 400, 600, 900 and 256 pixels
// jpeg_yolo_256x256 supports square tiles of 256 pixels only
var tile_delay = 50  // delay in ms between consecutive calls to ML Engine online predictions API (can be 0)

var scrollTimer;

// airports with their coordinates
var airports = new Object()
airports.TLS = [43.629450, 1.364613]
airports.LAX = [33.943560, -118.411534]
airports.SFO = [37.618887, -122.379771]
airports.SFO2 = [37.61986428512326, -122.39355755433655]
airports.CDG = [49.009441, 2.557194]
airports.CDG2 = [49.00680173533534, 2.5710369114937066]
airports.LBG = [48.961900, 2.439519]
airports.LBG2 = [48.963465356635695, 2.4394921779098695]
airports.SEA = [47.443495, -122.307206]
airports.SEA2 = [47.43538206139657, -122.30503877511597]
airports.NRT = [35.764783, 140.390962]
airports.ICN = [37.461626, 126.443786]
airports.DMA = [32.160020, -110.836055]

// test and training images
