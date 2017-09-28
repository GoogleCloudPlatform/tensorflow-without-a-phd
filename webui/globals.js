/**
 * Copyright 2017 Google Inc.
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

var zone_width = 660
var zone_height = 360
var tile_size = 200  // zone is always tiled with one tile more than necessary in each direction to force overlap
// acceptable tile sizes depend on ML Engine model used. Currently plane_jpeg_scan_100_200_300_400_600_900.
var tile_delay = 20 // delay in ms between consecutive calls to ML Engine online predictions API (can be 0)

var reload_once = false

// all button event handlers
var analyzeButton = document.getElementById('analyze-button');
var authorizeButton = document.getElementById('authorize-button');
var signoutButton = document.getElementById('signout-button');

// airports with their coordinates
var airports = new Object()
airports.TLS = [43.629450, 1.364613]
airports.LAX = [33.943560, -118.411534]
airports.SFO = [37.618887, -122.379771]
airports.CDG = [49.009441, 2.557194]
airports.LBG = [48.961900, 2.439519]
airports.SEA = [47.443495, -122.307206]
airports.NRT = [35.764783, 140.390962]
airports.ICN = [37.461626, 126.443786]
