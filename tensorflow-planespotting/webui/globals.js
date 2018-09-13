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
var tile_delay = 50 // delay in ms between consecutive calls to ML Engine online predictions API (can be 0)

// all button event handlers
var analyzeButton = document.getElementById('analyze-button');
var authorizeButton = document.getElementById('authorize-button');
var signoutButton = document.getElementById('signout-button');
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
trainImageDir = "USGS_public_domain_airports/"
testImageDir = "USGS_public_domain_airports_eval/"
var sampleImages = new Object()
sampleImages.test_DEN = testImageDir + "USGS_DEN.jpg"
sampleImages.test_LAX = testImageDir + "USGS_LAX.jpg"
sampleImages.test_MSY = testImageDir + "USGS_MSY.jpg"
sampleImages.test_TUC1l = testImageDir + "USGS_TUC1l.jpg"
sampleImages.test_TUC1s = testImageDir + "USGS_TUC1s.jpg"
sampleImages.test_TUC2s = testImageDir + "USGS_TUC2s.jpg"
sampleImages.test_TUC3s = testImageDir + "USGS_TUC3s.jpg"
sampleImages.test_TUC4s = testImageDir + "USGS_TUC4s.jpg"
sampleImages.train_ATL = trainImageDir + "USGS_ATL.jpg"
sampleImages.train_AUS = trainImageDir + "USGS_AUS.jpg"
sampleImages.train_BHM = trainImageDir + "USGS_BHM.jpg"
sampleImages.train_BOS = trainImageDir + "USGS_BOS.jpg"
sampleImages.train_BWI = trainImageDir + "USGS_BWI.jpg"
sampleImages.train_CLT = trainImageDir + "USGS_CLT.jpg"
sampleImages.train_CVG = trainImageDir + "USGS_CVG.jpg"
sampleImages.train_DSM = trainImageDir + "USGS_DSM.jpg"
sampleImages.train_EYW = trainImageDir + "USGS_EYW.jpg"
sampleImages.train_FAT = trainImageDir + "USGS_FAT.jpg"
sampleImages.train_FLG = trainImageDir + "USGS_FLG.jpg"
sampleImages.train_FLL = trainImageDir + "USGS_FLL.jpg"
sampleImages.train_FWA = trainImageDir + "USGS_FWA.jpg"
sampleImages.train_IAD = trainImageDir + "USGS_IAD.jpg"
sampleImages.train_IAH = trainImageDir + "USGS_IAH.jpg"
sampleImages.train_IND = trainImageDir + "USGS_IND.jpg"
sampleImages.train_MDW = trainImageDir + "USGS_MDW.jpg"
sampleImages.train_MIA = trainImageDir + "USGS_MIA.jpg"
sampleImages.train_OAK = trainImageDir + "USGS_OAK.jpg"
sampleImages.train_OHR = trainImageDir + "USGS_OHR.jpg"
sampleImages.train_PHX = trainImageDir + "USGS_PHX.jpg"
sampleImages.train_SAN = trainImageDir + "USGS_SAN.jpg"
sampleImages.train_SEA = trainImageDir + "USGS_SEA.jpg"
sampleImages.train_SFO = trainImageDir + "USGS_SFO.jpg"
sampleImages.train_SJC = trainImageDir + "USGS_SJC.jpg"

sampleImages.train_DMA = trainImageDir + "USGS_DMA.jpg"
sampleImages.train_DMA2 = trainImageDir + "USGS_DMA2.jpg"
