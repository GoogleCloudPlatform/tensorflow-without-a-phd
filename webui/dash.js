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

document.addEventListener("DOMContentLoaded", function(event) {
    var zone = document.getElementById("zone")
    if (zone) {
        zone.style.width = zone_width + "px"
        zone.style.height = zone_height + "px"
    }
    var cap = document.getElementById("cap")
    if (cap) {
        cap.style.height = cap.clientWidth / zone_width  * zone_height + "px"
        cap.width = zone_width
        cap.height = zone_height
    }
});

function updateSigninStatus(isSignedIn) {
    if (isSignedIn) {
        authorizeButton.style.display = 'none';
        signoutButton.style.display = 'block';
        analyzeButton.style.display = 'block';
    } else {
        authorizeButton.style.display = 'block';
        signoutButton.style.display = 'none';
        analyzeButton.style.display = 'none';
    }
}

function analyze() {
    resetResults()
    payload_tiles.map(function (tile) {
        processingResults(tile)
        displayProcessingMarker(tile)
        var body = mlengineJSONify([tile])  // TODO: fix the online prediction serving_input_fn and remove the []
        // magic formula: the body of the request goes into the "resource" parameter
        mlengine.projects.predict({
            name: "projects/cloudml-demo-martin/models/plane_jpeg_scan_100_200_300_400_600_900/versions/v8",
            resource: body
        })
            .then(function (res) {
                if (res.result.error) {
                    undisplayProcessingMarker(tile)
                    displayErrorResults(tile, res.result.error)
                }
                else {
                    var nb_planes = 0
                    var nb_results = 0
                    var result_markers = []
                    res.result.predictions.map(function(prediction) {
                        if (prediction.classes) {
                            nb_planes++
                            result_markers.push(prediction.boxes)
                        }
                    })
                    undisplayProcessingMarker(tile)
                    displayResultMarkers(tile, result_markers)
                    displayResults(tile, nb_planes)
                    console.info("Found planes:" + nb_planes)
                }
            }, logError)
    })
}

function signout() {
    gapi.auth2.getAuthInstance().signOut()
    checkSignIn()
}

function authorize() {
    gapi.auth2.getAuthInstance().signIn()
    checkSignIn()
}

function resetResults() {
    var zone = document.getElementById("zone")
    if (zone)
        zone.innerHTML = ""
    var sap = document.getElementById("sap")
    if (sap) {
        sap.innerHTML = ""
    }
}

function posIdentifier(tile, prefix) {
    return prefix + tile.pos.x + "_" + tile.pos.y + "_" + tile.pos.sz
}

function displayProcessingMarker(tile) {
    var zone = document.getElementById("zone")
    if (zone) {
        var marker = document.createElement("div")
        marker.classList = "zone-processing-marker"
        marker.id = posIdentifier(tile, "processing_")
        marker.style.top = tile.pos.y + 'px'
        marker.style.left = tile.pos.x + 'px'
        marker.style.width = tile.pos.sz + 'px'
        marker.style.height = tile.pos.sz + 'px'
        zone.appendChild(marker)
        setTimeout(function() {marker.style.opacity = 0.4}, 100)
    }
}

function undisplayProcessingMarker(tile) {
    var marker = document.getElementById(posIdentifier(tile, "processing_"))
    if (marker)
        marker.style.opacity = "0"
}

function displayResultMarkers(tile, markers) {
    var zone = document.getElementById("zone")
    if (zone) {
        markers.map(function(box) {
            var marker = document.createElement("div")
            marker.classList = "zone-marker"
            marker.style.top = box[0] * tile.pos.sz + tile.pos.y + 'px'
            marker.style.left = box[1] * tile.pos.sz + tile.pos.x + 'px'
            marker.style.width = (box[2] - box[0]) * tile.pos.sz + 'px'
            marker.style.height = (box[3] - box[1]) * tile.pos.sz + 'px'
            zone.appendChild(marker)
        })
    }
}

function processingResults(tile) {
    var sap = document.getElementById("sap")
    if (sap) {
        sap.innerHTML += "<div id='" + posIdentifier(tile, "reqid_") + "'>Processing...</div>"
    }
}

function displayResults(tile, nb) {
    var reqInfoMsg = document.getElementById(posIdentifier(tile, "reqid_"))
    if (reqInfoMsg) {
        reqInfoMsg.innerText = "Planes: " + nb
    }
}

function displayErrorResults(err) {
    var sap = document.getElementById("sap")
    if (sap)
        sap.innerText = err
}

function displayPayload(tiles) {
    var jap = document.getElementById("jap")
    if (jap)
        jap.innerText = mlengineJSONify(tiles)
}

function mlengineJSONify(tiles) {
    var payload = new Object()
    payload.instances = [new Object()]  // single instance
    payload.instances[0].square_size = tile_size
    payload.instances[0].image_bytes = []
    tiles.map(function(tile) {
        var container = new Object()
        container.b64 = tile.image_bytes
        payload.instances[0].image_bytes.push(container)
    })
    json_payload = JSON.stringify(payload)
    return json_payload
}

function enableMapScroll() {
    var zc = document.getElementById("zone-container")
    if (zc) {
        zc.style.pointerEvents = "none"
    }
    var sap = document.getElementById("sap")
    if (sap) sap.innerHTML = "ready"
    console.log("+++Enable Map Scroll")
}

function disableMapScroll() {
    var zc = document.getElementById("zone-container")
    if (zc) {
        zc.style.pointerEvents = "auto"
        zc.style.cursor = "wait"
    }
    var sap = document.getElementById("sap")
    if (sap) sap.innerHTML = "Grabbing pixels..."
    console.log("---Disable Map Scroll")
}