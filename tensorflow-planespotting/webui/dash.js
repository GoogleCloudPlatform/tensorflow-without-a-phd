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

    // check if model selector contains data, if so make it visible
    addModel();

    // form event handlers
    addyourown.addEventListener('submit', function(evt) {
        if (addyourown.button.type == 'submit')
            toggleAddYourOwnEdit();
        else {
            addModel();
            toggleAddYourOwnButton();
        }
        evt.preventDefault();
        return false
    });

    addyourown.model.addEventListener('keyup', function(evt) {
        if (evt.which == 27) {
            toggleAddYourOwnButton();
            //evt.stopPropagation();
            evt.preventDefault();
        }
    })

    modsel.addEventListener('submit', function(evt) {
        analyze();
        evt.preventDefault();
        return false
    });

    signinUI.addEventListener('submit', function(evt) {
        if (signinUI.authorize.type == "submit")
            authorize();
        else if (signinUI.signout.type == "submit")
            signout();
        evt.preventDefault();
        return false
    });

});

function updateSigninStatus(isSignedIn) {
    if (isSignedIn) {
        signinUI.authorize.type = 'hidden';
        signinUI.signout.type = 'submit';
        modsel.analyze.type = 'submit'
    } else {
        signinUI.authorize.type = 'submit';
        signinUI.signout.type = "hidden";
        modsel.analyze.type = 'hidden'
    }
}

function analyze() {
    resetResults();
    var delay = 0;

    model_name = document.modsel.model.value
    // format some_project/some_model/some_version
    var model_parts = model_name.split('/');
    model_project_name = model_parts[0];
    model_basename = model_parts[1];
    model_baseversion = model_parts[2];

    var model_url = "projects/" + model_project_name;
    if (model_basename) model_url += "/models/" + model_basename;
    if (model_baseversion) model_url += "/versions/" + model_baseversion;

    // hack to revert coords for some old versions
    var model_version = model_url.match(/\/v([0-9]+)/);
    if (model_version == null)
        model_version = 99;  // all new version do not need the hack
    else
        model_version = model_version[1];

    processingURL(model_url);

    payload_tiles.map(function (tile) {
        setTimeout(function() {
            processingResults(tile)
            displayProcessingMarker(tile)
            var body = mlengineJSONify(tile)
            // magic formula: the body of the request goes into the "resource" parameter
            mlengine.projects.predict({
                name: model_url,
                resource: body
            })
                .then(function (res) {
                    if (res.result.error) {
                        undisplayProcessingMarker(tile)
                        displayErrorResults(tile, res.result.error)  // Error from ML Emgine
                    }
                    else {
                        var nb_planes = 0
                        var result_markers = []
                        res.result.predictions.map(function(prediction) {
                            //code for endpoint jpeg_yolo_256x256
                            if (prediction.rois !== undefined) {
                                for (var i = 0; i < prediction.rois.length; i++) {
                                    var roi = prediction.rois[i]
                                    var confidence = prediction.rois_confidence[i]
                                    if (confidence > 0.69) {
                                        nb_planes++
                                        result_markers.push(roi)
                                    }
                                }
                            }
                            // code for endpoint plane_jpeg_scan_100_200_300_400_600_900
                            else if (prediction.classes !== undefined) {
                                if (prediction.classes) {  // if this is a plane classes=1
                                    nb_planes++
                                    result_markers.push(prediction.boxes)
                                }
                            }
                        })
                        undisplayProcessingMarker(tile)
                        displayResultMarkers(tile, result_markers, model_version)
                        displayResults(tile, nb_planes)
                        console.info("Found planes:" + nb_planes)
                    }
                }, function(e) {
                    displayErrorMarker(tile)
                    displayRequestStatusError(tile, e)  // HTTP Error
                    logError(e)
                })
        }, (delay++)*tile_delay)
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
        var opacity = getComputedStyle(marker).opacity
        marker.style.opacity = 0
        zone.appendChild(marker)
        setTimeout(function() {marker.style.opacity = opacity}, 10)  // to allow opacity animation
    }
}

function undisplayProcessingMarker(tile) {
    var marker = document.getElementById(posIdentifier(tile, "processing_"))
    if (marker)
        marker.style.opacity = "0"
}

function displayErrorMarker(tile) {
    var marker = document.getElementById(posIdentifier(tile, "processing_"))
    if (marker)
        marker.style.backgroundColor = "red"
}

function invertCoords(box) {
    var bbox = []
    bbox[0] = box[1]
    bbox[1] = box[0]
    bbox[2] = box[3]
    bbox[3] = box[2]
    return bbox
}

function displayResultMarkers(tile, markers, model_version) {
    var zone = document.getElementById("zone")
    if (zone) {
        markers.map(function(box) {
            var marker = document.createElement("div")
            marker.classList = "zone-marker"
            var invert_coords = !(model_version == 19 || model_version == 20 || model_version == 23)
            var bbox = invert_coords ? invertCoords(box) : box
            marker.style.left = bbox[0] * tile.pos.sz + tile.pos.x + 'px'
            marker.style.top = bbox[1] * tile.pos.sz + tile.pos.y + 'px'
            marker.style.width = (bbox[2] - bbox[0]) * tile.pos.sz + 'px'
            marker.style.height = (bbox[3] - bbox[1]) * tile.pos.sz + 'px'
            zone.appendChild(marker)
        })
    }
}

function processingURL(url) {
    var sap = document.getElementById("sap")
    if (sap) {
        if (url && url != 'projects/')
            sap.innerHTML += url;
        else
            sap.innerHTML += "ERROR: there is no model specified";
    }
}

function processingResults(tile) {
    var sap = document.getElementById("sap")
    if (sap) {
        sap.innerHTML += "<div id='" + posIdentifier(tile, "reqid_") + "'>Processing...</div>"
    }
}

function displayRequestStatusError(tile, e) {
    var reqInfoMsg = document.getElementById(posIdentifier(tile, "reqid_"))
    if (reqInfoMsg) {
        var msg = ""
        if (e.result)
            msg = e.result.error.message
        reqInfoMsg.innerText = "Error " + e.status + ", " + msg
    }
}

function displayResults(tile, nb) {
    var reqInfoMsg = document.getElementById(posIdentifier(tile, "reqid_"))
    if (reqInfoMsg) {
        reqInfoMsg.innerText = "Planes: " + nb
    }
}

function displayErrorResults(tile, err) {
    var reqInfoMsg = document.getElementById(posIdentifier(tile, "reqid_"))
    if (reqInfoMsg) {
        reqInfoMsg.innerText = err
    }
}

function displayPayload(tiles) {
    var jap = document.getElementById("jap")
    if (jap)
        jap.innerText = mlengineJSONify(tiles[0])
}

function mlengineJSONify(tile) {
    var payload = new Object()
    payload.instances = [new Object()]  // single instance
    payload.instances[0].square_size = tile_size
    payload.instances[0].image_bytes = new Object()
    payload.instances[0].image_bytes.b64 = tile.image_bytes
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

function addModel() {
    var n
    for (n = 0; n < document.modsel.model.options.length; n++){
        if (document.modsel.model.options[n].value == document.addyourown.model.value){
            document.modsel.model.value = document.addyourown.model.value
            return
        }
    }
    if (document.addyourown.model.value) {
        var option = document.createElement("option")
        option.text = document.addyourown.model.value
        option.value = document.addyourown.model.value
        document.modsel.model.add(option)
        // select the newly added value
        document.modsel.model.value = document.addyourown.model.value
        n++
    }
    // make the select box visible if it has more than one value now
    if (n>0)
        document.modsel.model.style.visibility = "visible"
}

function toggleAddYourOwnEdit() {
    addyourown.button.type = 'hidden';
    addyourown.model.type = 'text';
    addyourown.model.focus()
}

function toggleAddYourOwnButton() {
    addyourown.button.type = 'submit';
    addyourown.model.type = 'hidden';
    setTimeout(function(){addyourown.button.focus()}, 0)
}

