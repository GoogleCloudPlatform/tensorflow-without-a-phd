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
    payload = new Object()
    payload.instances = new Object()
    payload.instances.image_bytes = grabbed
    payload = JSON.stringify(payload)
    processingResults()
    // magic formula: the body of the request goes into the "resource" parameter
    mlengine.projects.predict({name:"projects/cloudml-demo-martin/models/plane_jpeg_scan_200x200", resource:payload})
        .then(function(res) {
            if (res.result.error)
                displayErrorResults(res.result.error)
            else {
                var nb_planes = 0
                var nb_results = 0
                var size = 200
                var zone = document.getElementById("zone")
                for (var i = 0; i < res.result.predictions.length; i++) {
                    var p = res.result.predictions[i]
                    if (p.classes) {
                        var marker = document.createElement("div")
                        marker.classList = "zone-marker"
                        marker.style.top = p.boxes[0] * size + 'px'
                        marker.style.left = p.boxes[1] * size + 'px'
                        marker.style.width = (p.boxes[2] - p.boxes[0]) * size + 'px'
                        marker.style.height = (p.boxes[3] - p.boxes[1]) * size + 'px'
                        zone.appendChild(marker)
                        nb_planes++
                    }
                    nb_results++
                }
                displayResults(nb_planes)
                console.info("Found planes:" + nb_planes)
            }
        }, logError)
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
}

function processingResults() {
    var sap = document.getElementById("sap")
    if (sap)
        sap.innerText = "Processing..."
}

function displayResults(nb) {
    var sap = document.getElementById("sap")
    if (sap)
        sap.innerText = "Planes: " + nb
}

function displayErrorResults(err) {
    var sap = document.getElementById("sap")
    if (sap)
        sap.innerText = err
}