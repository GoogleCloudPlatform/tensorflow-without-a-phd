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
    payload = new Object()
    payload.instances = new Object()
    payload.instances.image_bytes = grabbed
    payload = JSON.stringify(payload)
    // magic formula: the body of the request goes into the "resource" parameter
    mlengine.projects.predict({name:"projects/cloudml-demo-martin/models/plane_jpeg/versions/v01", resource:payload})
        .then(function(res) {
            var nb_planes = 0
            var nb_results = 0
            for (var i=0; i<res.result.predictions.length; i++) {
                var p = res.result.predictions[i]
                if (p.classes)
                    nb_planes++
                nb_results++
            }
            console.info("Found planes:" + nb_planes)
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