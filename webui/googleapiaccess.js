/**
 * Copyright 2016 Google Inc.
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

var auth2 // The Sign-In object.

// Google Maps API initialisation
function initMap() {
    googlemap = new google.maps.Map(document.getElementById('map'), {
        zoom: 16,
        center: { lat: 43.629450, lng: 1.364613 }, // Toulouse Blagnac airport
        mapTypeId: google.maps.MapTypeId.SATELLITE
    })
    google.maps.event.addListenerOnce(googlemap, 'idle', grabPixels)
}
// Google Auth2, PubSub, CRM API initialisation
function handleClientLoad() {
}

function grabPixels() {
    e = document.getElementById('cap')
    m = document.getElementById('map')
    html2canvas(m, {
        onrendered: function (canvas) {
            e.appendChild(canvas);
        },
        width: 300,
        height: 300,
        useCORS: true
    })
}

function checkSignIn() {
    auth2 = gapi.auth2.getAuthInstance();
    // Listen for sign-in state changes.
    auth2.isSignedIn.listen(updateSigninStatus);
    // Handle the initial sign-in state.
    updateSigninStatus(auth2.isSignedIn.get());
}

function logError(err) {
    console.log(err)
}

function updateSigninStatus(isSignedIn) {
    if (isSignedIn) {
        //authorizeButton.style.display = 'none';
        //signoutButton.style.display = 'block';
        //angular.element(projectsSelect).scope().loadProjects();
    } else {
        //authorizeButton.style.display = 'block';
        //signoutButton.style.display = 'none';
        //projectsSelect.style.display = 'none';
        //topicsSelect.style.display = 'none';
        //startFetchingRidesButton.style.display = 'none';
        //stopFetchingRidesButton.style.display = 'none';
    }
}
