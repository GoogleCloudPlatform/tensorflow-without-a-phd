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

var auth2 // The Sign-In object.

// Google Maps API initialisation
function initMap() {
    var query = new URLSearchParams(window.location.search)
    var lat = query.get("lat")
    var lng = query.get("lng")
    var latlng
    if (!lat || lat == "undefined") // in case 'undefined' ended up in the URL
        latlng = { lat: 43.629450, lng: 1.364613 } // Toulouse Blagnac airport
    else
        latlng = {lat: parseFloat(lat), lng: parseFloat(lng)}
    googlemap = new google.maps.Map(document.getElementById('map'), {
        zoom: 16,
        center: latlng,
        mapTypeId: google.maps.MapTypeId.SATELLITE
    })
    google.maps.event.addListener(googlemap, 'idle', grabPixels)
    google.maps.event.addListener(googlemap, 'click', setMapLocationInURL)
}
// Google Auth2, PubSub, CRM API initialisation
function handleClientLoad() {
    loadAuth2()
        .then(initAuth2, checkSignIn)
        .then(checkSignIn)
        .then(loadMLEngine, logError)
        .then(initMLEngine, logError)
}

function loadAuth2() {
    return new Promise(function(resolve, reject) {
        gapi.load('client:auth2', resolve)
    })
}

function initAuth2() {
    var toto = gapi.auth2.init({
        //client_id: '19808069448-df7e5a57c3ftmfk3e9tptk6s7942qpah.apps.googleusercontent.com',
        client_id: '606116430098-mjcbomnkksirtv4ped18biue5j10vm87.apps.googleusercontent.com',
        scope: 'profile https://www.googleapis.com/auth/cloud-platform'
    })
    return toto.then() // The API does not return a Promise but an object that returns a Promise from its .then() function
}

function checkSignIn() {
    auth2 = gapi.auth2.getAuthInstance();
    // Listen for sign-in state changes.
    auth2.isSignedIn.listen(updateSigninStatus);
    // Handle the initial sign-in state.
    updateSigninStatus(auth2.isSignedIn.get());
}

function loadGapiClient() {
    return new Promise(function(resolve, reject) {
        gapi.load('client', resolve)
    })
}

function loadMLEngine() {
    return gapi.client.load('ml', 'v1')
}

function initMLEngine() {
    mlengine = gapi.client.ml
}

function logError(err) {
    console.log(err)
}

function setMapLocationInURL(evt) {
    history.pushState(null,null,'?lat=' + evt.latLng.lat() + '&lng=' + evt.latLng.lng())
}


