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

// Google Maps API initialisation
function initMap() {
    var query = new URLSearchParams(window.location.search)
    var lat = query.get("lat")
    var lng = query.get("lng")
    var zoom = query.get("zoom")
    var latlng
    if (!lat || lat == "undefined") // in case 'undefined' ended up in the URL
        latlng = { lat: airports.TLS[0], lng: airports.TLS[1] } // Toulouse Blagnac airport
    else
        latlng = {lat: parseFloat(lat), lng: parseFloat(lng)}
    if (!zoom || zoom == "undefined")
        zoom = 16
    else
        zoom = parseInt(zoom)
    googlemap = new google.maps.Map(document.getElementById('map'), {
        zoom: zoom,
        center: latlng,
        mapTypeId: google.maps.MapTypeId.SATELLITE
    })
    // this works on map load
    google.maps.event.addListenerOnce(googlemap, 'tilesloaded', grabPixels)
    // this works after the maps has loaded once, and for some reason the setTimeout is necessary
    google.maps.event.removeListener(googlemapevtlistener)
    googlemapevtlistener = google.maps.event.addListener(googlemap, 'idle', function() {
        setTimeout(grabPixels, 1)
    })
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

function setMapLocationInURL(latlng, zoom, reloadonce) {
    var urlparams = '?lat=' + latlng.lat() + '&lng=' + latlng.lng() + '&zoom=' + zoom
    if (reloadonce == 1)
        urlparams += '&r=' + reloadonce
    history.pushState(null,null, urlparams)
}

function centerMap(lat, lng) {
    googlemap.setCenter({lat:lat, lng:lng})
}

function onScroll() {
    clearTimeout(scrollTimer)
    scrollTimer = setTimeout(onScrollDone, 500)
}

function onScrollDone() {
    grabPixels()
}

function centerImage(filename) {
    var img = new Image();
    img.src = filename;
    var imgnod  = document.getElementById('imgmap');
    imgnod.innerHTML = "";
    imgnod.appendChild(img);
    makeDragScrollable(imgnod)
    function onLoaded() {
        if (img.complete)
            grabPixels()
        else
            setTimeout(onLoaded, 500)
    }
    setTimeout(onLoaded, 500)
}

function switchToMap() {
    document.getElementById('map').style.display = "block"
    document.getElementById('imgmap') .style.display = "none"
    document.getElementById('zoomctrl') .style.display = "none"
}

function switchToImage() {
    document.getElementById('map').style.display = "none"
    document.getElementById('imgmap') .style.display = "block"
    document.getElementById('zoomctrl') .style.display = "block"
}

function centerMapOnCode(airportCode) {
    if (airportCode.startsWith("train_") || airportCode.startsWith("test_")) {
        filename = sampleImages[airportCode]
        if (filename === undefined)
            filename = sampleImages.ATL
        centerImage(filename)
        switchToImage()
    }
    else {
        coords = airports[airportCode]
        if (coords === undefined)
            coords = airports.TLS
        switchToMap()
        centerMap(coords[0], coords[1])
    }
    google.maps.event.addListenerOnce(googlemap, 'tilesloaded', grabPixels)
}


