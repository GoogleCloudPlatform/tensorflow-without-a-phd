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
var defaultMarkerWidth = 50
var minMarkerWidth = 10
var markerSaveFilename = ""
var markersOnDisk = [
    'USGS_OAK',
    'USGS_FAT',
    'USGS_FLG',
    'USGS_LAX',
    'USGS_PHX',
    'USGS_SFO',
    'USGS_BHM',
    'USGS_SAN',
    'USGS_SJC',
    'USGS_DEN',
    'USGS_IAD',
    'USGS_MIA',
    'USGS_FLL',
    'USGS_EYW',
    'USGS_ATL',
    'USGS_MDW',
    'USGS_OHR',
    'USGS_IND',
    'USGS_FWA',
    'USGS_DSM',
    'USGS_MSY',
    'USGS_BOS',
    'USGS_BWI',
    'USGS_CLT',
    'USGS_IAH',
    'USGS_CVG',
    'USGS_AUS',
    'USGS_SEA',
    'USGS_TUC1l',
    'USGS_TUC1s',
    'USGS_TUC2s',
    'USGS_TUC3s',
    'USGS_TUC4s']

document.addEventListener("DOMContentLoaded", function(event) {
    var image = document.querySelector("#container img")

    image.addEventListener("click", function(e) {
        marker = newMarkerByCenter(image, e.offsetX, e.offsetY, defaultMarkerWidth)
        addMarkerEventListeners(image, marker)
    })

    image.focus()
    image.addEventListener("keypress", function(e) {
        if (e.key == "Enter") {
            var data = listMarkers(image)
            var blob = new Blob([JSON.stringify(data)], {type: "application/json"});
            var a = document.getElementById("downloadlink")
            a.href = URL.createObjectURL(blob)
            a.download = markerSaveFilename
            a.click()
        }
    })

    image.addEventListener("load", function(e) {loadCorrespondingMarkers(e.target)})

    image.addEventListener("wheel", function(e) { // scroll while mouse wheel or Alt pressed: do nothing
        if (e.buttons == 4 || e.altKey) { endEvent(parent, e)}
    })

    var sel = document.querySelector("#sel")
    loadKnownMarkers(sel, image.src, "USGS_public_domain_airports", markersOnDisk)
})

function loadKnownMarkers(select, baseurl, folder, list) {
    var global_stats = new Object()
    global_stats.nfiles = 0
    global_stats.nb = 0
    global_stats.max = 0
    global_stats.min = 9999

    list.forEach(function(airport) {
        var option = document.createElement("option")
        option.value = airport
        option.innerText = airport
        select.appendChild(option)

        var filename = airport + '.json'
        var jsonurl  = new URL(folder, baseurl)
        jsonurl  = new URL(filename, jsonurl.href)
        loadMarkers(jsonurl.href, function (markerInfo) {
            var stats = computeStats(markerInfo)
            option.innerText = airport + " (" + stats.nb + ")"
            global_stats.nfiles += 1
            global_stats.nb += stats.nb
            global_stats.max = Math.max(stats.max, global_stats.max)
            global_stats.min = Math.min(stats.min, global_stats.min)
            displayMarkersOnDisk(global_stats)
        })
    })
}

function selectAirport(airport) {
    var image = document.querySelector("#container img")
    var url = new URL(airport + ".jpg", image.src)
    image.src = url.href
}

function addMarkerEventListeners(parent, marker) {
    marker.addEventListener("wheel", function(e) {
        if (e.buttons == 4 || e.altKey) { // mouse wheel or Alt pressed
            defaultMarkerWidth = resizeMarker(this, -e.deltaY)
            endEvent(parent, e)
        }
    })

    marker.addEventListener("contextmenu", function(e) {
        deleteMarker(this)
        endEvent(parent, e)
    })

    marker.addEventListener("click", function(e) {
        moveMarker(this, e.offsetX, e.offsetY)
        endEvent(parent, e)
    })
}

function endEvent(parent, event) {
    parent.focus()
    event.stopPropagation()
    event.preventDefault()
}

function newMarkerByCenter(parent, cx, cy, w) {
    var marker = document.createElement("div")
    marker.classList = "marker"
    marker.style.left = cx - w/2 + 'px'
    marker.style.top = cy - w/2 + 'px'
    marker.style.width = w + 'px'
    marker.style.height = w + 'px'
    parent.parentElement.appendChild(marker)
    refreshStats(parent.parentElement)
    return marker
}

function newMarkerByTopLeft(parent, x, y, w) {
    var marker = document.createElement("div")
    marker.classList = "marker"
    marker.style.left = x + 'px'
    marker.style.top = y + 'px'
    marker.style.width = w + 'px'
    marker.style.height = w + 'px'
    parent.parentElement.appendChild(marker)
    return marker
}

function deleteMarker(marker) {
    var parent = marker.parentElement
    marker.parentElement.removeChild(marker)
    refreshStats(parent)
}

function resizeMarker(marker, delta) {
    var w = marker.clientWidth
    var x = marker.offsetLeft
    var y = marker.offsetTop
    if (w + delta < minMarkerWidth)
        delta = minMarkerWidth - w
    w += delta
    x -= delta / 2
    y -= delta / 2
    marker.style.left = x + "px"
    marker.style.top = y + "px"
    marker.style.width = w + 'px'
    marker.style.height = w + 'px'
    refreshStats(marker.parentElement.parentElement)
    return w
}

function moveMarker(marker, cx, cy) {
    var w = marker.clientWidth
    marker.style.left = cx + marker.offsetLeft - w/2 + 'px'
    marker.style.top = cy + marker.offsetTop - w/2 + 'px'
}

function listMarkers(parent) {
    var markers = parent.parentElement.querySelectorAll(".marker")
    var markerInfo = new Object()
    markerInfo.markers = []
    markers.forEach(function (m) {
        var marker = new Object()
        marker.x = m.offsetLeft
        marker.y = m.offsetTop
        marker.w = m.clientWidth
        markerInfo.markers.push(marker)
    })
    return markerInfo
}

function restoreMarkers(parent, markers) {
    markers.forEach(function(m) {
        var marker = newMarkerByTopLeft(parent, m.x, m.y, m.w)
        addMarkerEventListeners(parent, marker)
    })
}

function removeAllMarkers(parent) {
    var oldmarkers = parent.parentElement.querySelectorAll(".marker")
    oldmarkers.forEach(deleteMarker)
}

function loadCorrespondingMarkers(element) {
    var fullname = element.src
    var ext = fullname.split('.').pop()
    var filename = fullname.split('/').pop()
    var nakedname = filename.substring(0, filename.length-ext.length)
    var jsonname = nakedname + "json"
    markerSaveFilename = jsonname
    var jsonurl = new URL(jsonname, element.src)
    removeAllMarkers(element)
    loadMarkers(jsonurl.href, function(markerInfo) {
        restoreMarkers(element, markerInfo.markers)
        displayStats(computeStats(markerInfo))
    })
    var nakedname = nakedname.substring(0, nakedname.length-1) // remove "."
    var sel = document.querySelector("#sel option[value='"+ nakedname +"']")
    if (sel)
        sel.selected = true
}

function loadMarkers(name, callback) {
    markers = fetch(name)
        .then(function(response) {
            if (response.ok)
                response.json().then(callback)
            else
                throw new Error("Error fetching JSON for markers")
        })
        .catch(function(error) {console.log(error)})
}

function computeStats(markerInfo) {
    var nb = markerInfo.markers.length
    var marker_sizes = markerInfo.markers.map(function (e) {return e.w})
    var max = marker_sizes.reduce(function(a,b) {
        return Math.max(a, b)
    }, 0)
    var min = marker_sizes.reduce(function(a,b) {
        return Math.min(a, b)
    }, max)
    return {nb: nb, min: min, max: max}
}

function displayStats(stats) {
    var nb_elem = document.getElementById("stats_nbmarkers")
    nb_elem.innerText = stats.nb
    var min_elem = document.getElementById("stats_minsz")
    min_elem.innerText = stats.min
    var max_elem = document.getElementById("stats_maxsz")
    max_elem.innerText = stats.max
}

function refreshStats(parent) {
    displayStats(computeStats(listMarkers(parent)))
}

function displayMarkersOnDisk(stats) {
    var nbf_elem = document.getElementById("gstats_nbfiles")
    nbf_elem.innerText = stats.nfiles
    var nb_elem = document.getElementById("gstats_nbmarkers")
    nb_elem.innerText = stats.nb
    var min_elem = document.getElementById("gstats_minsz")
    min_elem.innerText = stats.min
    var max_elem = document.getElementById("gstats_maxsz")
    max_elem.innerText = stats.max
}