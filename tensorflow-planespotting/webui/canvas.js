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

function is_rect_intersecting(x1, y1, w1, h1, x2, y2, w2, h2) {
    return (x1 <= x2+w2 &&
        x2 <= x1+w1 &&
        y1 <= y2+h2 &&
        y2 <= y1+h1)
}

function grabPixels() {
    // zone to grab
    var zone = document.getElementById('zone')
    // visualisation of the grabbed pixels for control
    var visu = document.getElementById('cap')
    var map = document.getElementById('map')
    resetResults()

    if (map.style.display !== "none")
        grabPixelsFromMap(map, zone, visu)
    else
        grabPixelsFromImage(zone, visu)
}

function grabPixelsFromImage(zone, visu) {
    var imgmap = document.getElementById("imgmap")
    var img = document.querySelector("#imgmap img")
    if (img == null)
        return

    var offscreen = document.createElement('canvas')
    offscreen.width = zone.clientWidth
    offscreen.height = zone.clientHeight

    var ctx = offscreen.getContext('2d')
    var zoomX = img.naturalWidth / img.width
    var zoomY = img.naturalHeight / img.height
    ctx.drawImage(img,
        (zone.offsetLeft + imgmap.scrollLeft) * zoomX,
        (zone.offsetTop + imgmap.scrollTop) * zoomY,
        zone.clientWidth * zoomX,
        zone.clientHeight * zoomY,
        0,
        0,
        zone.clientWidth,
        zone.clientHeight)

    processPixels(offscreen, 0, 0, zone.clientWidth, zone.clientHeight, visu)
}

function grabPixelsFromMap(map, zone, visu) {
    disableMapScroll()
    setMapLocationInURL(googlemap.getCenter(), googlemap.getZoom(), 0)

    // Hack to get the pixels from Google Maps: with som knowledge of the HTML structure
    // of a Google Maps satellite image rendition, it is possible to get to the image tiles directly
    // and download them into a canvas. Google Maps supports CORS (Cross Origin Ressource Sharing) so
    // access to the pixels in a canvas is possible.
    var offscreen = document.createElement('canvas')
    offscreen.width = map.clientWidth
    offscreen.height = map.clientHeight
    var ctx = offscreen.getContext('2d')
    ctx.clearRect(0, 0, offscreen.width, offscreen.height)
    var imgNodes = document.querySelectorAll("#map img")
    var images = []
    imgNodes.forEach(function(nod) {
        if (nod.parentElement != null &&
            nod.parentElement.parentElement != null &&
            nod.parentElement.parentElement.parentElement != null &&
            nod.parentElement.parentElement.parentElement.parentElement != null) {
            var center_offset_x = nod.parentElement.parentElement.parentElement.parentElement.offsetLeft
            var center_offset_y = nod.parentElement.parentElement.parentElement.parentElement.offsetTop
            var img_x = nod.parentElement.offsetLeft
            var img_y = nod.parentElement.offsetTop
            var img_width = 256
            var img_height = 256
            var transformation_matrix = getComputedStyle(nod.parentElement.parentElement).getPropertyValue("transform")
            if (transformation_matrix !== "none") {
                var values = transformation_matrix.substring(0, transformation_matrix.length - 1).split(", ")
                var offset_x = parseInt(values[4])
                var offset_y = parseInt(values[5])
                var map_tile_x = img_x + center_offset_x + offset_x
                var map_tile_y = img_y + center_offset_y + offset_y
                if (is_rect_intersecting(map_tile_x, map_tile_y, img_width, img_height, zone.offsetLeft, zone.offsetTop, zone.clientWidth, zone.clientHeight)) {
                    images.push({x:map_tile_x, y:map_tile_y, src: nod.src})
                }
            }
        }
    })

    var promises = images.map(function (im) {
        return new Promise(function(resolve) {
            var img = new Image()
            img.crossOrigin = ""
            img.onload = function () {
                ctx.drawImage(this, im.x, im.y)
                resolve()
            }
            img.src = im.src
            // Debug: superimpose loaded images on the map
            //img.style = "position:absolute; left:" + (im.x) + "px; top:" + (im.y) + "px; opacity:0.6; border: 0px yellow solid"
            //document.getElementById("zone-container").appendChild(img)
            //return 0
        })
    })

    Promise.all(promises).then(function() {
        processPixels(offscreen, zone.offsetLeft, zone.offsetTop, zone.clientWidth, zone.clientHeight, visu)
        enableMapScroll()
    })
}


function processPixels(canvas, sx, sy, sw, sh, visu) {
    var sctx = canvas.getContext("2d")
    var vctx = visu.getContext("2d")

    // copy from source to destination context
    //var step = tile_size
    var data = sctx.getImageData(sx, sy, sw, sh)
    vctx.putImageData(data, 0, 0)
    // drawimage would always work but the goal here is to check wether access to pixels is possible
    // That is why we use getImageData / putImageData.
    //vctx.drawImage(canvas, 0, 0)

    payload_tiles = []
    // This will always tile with overlapping tiles unless there is s single tile position possible
    // ex: zone width 500, tile width 200: 3 tile positions on x axis
    // ex: zone width 200, tile width 200: 1 tile position
    // ex: zone width 600, tile width 200: 4 tile positions to ensure overlap
    var nx = Math.floor(data.width / tile_size)
    var ny = Math.floor(data.height / tile_size)
    if (nx>=5) nx++ // more overlap for large zones
    if (ny>=5) ny++
    var xstep = (data.width - tile_size) / nx
    var ystep = (data.height - tile_size) / ny
    xstep = xstep > 0 ? xstep : tile_size
    ystep = ystep > 0 ? ystep : tile_size
    var epsilon = 0.0001
    for (var y=0; y+tile_size<=data.height+epsilon; y+=ystep) {
        for (var x=0; x + tile_size <= data.width+epsilon; x+=xstep) {
            var xoffset = Math.floor(x)
            var yoffset = Math.floor(y)
            var tile = sctx.getImageData(sx + xoffset, sy + yoffset, tile_size, tile_size)
            var b64jpegtile = imageCropAndExport(canvas, sx + xoffset, sy + yoffset, tile_size, tile_size)
            var tile = {image_bytes:b64jpegtile, pos:{x:xoffset, y:yoffset, sz:tile_size}}
            payload_tiles.push(tile)
        }
    }
    displayPayload(payload_tiles)
}

function imageCropAndExport(img, x, y, w, h) {
    var offscreen = document.createElement('canvas')
    offscreen.width = w
    offscreen.height = h
    var offcanvas = offscreen.getContext('2d')
    offcanvas.drawImage(img,x,y,w,h,0,0,w,h)
    var txt = offscreen.toDataURL('image/jpeg')

    // var nod = document.getElementById('exported')
    // var imnod = new Image()
    // imnod.src = txt
    // nod.appendChild(imnod)
    // console.info(txt)

    txt = txt.substring("data:image/jpeg;base64,".length)
    return txt
}
