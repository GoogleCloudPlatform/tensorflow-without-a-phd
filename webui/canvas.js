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

function grabPixels() {
    var map = document.getElementById('map')
    var imgmap = document.getElementById('imgmap')
    var zone = document.getElementById('zone')
    var visu = document.getElementById('cap')

    var node_to_grab = map
    if (map.style.display == "none")
        node_to_grab = imgmap

    resetResults()
    disableMapScroll()

    html2canvas(node_to_grab, {
        onrendered: function(canvas) {processPixels(canvas, zone.offsetLeft, zone.offsetTop, zone.clientWidth, zone.clientHeight, visu)},
        width: node_to_grab.clientWidth,
        height: node_to_grab.clientHeight,
        useCORS: true
    })
}

function processPixels(canvas, sx, sy, sw, sh, visu) {
    var sctx = canvas.getContext("2d")
    var vctx = visu.getContext("2d")

    // copy from source to destination context
    var sz = tile_size
    //var step = tile_size
    var data = sctx.getImageData(sx, sy, sw, sh)
    vctx.putImageData(data, 0, 0)

    // hack: if grab fails, this will be the browser's default background color
    // forcing a reload usually makes the grab work again
    if (hasBackgroundInAnyCorner(data)) {
        var query = new URLSearchParams(window.location.search)
        var rlonce = query.get("r")
        if (rlonce != "1") {
            setMapLocationInURL(googlemap.getCenter(), googlemap.getZoom(), 1)
            setTimeout(grabPixels, 100)  // it works better to retry the grab
            //location.reload()         // rather than reload everyting...
            return
        }
    }

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
    for (var y=0; y+sz<=data.height+epsilon; y+=ystep) {
        for (var x=0; x + sz <= data.width+epsilon; x+=xstep) {
            var xoffset = Math.floor(x)
            var yoffset = Math.floor(y)
            var tile = sctx.getImageData(sx + xoffset, sy + yoffset, sz, sz)
            var b64jpegtile = imageCropAndExport(canvas, sx + xoffset, sy + yoffset, sz, sz)
            var tile = {image_bytes:b64jpegtile, pos:{x:xoffset, y:yoffset, sz:sz}}
            payload_tiles.push(tile)
        }
    }
    setMapLocationInURL(googlemap.getCenter(), googlemap.getZoom(), 0)
    displayPayload(payload_tiles)
    enableMapScroll()
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

function hasBackgroundInAnyCorner(imgdata) {
    var data = imgdata.data
    var w = imgdata.width
    var h = imgdata.height
    var r=229, g=227, b=223
    function is_background_pix(x, y) {return data[(y*w+x)*4]==r && data[(y*w+x)*4+1]==g && data[(y*w+x)*4+2]==b}
    a = is_background_pix(0,0)
    return is_background_pix(0,0) || is_background_pix(w-1, 0) || is_background_pix(0, h-1) || is_background_pix(w-1, h-1)
}
