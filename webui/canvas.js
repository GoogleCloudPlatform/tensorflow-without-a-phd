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
    var zone = document.getElementById('zone')
    var playground = document.getElementById('tap')
    var visu = document.getElementById('cap')
    html2canvas(map, {
        onrendered: function(canvas) {processPixels(canvas, zone.offsetLeft, zone.offsetTop, zone.clientWidth, zone.clientHeight, visu, playground)},
        width: map.clientWidth,
        height: map.clientHeight,
        useCORS: true
    })

    // reset previous results
}

function processPixels(canvas, sx, sy, sw, sh, visu, playground) {

    resetResults()

    var sctx = canvas.getContext("2d")
    var vctx = visu.getContext("2d")
    //var dctx = playground.getContext("2d")

    // TODO: currently generating one instance with multiple images in it. Fix it.

    // copy from source to destination context
    var sz = 200    //
    var step = 200 // just one tile
    var data = sctx.getImageData(sx, sy, sw, sh)
    vctx.putImageData(data, 0, 0)

    // hack: if grab fails, this will be the browser's default background color
    // forcing a reload usually makes the gab work again
    if (hasBackgroundInAnyCorner(data))
        location.reload()

    payload.instances[0].image_bytes = []
    for (var y=0,dy=0; y+sz<=data.height; y+=step,dy+=sz+1) {
        for (var x = 0, dx = 0; x + sz <= data.width; x += step, dx += sz + 1) {
            var tile = sctx.getImageData(sx + Math.floor(x), sy + Math.floor(y), sz, sz)
            var jpegtile = imageCropAndExport(canvas, sx + Math.floor(x), sy + Math.floor(y), sz, sz)
            //dctx.drawImage(canvas, x, y, sz, sz, dx, dy, sz, sz)
            //dctx.putImageData(tile, dx, dy)
            payload.instances[0].image_bytes.push(b64Data2MLEngineFormat(jpegtile))
        }
    }
    displayPayload(payload)
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
    return is_background_pix(0,0) || is_background_pix(w-1, 0) || is_background_pix(0, h-1) || is_background_pix(w-1, h-1)
}

function b64Data2MLEngineFormat(b64) {
    // ? URL decode ?
    var container = new Object()
    container.b64 = b64
    return container
}