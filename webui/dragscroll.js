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

var imgmapX = 0
var imgmapY = 0
var imgmapVX = 0
var imgmapVY = 0
var imgmapMoving = false
var dragnode

function decel(v, decay) {
    var pv = v
    v = v*decay
    if (v*v < 0.1)
        v = 0
    return v
}

function makeDragScrollable(nod) {
    dragnode = nod
    nod.addEventListener("scroll", onScroll)
    nod.addEventListener("mousedown", dragscrollOnMouseDown)
    nod.addEventListener("mouseup", dragscrollOnMouseUp)
    nod.addEventListener("mousemove", dragscrollOnMouseMove)
    nod.addEventListener("dragstart", function(e) {e.preventDefault(); return false})
    nod.addEventListener("drop", function(e) {e.preventDefault(); return false})
}

function dragscrollOnMouseDown(e) {
    imgmapMoving = true
    imgmapX = e.clientX
    imgmapY = e.clientY
}

function dragscrollOnMouseUp(e) {
    imgmapMoving = false
    setTimeout(dragscrollInertia, 20)
}

function dragscrollOnMouseMove(e) {
    if (imgmapMoving && e.buttons) {
        var deltaX = e.clientX - imgmapX
        var deltaY = e.clientY - imgmapY
        imgmapVX = imgmapX - e.clientX
        imgmapVY = imgmapY - e.clientY
        imgmapX = e.clientX
        imgmapY = e.clientY
        e.currentTarget.scrollTo(
            e.currentTarget.scrollLeft - deltaX,
            e.currentTarget.scrollTop - deltaY)
    }
}

function dragscrollInertia() {
    if (dragnode !== undefined) {
        imgmapVX = decel(imgmapVX, 0.9)
        imgmapVY = decel(imgmapVY, 0.9)
        dragnode.scrollTo(
            dragnode.scrollLeft + imgmapVX,
            dragnode.scrollTop + imgmapVY)
    }
    if (imgmapVX != 0 || imgmapVY != 0) {
        setTimeout(dragscrollInertia, 20)
    }
}

function zoomIn() {
    zoom(1.2)
}

function zoomOut() {
    zoom(1.0/1.2)
}

function zoom(factor) {
    if (dragnode !== undefined) {
        var imgnod
        dragnode.childNodes.forEach(function (nod) {
            if (nod.nodeName == "IMG")
                imgnod = nod
        })
        if (imgnod !== undefined) {
            var prev_zoomlvl = imgnod.naturalWidth / (imgnod.width)
            var zoomlvl = imgnod.naturalWidth / (imgnod.width * factor)
            // min max zoom levels set here
            if (zoomlvl <= 4 && zoomlvl >= 0.25) {
                imgnod.width = imgnod.width * factor
                // adjust scroll position so as to zoom on center of the screen
                dragnode.scrollLeft = (dragnode.scrollLeft + dragnode.clientWidth / 2) * prev_zoomlvl / zoomlvl - dragnode.clientWidth / 2
                dragnode.scrollTop = (dragnode.scrollTop + dragnode.clientHeight / 2) * prev_zoomlvl / zoomlvl - dragnode.clientHeight / 2
            }
        }
    }
}