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

var auth2

var payload = new Object()
payload.instances = [new Object()]
payload.instances[0].image_bytes = []
payload.instances[0].square_size = 200

// Google maps API map instance
var googlemap = null
var mlengine = null
// https://ml.googleapis.com/v1/projects/cloudml-demo-martin/models/plane/versions/v1:predict

// all button event handlers
var analyzeButton = document.getElementById('analyze-button');
var authorizeButton = document.getElementById('authorize-button');
var signoutButton = document.getElementById('signout-button');
