# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# If you want to run training with tf-nightly on ML Engine
# (or any other version not yet accessible through --runtime-version)
# the config script is here. Rename this file from setup.tf-nightly.py
# to setup.py and run your usual gcloud ml-engine command. ML Engine
# will pip-install your requirements before starting your training code.

import setuptools

setuptools.setup(
    install_requires=['tf-nightly-gpu'],
    # for a specific version of tensorflow
    #install_requires=['tensorflow-gpu>=1.10'],
    packages=setuptools.find_packages(),
    include_package_data=True
)