#!/usr/bin/python3

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Creates a simple TVM modules."""

import os
from os import path as osp
import json
import shutil
import sys

import tvm
from tvm import relay
import tvm.relay.testing


def main():
    build_dir = osp.abspath(sys.argv[1])
    if not osp.isdir(build_dir):
        os.makedirs(build_dir, exist_ok=True)
    with open(osp.join(os.getcwd(), 'config'), 'r') as f:
        config = json.load(f)["model_path"]
    shutil.copyfile(osp.join(config, 'model.o'), osp.join(build_dir, 'model.o'))
    shutil.copyfile(osp.join(config, 'graph.json'), osp.join(build_dir, 'graph.json'))
    shutil.copyfile(osp.join(config, 'params.bin'), osp.join(build_dir, 'params.bin'))


if __name__ == '__main__':
    main()
