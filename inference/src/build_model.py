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
import shutil
import sys

import tvm
from tvm import relay
import tvm.relay.testing


def main():
    # dshape = (3, 150, 150)
    # batch_size = 1
    # # net, params = relay.testing.mlp.get_workload(batch_size=dshape[0], dtype='float32')
    # # net, params = relay.testing.squeezenet.get_workload(batch_size=dshape[0], dtype='float32')
    # net, params = relay.testing.mobilenet.get_workload(batch_size=batch_size, num_classes=10, image_shape=dshape, dtype='float32')

    # with relay.build_config(opt_level=3):
    #     graph, lib, params = relay.build_module.build(
    #         net, target='llvm --system-lib',  params=params)

    build_dir = osp.abspath(sys.argv[1])
    if not osp.isdir(build_dir):
        os.makedirs(build_dir, exist_ok=True)

    shutil.copyfile('/home/lifabing/sgx/best-partion/model.o', osp.join(build_dir, 'model.o'))
    shutil.copyfile('/home/lifabing/sgx/best-partion/graph.json', osp.join(build_dir, 'graph.json'))
    shutil.copyfile('/home/lifabing/sgx/best-partion/params.bin', osp.join(build_dir, 'params.bin'))


if __name__ == '__main__':
    main()
