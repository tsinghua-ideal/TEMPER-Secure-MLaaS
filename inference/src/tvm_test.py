#!/usr/bin/python3

import os
from os import path as osp
import sys

import tvm
from tvm import relay
import tvm.relay.testing
import numpy as np
from tvm.contrib import graph_runtime
import time


def main():
    dshape = (3, 32, 32)
    batch_size = 2

    # net, params = relay.testing.mlp.get_workload(batch_size=dshape[0], dtype='float32')
    # net, params = relay.testing.squeezenet.get_workload(batch_size=dshape[0], dtype='float32')
    net, params = relay.testing.mobilenet.get_workload(batch_size=batch_size, num_classes=10, image_shape=dshape, dtype='float32')

    data = tvm.nd.array(np.random.rand(dshape[0], dshape[1], dshape[2]))
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build_module.build(net, target="llvm --system-lib",  params=params)
        ctx = tvm.cpu(0)

        executor = graph_runtime.create(graph, lib, ctx)
        executor.set_input(**params)

        start = time.time()
        executor.set_input('data', data)
        executor.run()
        output = executor.get_output(0)

        end = time.time()
        top1_tvm = np.argmax(output.asnumpy()[0])
        
        print(str((end-start)*1000) + 'ms')


if __name__ == '__main__':
    main()