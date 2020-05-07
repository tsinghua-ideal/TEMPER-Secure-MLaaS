import onnx
import torch
import torch.onnx
import tvm.relay as relay
from thop import profile

import numpy as np
from self_defined_nn import *

import os
from os import path as osp
import subprocess


class Node:
    def __init__(self, index, latency=None):
        self.index = index
        self.latency = latency


class BinaryTree:
    def __init__(self, rootObj):
        self.key = rootObj
        self.leftChild = None
        self.rightChild = None

    def insertLeft(self, newNode):
        if self.leftChild is None:
            self.leftChild = BinaryTree(newNode)
        else:
            t = BinaryTree(newNode)
            t.leftChild = self.leftChild
            self.leftChild = t

    def insertRight(self, newNode):
        if self.rightChild is None:
            self.rightChild = BinaryTree(newNode)
        else:
            t = BinaryTree(newNode)
            t.rightChild = self.rightChild
            self.rightChild = t

    def getRightChild(self):
        return self.rightChild

    def getLeftChild(self):
        return self.leftChild

    def setRootVal(self, obj):
        self.key = obj

    def getRootVal(self):
        return self.key


class ModelSet:
    def __init__(self, model=None, input_size=None, unit=None, blocks_params=None):
        """
        :param model: The model (An instance of torch.nn.Module)
        :param input_size: The input size of model input
        :param unit: The grain of partition. Default: conv_block, separable_conv_block, BasicBlock
        :param blocks_params: A list of set of (model, parameter size). Default: null
        """
        if unit is None:
            unit = [conv_block, separable_conv_block, BasicBlock]
        if blocks_params is None:
            blocks_params = []
        self.unit = unit
        self.model = model
        self.blocks_params = blocks_params
        self.input_size = input_size

    def reinit(self, model, input_size, unit, blocks_params=None):
        """
        Call it if you want initialize ModelSet manually or change it
        :param model: The model (An instance of torch.nn.Module)
        :param input_size: The input size of model input
        :param unit: The grain of partition. Default: conv_block, separable_conv_block, BasicBlock
        :param blocks_params: A list of set of (model, parameter size). Default: null
        :return: True if no exception
        """
        self.model = model
        self.input_size = input_size
        if unit is None:
            unit = [conv_block, separable_conv_block, BasicBlock]
        self.unit = unit
        if blocks_params is None:
            blocks_params = []
        self.blocks_params = blocks_params
        return True

    def _stat_layer_params(self):
        """
        The first step of calculation, calculate the parameter size of the given model
        """
        input_tensor = torch.rand(self.input_size)
        for layer in self.model.modules():
            for block in self.unit:
                if isinstance(layer, block):
                    _, total_params = profile(layer, (input_tensor,), verbose=False)
                    self.blocks_params.append((layer, input_tensor.shape, float(total_params * 4. / (1024 ** 2.))))
                    input_tensor = layer(input_tensor)
                # print(layer)

    def _torch2onnx(self, torch_model, input_tensor):
        torch.onnx.export(torch_model, input_tensor, "temp.onnx", verbose=True, input_names=['input'],
                          output_names=['output'])

    def _onnx2tvm(self, input_tensor, onnx_model='temp.onnx', build_dir='./'):
        """
        compile and optimize the onnx model into TVM model.
        :param input_tensor:
        :param onnx_model:
        :param build_dir:
        """
        onnx_model = onnx.load(onnx_model)
        target = 'llvm --system-lib'

        input_name = 'input'
        shape_dict = {input_name: input_tensor.shape}
        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
        with relay.build_config(opt_level=4):
            graph, lib, params = relay.build_module.build(
                mod, target=target, params=params)

        if not osp.isdir(build_dir):
            os.makedirs(build_dir, exist_ok=True)

        lib.save(osp.join(build_dir, 'model.o'))
        with open(osp.join(build_dir, 'graph.json'), 'w') as f_graph_json:
            f_graph_json.write(graph)
        with open(osp.join(build_dir, 'params.bin'), 'wb') as f_params:
            f_params.write(relay.save_param_dict(params))

    def _torch2tvm(self, torch_model, input_tensor):
        """
        Transform a torch model into a TVM one. This function is preferred.
        :param torch_model:
        :param input_tensor:
        :return:
        """

    def _calculate_latency(self):
        """
        Get the output of RUST application by Process Module. A external bash can be applied:
        #!/bin/bash
        cargo build
        path=(the path to executable file
        for i in 0..50;
        do
            ftxsgx-runner ${path}
        done;
        :return: the latency
        """
        # note that dynamic path is preferred. Revise sgx-infer.sh to do this.
        ret = subprocess.run('source /home/lifabing/Documents/sgx-infer.sh', shell=True, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, encoding="utf-8", executable="/bin/bash", timeout=1000)
        arr = ret.stdout.split('\n')[0:-1]
        arr = np.array(arr, dtype='int')
        return arr.mean() / 1000

    def _get_block_latency(self):
        """
        calculate the latency of each block
        :return:
        """
        for i in range(len(self.blocks_params)):
            layer, shape, _ = self.blocks_params[i]
            self._torch2tvm(layer, torch.randn(shape))

            blocks_latency = (self._calculate_latency(),)
            self.blocks_params[i] = self.blocks_params[i] + blocks_latency

    def get_block_params(self):
        return self.blocks_params

    def partition(self, upper_params_size):
        strategies = []
        return strategies

    def run(self, upper_params_size):
        """
        find the strategies of partition
        :param upper_params_size: The maximum size of parameter size
        :return: An generator of strategies of partitions
        """
        if not self.blocks_params:
            self._stat_layer_params()
            self._get_block_latency()
        possible_partions = []


class Partition:
    """
    We can organize the strategies as an tree. Take the index of block as nodes and each node in the layer has two
    choices: partition or not. We can just travel the tree to find the possible strategies and total latency.
    class Node: index, latency(from last unparted one to the current one.
    """

    def __init__(self, block_params=None, upper_params_size=None):
        self.get_block_params = block_params
        self.upper_params_size = upper_params_size
        self.strategy = []

    def lookup(self, key):
        """
        look up the searched path.
        :return:
        """
        for parted in self.strategy:
            if key in parted:
                return parted[key]

    def binary_partition(self):
        """
        Maybe recursive function is helpful for coding. So, how to record the search path?
        Partition rule: if Latency_(1..n)≤Latency_(1..n-1)+ Latency_(n), do partition.
        1) Partition on B1 and B2, then for B1 ,look up the Latency Table and applied the partition rule.
        if there is only one block, calculate it;
        for B2,
            1. calculate total param_size, if is greater than upper_params_size, do partition.
            2. else applied the partition rule.
            3. else stop partition
        :return:
        """
        for i in range(len(self.get_block_params)):
            self.binary_partition()


if __name__ == '__main__':
    #     shape = [(1, 3, 224, 224), (1, 3, 150, 150), (1, 32, 75, 75)]
    #     input_tensor = torch.randn(shape[1])
    #     # x = torch2onnx(input_tensor, model)
    #     layers_params = stat_layer_params(model, input_tensor)
    #     # for x in layers_params:
    #     #     layer, params = x
    #     print(layers_params)
    import torchvision.models as models

    model = models.mobilenet_v2()
    ModelSet._torch2onnx(ModelSet(), model, torch.randn(1, 3, 150, 150))
    ModelSet._onnx2tvm(ModelSet(), torch.randn(1, 3, 150, 150))
