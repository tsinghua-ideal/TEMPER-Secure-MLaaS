import onnx
import torch
import torch.onnx
import tvm.relay as relay
from thop import profile

import math
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


def isRightChild(index):
    if (index - 1) % 2 == 1:
        return True
    else:
        return False


def getNearPartition(index):
    """

    :param index:
    :return: the layer number of the node.
    """
    partition = [int(math.log2(index + 1))]
    if index == 0:
        return partition
    else:
        index = int((index-1)/2)
        while not isRightChild(index):
            partition.append(int(math.log2(index + 1)))
            index = int((index - 1) / 2)
        return partition


def _torch2onnx(torch_model, input_tensor):
    torch.onnx.export(torch_model, input_tensor, "temp.onnx", verbose=True, input_names=['input'],
                      output_names=['output'])


def _onnx2tvm(input_tensor, onnx_model='temp.onnx', build_dir='./'):
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


def _torch2tvm(torch_model, input_tensor):
    """
    Transform a torch model into a TVM one. This function is preferred.
    :param torch_model:
    :param input_tensor:
    :return:
    """


def _calculate_latency(input_size):
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
    ret = subprocess.run('source /home/lifabing/sgx/best-partion/inference/src/sgx-infer.sh ' + input_size, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, encoding="utf-8", executable="/bin/bash", timeout=1000)
    arr = ret.stdout.split('\n')[0:-1]
    arr = np.array(arr, dtype='int')
    return arr.mean() / 1000


def calculate_latency(torch_model, input_tensor, onnx_model='temp.onnx', build_dir='./'):
    """
    calculate the latency of given model.
    :param torch_model:
    :param input_tensor:
    :param onnx_model:
    :param build_dir:
    :return:
    """
    # import pickle
    # with open('cp.pkl', 'wb') as f:
    #     pickle.dump((torch_model, input_tensor), f)
    _torch2onnx(torch_model, input_tensor)
    _onnx2tvm(input_tensor, onnx_model=onnx_model, build_dir=build_dir)
    shape = input_tensor.shape
    shape = str(shape[0]) + '/' + str(shape[1]) + '/' + str(shape[2]) + '/'+str(shape[3])
    return _calculate_latency(input_size=shape)


class ModelSet:
    def __init__(self, model=None, input_size=None, unit=None, blocks_params=None):
        """
        :param model: The model (An instance of torch.nn.Module)
        :param input_size: The input size of model input
        :param unit: The grain of partition. Default: conv_block, separable_conv_block, BasicBlock
        :param blocks_params: A list of set of (model, input shape, parameter size, latency). Default: null
        """
        if unit is None:
            unit = [conv_block, separable_conv_block, BasicBlock, nn.Dropout, nn.Linear]
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
        :param blocks_params: A list of set of (model, input shape, parameter size, latency). Default: null
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
        bp = []
        for layer in self.model.modules():
            for block in self.unit:
                if isinstance(layer, block):
                    _, total_params = profile(layer, (input_tensor,), verbose=False)
                    bp.append((layer, input_tensor.shape, float(total_params * 4. / (1024 ** 2.))))
                    input_tensor = layer(input_tensor)
                # print(layer)
        self.blocks_params = bp

    def _get_block_latency(self):
        """
        calculate the latency of each block
        :return:
        """
        for i in range(len(self.blocks_params)):
            layer, shape, _ = self.blocks_params[i]
            # _torch2tvm(layer, torch.randn(shape))
            _torch2onnx(layer, torch.randn(shape))
            _onnx2tvm(torch.randn(shape))

            blocks_latency = (_calculate_latency(str(shape[0]) + '/' + str(shape[1]) + '/' + str(shape[2]) + '/'+str(shape[3])),)
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


class Partition:
    """
    We can organize the strategies as an tree. Take the index of block as nodes and each node in the layer has two
    choices: partition or not. We can just travel the tree to find the possible strategies and total latency.
    class Node: index, latency(from last unparted one to the current one.
    """

    def __init__(self, block_params=None, upper_params_size=None):
        """

        :param block_params: A list of set of (model, input shape, parameter size, latency). Default: null
        :param upper_params_size:
        """
        self.block_params = block_params
        self.upper_params_size = upper_params_size
        self.look_up_table = []
        self.search_tree = []
        for i in range(len(block_params)):
            self.look_up_table.append({(i,): block_params[i][3]})

    def lookup(self, key):
        """
        look up the searched path.
        :return:
        """
        for parted in self.look_up_table:
            if key in parted:
                return parted[key]
        return None

    def travel_for_block(self, index):
        """
        travel the tree for a block to fuse
        :param index:
        :return: A fused model
        """
        # To be filled
        nearestPartition = getNearPartition(index)
        nearestPartition.reverse()
        input_shape = self.block_params[nearestPartition[0]][1]
        fused_model = self.block_params[nearestPartition[0]][0]
        for i in range(1, len(nearestPartition)):
            fused_model = nn.Sequential(fused_model, self.block_params[nearestPartition[i]][0])
        return input_shape, fused_model, nearestPartition

    def partition_rule(self, index):
        n_plus_one = int(math.log2(index+1) + 1)
        if len(self.block_params) < n_plus_one:
            return False
        input_shape, model_n, partition = self.travel_for_block(index)
        model_n_plus_one = nn.Sequential(model_n, self.block_params[n_plus_one][0])

        latency_n = self.lookup(tuple(partition))
        import copy
        partition_n_plus_one = copy.copy(partition)
        partition_n_plus_one.append(n_plus_one)
        latency_n_plus_one = self.lookup(tuple(partition_n_plus_one))
        if not latency_n:
            latency_n = calculate_latency(model_n, torch.randn(input_shape))
            self.look_up_table.append({tuple(partition): latency_n})
        if not latency_n_plus_one:
            latency_n_plus_one = calculate_latency(model_n_plus_one, torch.randn(input_shape))
            self.look_up_table.append({tuple(partition_n_plus_one): latency_n_plus_one})
        if latency_n + self.block_params[n_plus_one][-1] > latency_n_plus_one:
            return True
        else:
            return False

    def binary_partition(self):
        """
        Maybe recursive function is helpful for coding. However, how to record the search path?
        Partition rule: if Latency_(1..n)≤Latency_(1..n-1)+ Latency_(n), do partition.
        1) Partition on B1 and B2, then for B1 ,look up the Latency Table and applied the partition rule.
        if there is only one block, calculate it;
        for B2,
            1. calculate total param_size, if is greater than upper_params_size, do partition.
            2. else applied the partition rule.
            3. else stop partition
        :return:
        """
        self.search_tree = np.zeros(2 ** len(self.block_params) - 1)
        self.search_tree[0] = 1
        for i in range(2 ** (len(self.block_params)-1) - 1):
            if self.search_tree[i] == 0:
                continue
            if self.partition_rule(i):
                # add left child
                self.search_tree[2 * i + 1] = 1
            # add right child
            self.search_tree[2 * i + 2] = 1

    def get_strategy(self):
        """
        seems to have some problems on look up the key, so i set the unknown result to 10000
        :return:
        """
        # print(self.look_up_table)
        latency = []
        for i in range(2 ** (len(self.block_params)-1) - 1, 2 ** len(self.block_params) - 1):
            if self.search_tree[i] == 0:
                continue
            else:
                strategy = []
                part = ()
                index = i
                while index is not 0:
                    layer_n = int(math.log2(index + 1))
                    part = part + (layer_n,)
                    if isRightChild(index):
                        strategy.append(part)
                        part = ()
                    if index is 1:
                        strategy[-1] = strategy[-1] + (0,)
                    if index is 2:
                        strategy.append((0,))
                    index = int((index - 1) / 2)
                latency.append(strategy)
        print(latency)
        total_latency = []
        for lt in latency:
            total = 0
            for key in lt:
                key = tuple(reversed(key))
                la = self.lookup(key)
                if la:
                    total += la
                else:
                    total_latency.append(10000)
                    continue
            total_latency.append(total)
        total_latency = np.array(total_latency, dtype='float64')
        result = latency[np.argmin(total_latency)]
        print(result)
        for slice in result:
            slice = tuple(reversed(slice))



if __name__ == '__main__':
    #     shape = [(1, 3, 224, 224), (1, 3, 150, 150), (1, 32, 75, 75)]
    #     input_tensor = torch.randn(shape[1])
    #     # x = torch2onnx(input_tensor, model)
    #     layers_params = stat_layer_params(model, input_tensor)
    #     # for x in layers_params:
    #     #     layer, params = x
    #     print(layers_params)
    # import torchvision.models as models

    model = mobilenet(1000)
    # # model = ResNet1(BasicBlock, 10)
    # # model = nn.Sequential(mobilenet1(), mobilenet2(), mobilenet3())
    ms = ModelSet(model, (1, 3, 224, 224))
    ms.run(70)
    import pickle
    with open('modelset.o', 'wb') as f:
        pickle.dump(ms, f)
    # with open('modelset.o', 'rb') as f:
    #     ms = pickle.load(f)
    pt = Partition(ms.blocks_params)
    del ms
    pt.binary_partition()
    pt.get_strategy()
    with open('partition.o', 'wb') as f:
        pickle.dump(pt, f)

