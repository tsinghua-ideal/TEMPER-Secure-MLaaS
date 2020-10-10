import onnx
import torch
import torch.onnx
import tvm.relay as relay
from thop import profile

import math
import numpy as np
from torchvision.models.resnet import Bottleneck

from self_defined_nn import *

import os
from os import path as osp
import subprocess
import pickle


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
        index = int((index - 1) / 2)
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


def _calculate_latency(input_size, heap_size=0x40):
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
    ret = subprocess.run('source /home/lifabing/sgx/best-partion/inference/src/sgx-infer.sh ' + input_size + ' ' +
                         str(heap_size), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8",
                         executable="/bin/bash", timeout=1000)
    while ret.stdout.startswith('Attaching debugger'):
        heap_size += 4
        ret = subprocess.run('source /home/lifabing/sgx/best-partion/inference/src/sgx-infer.sh ' + input_size + ' ' +
                             str(heap_size), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             encoding="utf-8",
                             executable="/bin/bash", timeout=1000)
    arr = ret.stdout.split('\n')[0:-1]
    arr = np.array(arr, dtype='int')
    return arr.mean() / 1000


def calculate_latency(torch_model, input_tensor, heap_size=0x40, onnx_model='temp.onnx', build_dir='./'):
    """
    calculate the latency of given model.
    :param heap_size:
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
    shape = str(shape[0]) + '/' + str(shape[1]) + '/' + str(shape[2]) + '/' + str(shape[3])
    return _calculate_latency(input_size=shape, heap_size=heap_size)


class ModelSet:
    def __init__(self, model=None, input_size=None, unit=None, blocks_params=None, expansion=6, balance_point=42):
        """
        :param model: The model (An instance of torch.nn.Module)
        :param input_size: The input size of model input
        :param unit: The grain of partition. Default: conv_block, separable_conv_block, BasicBlock
        :param blocks_params: A list of set of (model, input shape, output shape, parameter size, latency_origin, latency_step). Default: null
        """
        if unit is None:
            unit = [conv_block, separable_conv_block, BasicBlock, Bottleneck]
        if blocks_params is None:
            blocks_params = []
        self.unit = unit
        self.model = model
        self.input_size = input_size
        self.blocks_params = blocks_params
        self.expansion = expansion
        self.balance_point = balance_point
        self.strategy = None

    def reinit(self, model, input_size, unit, blocks_params=None):
        """
        Call it if you want initialize ModelSet manually or change it
        :param model: The model (An instance of torch.nn.Module)
        :param input_size: The input size of model input
        :param unit: The grain of partition. Default: conv_block, separable_conv_block, BasicBlock
        :param blocks_params: A list of set of (model, input shape, output shape, parameter size, latency_origin, latency_step). Default: null
        :return: True if no exception
        """
        self.model = model
        self.input_size = input_size
        if unit is None:
            unit = [conv_block, separable_conv_block, BasicBlock, Classifier]
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
                # if isinstance(layer, Classifier):
                #     _, total_params = profile(layer, (input_tensor,), verbose=False)
                #     bp.append((layer, input_tensor.shape, float(total_params * 4. / (1024 ** 2.))))
                #     input_tensor = layer(input_tensor)
                if isinstance(layer, block):
                    _, total_params = profile(layer, (input_tensor,), verbose=False)
                    input_shape = input_tensor.shape
                    input_tensor = layer(input_tensor)
                    bp.append((layer, input_shape, input_tensor.shape, float(total_params * 4. / (1024 ** 2.))))
                # print(layer)
        self.blocks_params = bp

    def _get_block_latency(self):
        """
        calculate the latency of each block
        :return:
        """
        for i in range(len(self.blocks_params)):
            layer, shape, _, _ = self.blocks_params[i]
            # _torch2tvm(layer, torch.randn(shape))
            _torch2onnx(layer, torch.randn(shape))
            _onnx2tvm(torch.randn(shape))

            blocks_latency = (
                _calculate_latency(str(shape[0]) + '/' + str(shape[1]) + '/' + str(shape[2]) + '/' + str(shape[3]),
                                   heap_size=self.balance_point),)
            self.blocks_params[i] = self.blocks_params[i] + blocks_latency
            blocks_latency = (
                _calculate_latency(str(shape[0]) + '/' + str(shape[1]) + '/' + str(shape[2]) + '/' + str(shape[3]),
                                   heap_size=160),)
            self.blocks_params[i] = self.blocks_params[i] + blocks_latency

    def get_block_params(self):
        return self.blocks_params

    def partition(self):
        """
        Dynamic programming is applied to this problem now.
        state transition equation:
            f(n)=min{f(n−1)+L(n)+T(n),f(0)+L(0⋅⋯n)+T(0),f(1)+L(1⋅⋯n)+T(1),…f(n−1)+L(n−1⋅⋯n)+T(n−1)}
        The L function represents the actual running time, and the T function represents the transmission overhead
        partition_flag: record the partition point and judge whether transition or loading parameters
        policy:  record the policy of n
        :return:
        """
        layers_n = len(self.blocks_params)
        latency = [[-1 for i in range(layers_n)] for i in range(layers_n)]
        for i in range(0, layers_n):
            for j in range(i+1, layers_n):
                total_params = 0
                for idx in range(i, j):
                    total_params += self.blocks_params[idx][3]
                if total_params > 42:
                    for idx in range(i, j):
                        latency[i][j] += self.blocks_params[idx][5]
                else:
                    for idx in range(i, j):
                        latency[i][j] += self.blocks_params[idx][4]
        partition_flag = [[0 for i in range(layers_n)] for i in range(layers_n)]
        func = [-1 for i in range(layers_n)]
        func[0] = latency[0][1]
        type = 0
        for i in range(1, layers_n):
            size = self.blocks_params[i][1]
            params = self.blocks_params[i][3]
            trans = size[0] * size[1] * size[2] * size[3] * 4 / 1024 / 1024
            loading = -5.335e-13*params**3 + 1.213e-08*params**2 + 0.0006457**params + 1.56
            min_func = func[0] + latency[i][i+1] + min(trans, loading)
            type = 1 if trans > loading else 2
            partition_point = i 
            for j in range(0, i):
                size = self.blocks_params[i][1]
                params = self.blocks_params[i][3]
                trans = size[0] * size[1] * size[2] * size[3] * 4 / 1024 / 1024
                loading = -5.335e-13 * params ** 3 + 1.213e-08 * params ** 2 + 0.0006457 ** params + 1.56
                if func[j] + latency[j][i] + min(trans, loading) < min_func:
                    min_func = func[j] + latency[j][i] + min(trans, loading)
                    type = 1 if trans > loading else 2
                    partition_point = j
            partition_flag[i][partition_point] = type
        print(partition_flag)
            
    def generate_model(self, build_dir='model/'):
        idx = 0
        for stg in self.strategy:
            model = self.blocks_params[stg[0]][0]
            input_size = self.blocks_params[stg[0]][1]
            if len(stg) > 1:
                for index in range(1, len(stg)):
                    model = nn.Sequential(model, self.blocks_params[stg[index]][0])
            _torch2onnx(model, torch.rand(input_size))
            path = osp.join(build_dir, 'part' + str(idx))
            if not osp.exists(path):
                os.makedirs(path)
            _onnx2tvm(torch.rand(input_size), build_dir=path)

            _onnx2tvm(torch.rand(input_size), build_dir='./')
            print('Block latency: ', _calculate_latency(str(input_size[0]) + '/' + str(input_size[1]) + '/' +
                                                        str(input_size[2]) + '/' + str(input_size[3])))
            print('Writing model into ' + path)
            idx += 1

    def run(self):
        """
        find the strategies of partition
        :return: An generator of strategies of partitions
        """
        if not self.blocks_params:
            self._stat_layer_params()
            self._get_block_latency()
            print('partition')
            self.partition()


if __name__ == '__main__':
    #     shape = [(1, 3, 224, 224), (1, 3, 150, 150), (1, 32, 75, 75)]
    #     input_tensor = torch.randn(shape[1])
    #     # x = torch2onnx(input_tensor, model)
    #     layers_params = stat_layer_params(model, input_tensor)
    #     # for x in layers_params:
    #     #     layer, params = x
    #     print(layers_params)
    # import torchvision.models as models

    # do a new partition
    # model = mobilenet(1000)
    # model = ResNet18(1000)
    # model = ResNet1(BasicBlock, 10)
    # model = nn.Sequential(mobilenet1(), mobilenet2(), mobilenet3())
    # import torchvision.models as models
    # model = models.resnet50(pretrained=False)
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    ms = ModelSet(model, (1, 3, 224, 224))
    ms.run()
    with open('modelset.o', 'wb') as f:
        pickle.dump(ms, f)

    # look up for an old partition
    # with open('mobilenetv1-1.o', 'rb') as f:
    #     ms = pickle.load(f)
    #     # ms.expansion = 12
    #     s = []
    #     for i in ms.strategy:
    #         if i not in s:
    #             s.append(i)
    #     ms.strategy = s
    #     print(ms.strategy)
    #     ms.generate_model('model/mobilenetv1')
    # _torch2onnx(ms.block_params[0][0], torch.rand(1, 3, 224, 224))
    # _onnx2tvm(torch.rand(1, 3, 224, 224), build_dir='model/part0/')
