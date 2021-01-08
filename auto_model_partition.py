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
    with relay.build_config(opt_level=3):
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
                         executable="/bin/bash", timeout=500)
    while ret.stdout.startswith('Attaching debugger'):
        heap_size += 4
        if heap_size > 50:
            heap_size += 200
        ret = subprocess.run('source /home/lifabing/sgx/best-partion/inference/src/sgx-infer.sh ' + input_size + ' ' +
                             str(heap_size), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             encoding="utf-8",
                             executable="/bin/bash", timeout=500)
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


def size2memory(size):
    return size[0] * size[1] * size[2] * size[3] * 4 / 1024 / 1024


class ModelSet:
    def __init__(self, model=None, input_size=None, unit=None, blocks_params=None, expansion=6, balance_point=45):
        """
        :param model: The model (An instance of torch.nn.Module)
        :param input_size: The input size of model input
        :param unit: The grain of partition. Default: conv_block, separable_conv_block, BasicBlock
        :param blocks_params: A list of set of (model, input shape, output shape, parameter size, latency_origin,
            latency_step, latency_before, latency_after). Default: null        """
        if unit is None:
            unit = [conv_block, separable_conv_block, BasicBlock, Bottleneck, vgg_classifier]
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
        :param blocks_params: A list of set of (model, input shape, output shape, parameter size, latency_origin,
            latency_step, latency_before, latency_after). Default: null
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
                # if isinstance(layer, nn.Embedding):
                #     input_tensor = torch.LongTensor(input_tensor.detach().numpy())
                #     _, total_params = profile(layer, (input_tensor,), verbose=False)
                #     input_shape = input_tensor.shape
                #     input_tensor = layer(input_tensor)
                #     bp.append((layer, input_shape, input_tensor.shape, float(total_params * 4. / (1024 ** 2.))))
                #     continue
                # if isinstance(layer, nn.LSTM):
                #     input_tensor = torch.LongTensor(input_tensor.numpy())
                #     _, total_params = profile(layer, (input_tensor,), verbose=False)
                #     input_shape = input_tensor.shape
                #     input_tensor, _ = layer(input_tensor)
                #     bp.append((layer, input_shape, input_tensor.shape, float(total_params * 4. / (1024 ** 2.))))
                #     continue
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
            layer, shape, _, params = self.blocks_params[i]
            # _torch2tvm(layer, torch.randn(shape))
            _torch2onnx(layer, torch.randn(shape))
            _onnx2tvm(torch.randn(shape))

            if params + size2memory(shape) + 7 > self.balance_point:
                blocks_latency = (
                    _calculate_latency(str(shape[0]) + '/' + str(shape[1]) + '/' + str(shape[2]) + '/' + str(shape[3]),
                                       heap_size=160),)
                self.blocks_params[i] = self.blocks_params[i] + blocks_latency
                self.blocks_params[i] = self.blocks_params[i] + blocks_latency
            else:
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
        # self.blocks_params[0][1] = [0, 0, 0, 0]
        layers_n = len(self.blocks_params)
        latency = [[0 for i in range(layers_n)] for i in range(layers_n)]
        params_table = [[0 for i in range(layers_n)] for i in range(layers_n)]
        IAs_table = [[0 for i in range(layers_n)] for i in range(layers_n)]
        for i in range(0, layers_n):
            latency[i][i] = self.blocks_params[i][4]
            params_table[i][i] = self.blocks_params[i][3]
            IAs_table[i][i] = size2memory(self.blocks_params[i][1])
            for j in range(i+1, layers_n):
                total_params = 0
                IAs = 0
                for idx in range(i, j+1):
                    total_params += self.blocks_params[idx][3]
                    IAs = max(IAs, size2memory(self.blocks_params[idx][1]))
                params_table[i][j] = total_params
                IAs_table[i][j] = IAs
                if total_params + IAs + 7 > self.balance_point:
                    for idx in range(i, j+1):
                        latency[i][j] += self.blocks_params[idx][5]
                else:
                    for idx in range(i, j+1):
                        latency[i][j] += self.blocks_params[idx][4]
        partition_flag = [[0 for i in range(layers_n)] for i in range(layers_n)]
        func = [0 for i in range(layers_n+1)]
        func[1] = latency[0][0]
        for i in range(1, layers_n):
            point_type = 0
            min_func = 9999
            partition_point = -1
            for j in range(0, i+1):
                trans = 15 * size2memory(self.blocks_params[j][1]) if j > 0 else 0
                params = params_table[j][i]
                loading = -0.0004522 * params ** 3 + 0.1028 * params ** 2 + 0.2135 ** params + 3.148 if j > 0 else 0
                loading = 999
                # trans = 999
                if func[j] + latency[j][i] + min(trans, loading) < min_func:
                    min_func = func[j] + latency[j][i] + min(trans, loading)
                    point_type = 1 if trans > loading else 2
                    partition_point = j
                    print(trans)
            func[i+1] = min_func
            print('n={} partition point:{} partition type:{}'.format(i, partition_point, point_type))
            partition_flag[i][partition_point] = point_type
        self.strategy = partition_flag
        partition_flag = np.array(partition_flag)
        latency = np.array(latency)
        l = 0
        for i in range(0, layers_n):
            l += self.blocks_params[i][4]
        print(l)
        print(func[-1])

    def generate_block_model(self, build_dir='model/'):
        for idx, bp in enumerate(self.blocks_params):
            model = bp[0]
            input_size = bp[1]
            _torch2onnx(model, torch.rand(input_size))
            path = osp.join(build_dir, str(idx))
            if not osp.exists(path):
                os.makedirs(path)
            _onnx2tvm(torch.rand(input_size), build_dir=path)

    def generate_model(self, build_dir='model/'):
        index_old = len(self.strategy) - 1
        modelset = []
        while index_old > 0:
            result = [i for i, j in enumerate(self.strategy[index_old]) if j == 1 or j == 2]
            if not result:
                return
            index_new = result[0] - 1
            model = self.blocks_params[index_new+1][0]
            input_size = self.blocks_params[index_new+1][1]
            for i in range(index_new+2, index_old+1):
                model = nn.Sequential(model, self.blocks_params[i][0])
            modelset.append((model, input_size))
            index_old = index_new
            # print(index_new)
        idx = 0
        trans = 0
        for (model, input_size) in reversed(modelset):
            path = osp.join(build_dir, str(idx))
            if not osp.exists(path):
                os.makedirs(path)
            trans += 15 * size2memory(input_size)
            _torch2onnx(model, torch.rand(input_size))
            _onnx2tvm(torch.rand(input_size), build_dir=path)
            print('Writing model into ' + path)
            idx += 1
        print('transmission latency', trans)
            # _onnx2tvm(torch.rand(input_size), build_dir='./')
            # print('Block latency: ', _calculate_latency(str(input_size[0]) + '/' + str(input_size[1]) + '/' +
            #                                             str(input_size[2]) + '/' + str(input_size[3])))

    def run(self):
        """
        find the strategies of partition
        :return: An generator of strategies of partitions
        """
        if not self.blocks_params:
            self._stat_layer_params()
            import time
            start = time.time()
            self._get_block_latency()
            end = time.time()
            print(str((end - start) * 1000) + 'ms')
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

    # do a new partition
    # model = mobilenet(1000)
    # # # model = ResNet18(1000)
    # # # model = ResNet1(BasicBlock, 10)
    # # # model = nn.Sequential(mobilenet1(), mobilenet2(), mobilenet3())
    # import self_defined_nn
    model = self_defined_nn.get_vgg('E', False)
    # import torchvision.models as models
    # # model = models.vgg16(pretrained=False)
    # _, total_params = profile(model, (torch.rand((1, 3, 224, 224)),), verbose=False)
    # print("%s | %.3f MB" % ('model', float(total_params * 4. / (1024 ** 2.))))
    # total_ops, total_params = profile(model, (torch.randn((1, 3, 224, 224)),), verbose=False)
    # print("%s | %.3f MB | %.3fG GFLOPs" % ('model', float(total_params * 4. / (1024 ** 2.)), total_ops / (1000 ** 3)))
    # import torchvision.models as models
    # model = models.segmentation.DeepLabV3(pretrained=False)
    # from pytorch_transformers import GPT2Tokenizer, GPT2Model
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # model = GPT2Model.from_pretrained('gpt2')
    # input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
    # outputs = model(input_ids)
    # last_hidden_states = outputs[0]
    # with open('bert.o', 'wb') as f:
    #     pickle.dump(model, f)
    # densenet169
    # model = DenseNet(32, (6, 12, 32, 32), 69)
    # densenet201
    # model = DenseNet(32, (6, 12, 48, 32), 64)
    # inception_v3

    # gnmt
    # from sample import gnmt, LSTMTagger
    # model = gnmt(10, hidden_size=10, num_layers=4, dropout=0.2)
    # # model = LSTMTagger(6, 6, 9, 3)
    # # input_size = (1, 1000, 1000)
    # # _torch2onnx(model, torch.LongTensor(torch.randint(1, 10, size=(1, 1024))))
    # # _onnx2tvm(torch.LongTensor(torch.randint(1, 10, size=(1, 1024))), build_dir='./')
    # # print('Block latency: ', _calculate_latency(str(input_size[0]) + '/' + str(input_size[1]), 0x400))
    # # _, total_params = profile(model, (torch.LongTensor(torch.randint(1, 10, size=(1, 1024))),), verbose=False)
    # # model = ResNet(Bottleneck, [3, 30, 48, 8])
    # input_size = (1, 3, 224, 224)
    # _torch2onnx(model, torch.randn((1, 3, 224, 224)))
    # _onnx2tvm(torch.randn((1, 3, 224, 224)), build_dir='./')
    # print('Block latency: ', _calculate_latency(str(input_size[0]) + '/' + str(input_size[1]) + '/' +
    #                                             str(input_size[2]) + '/' + str(input_size[3]), 0x30))
    # # ms = ModelSet(model, (1, 3, 224, 224), unit=[DenseBlock, Transition, conv_block, Dense_Classifier])
    # ms = ModelSet(model, (1, 1024), unit=[nn.LSTM, nn.Embedding, nn.Linear, nn.Dropout])
    ms = ModelSet(model, (1, 3, 224, 224), unit=[conv_block, vgg_classifier, nn.MaxPool2d])
    ms.run()
    with open('vgg19-dp-mul.o', 'wb') as f:
        pickle.dump(ms, f)

    # look up for an old partition
    # with open('/home/lifabing/sgx/best-partion/vgg16-dp-mul.o', 'rb') as f:
    #     ms = pickle.load(f)
    #     ms.partition()
    #     # ms.generate_model()
    #     path = '/home/lifabing/sgx/re-implementation/vessels/model/vgg16'
    #     # path = '/home/lifabing/sgx/cluster-inference/model/vgg16'
    #     ms.generate_model(path)
    #     big = 0
    #     for ipt in ms.blocks_params:
    #         if big < size2memory(ipt[1]):
    #             big = size2memory(ipt[1])
    #     print(big)
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
