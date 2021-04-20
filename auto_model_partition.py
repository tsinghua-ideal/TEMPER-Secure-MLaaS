import random
import time

import onnx
import torch
import torch.onnx
import tvm.relay as relay
from thop import profile

import math
import numpy as np
from torchvision.models.resnet import Bottleneck

import networkx as nx
import json
import matplotlib.pyplot as plt

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


def _calculate_latency(input_size, heap_size=0x28):
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
    from functools import reduce
    ln = reduce(lambda x, y: x * y, size)
    return ln * 4 / 1024 / 1024


def op_extract(op):
    op = op.split('_')
    op = op[1:-1]
    return '_'.join(op)


class ModelSet:
    def __init__(self, model=None, input_size=None, unit=None, blocks_params=None, expansion=6, balance_point=45,
                 graph_path='graph.json'):
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
        self.graph_path = graph_path
        self.topo = None
        # self._generate_topo()

    def _generate_topo(self):
        with open(self.graph_path, 'r') as f:
            graph = json.load(f)
            topo = nx.MultiDiGraph()
            for i, op in enumerate(graph['nodes']):
                if op['op'] == 'tvm_op':
                    topo.add_node(i, name=op_extract(op['name']))
                    for ipt in op['inputs']:
                        if ipt[0] in topo:
                            topo.add_edge(ipt[0], i)
            self.topo = topo
            # nx.draw(topo)
            # plt.show()

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
        topo = nx.MultiDiGraph()
        input_tensor = None
        for layer in self.model.modules():
            for block in self.unit:
                if isinstance(layer, block):
                    if input_tensor is None:
                        input_tensor = torch.rand(self.input_size)
                        input_shape = [self.input_size]
                    else:
                        input_shape = []
                        for idx in layer.input_nodes:
                            input_shape.append(topo.nodes[idx]['output_shape'])

                        input_tensor = torch.rand(input_shape[0])
                        find_add = list(list(layer.children())[0].children())[0]
                        if isinstance(find_add, add):
                            input_tensor = [input_tensor]
                            for idx in range(1, len(input_shape)):
                                input_tensor.append(torch.rand(input_shape[idx]))
                        else:
                            for idx in range(1, len(layer.input_nodes)):
                                input_tensor = torch.cat([input_tensor, torch.rand(input_shape[idx])], 1)
                    import copy
                    it = copy.copy(input_tensor)
                    _, params = profile(layer, (it,), verbose=False)
                    input_tensor = layer(input_tensor)

                    topo.add_node(layer.node, model=layer, input_shape=input_shape, output_shape=input_tensor.shape,
                                  params=float(params * 4. / (1024 ** 2.)))
                    for idx in layer.input_nodes:
                        topo.add_edge(idx, layer.node)
                # print(layer)
        self.topo = topo

    def _get_block_latency(self):
        """
        calculate the latency of each block
        :return:
        """
        for n, nbrs in self.topo.adjacency():
            block = self.topo.nodes[n]['model']
            shape = self.topo.nodes[n]['input_shape']
            params = self.topo.nodes[n]['params']
            find_add = list(list(block.children())[0].children())[0]
            if isinstance(find_add, add):
                self.topo.add_node(n, normal_latency=1, abnormal_latency=1)
            else:
                shape = self.topo.nodes[n]['input_shape'][0]
                _torch2onnx(block, torch.randn(shape))
                _onnx2tvm(torch.randn(shape))
                if params + size2memory(shape) + 7 > self.balance_point:
                    blocks_latency = _calculate_latency(
                        str(shape[0]) + '/' + str(shape[1]) + '/' + str(shape[2]) + '/' + str(shape[3]),
                        heap_size=320)
                    self.topo.add_node(n, normal_latency=blocks_latency, abnormal_latency=blocks_latency)
                else:
                    blocks_latency = _calculate_latency(
                        str(shape[0]) + '/' + str(shape[1]) + '/' + str(shape[2]) + '/' + str(shape[3]),
                        heap_size=self.balance_point)
                    self.topo.add_node(n, normal_latency=blocks_latency)
                    blocks_latency = _calculate_latency(
                        str(shape[0]) + '/' + str(shape[1]) + '/' + str(shape[2]) + '/' + str(shape[3]),
                        heap_size=160)
                    self.topo.add_node(n, abnormal_latency=blocks_latency)

    def get_block_params(self):
        return self.blocks_params

    def partition(self, smart_flag=False):
        """
        Dynamic programming is applied to this problem now.
        state transition equation:
            f(n)=min{f(n−1)+L(n)+T(n),f(0)+L(0⋅⋯n)+T(0),f(1)+L(1⋅⋯n)+T(1),…f(n−1)+L(n−1⋅⋯n)+T(n−1)}
        The L function represents the actual running time, and the T function represents the transmission overhead
        partition_flag: record the partition point and judge whether transition or loading parameters
        policy:  record the policy of n
        :return:
        """
        num_nodes = len(self.topo.nodes)
        cost = [[-1 for i in range(num_nodes)] for i in range(num_nodes)]
        abnormal_cost = [[-1 for i in range(num_nodes)] for i in range(num_nodes)]
        normal_cost = [[-1 for i in range(num_nodes)] for i in range(num_nodes)]
        total_params = [[0 for i in range(num_nodes)] for i in range(num_nodes)]
        for n in range(num_nodes - 1):
            print(n)
            for j in range(n + 1, num_nodes):
                child = self.topo.nodes[j]
                fcost = 0
                fParams = 0
                fAbnormal = 0
                fNormal = 0
                for father in child['model'].input_nodes:
                    Params = child['params']
                    Abnormal = child['abnormal_latency']
                    Normal = child['normal_latency']
                    IAs = size2memory(child['output_shape'])
                    while father is not 0 or father is not n:
                        if size2memory(self.topo.nodes[father]['output_shape']) > IAs:
                            IAs = size2memory(self.topo.nodes[father]['output_shape'])
                        if isinstance(list(list(self.topo.nodes[father]['model'].children())[0].children())[0], add):
                            if Params + total_params[n][father] + 7 + IAs > self.balance_point:
                                Abnormal = Abnormal + abnormal_cost[n][father]
                                Normal = Abnormal + abnormal_cost[n][father]
                            else:
                                Abnormal = Abnormal + abnormal_cost[n][father]
                                Normal = Normal + normal_cost[n][father]
                            Params = Params + total_params[n][father]
                            break
                        Params += self.topo.nodes[father]['params']
                        Abnormal += self.topo.nodes[father]['abnormal_latency']
                        Normal += self.topo.nodes[father]['normal_latency']
                        father = self.topo.nodes[father]['model'].input_nodes[0]
                    if Params + 7 + IAs > self.balance_point and fAbnormal < Abnormal:
                        fAbnormal = Abnormal
                        fcost = Abnormal
                        fParams = Params
                    elif Params + 7 + IAs < self.balance_point and fNormal < Normal:
                        fNormal = Normal
                        fcost = Normal
                        fParams = Params
                normal_cost[n][j] = fNormal
                abnormal_cost[n][j] = fAbnormal
                cost[n][j] = fcost
                total_params[n][j] = fParams
        partition_flag = [[0 for i in range(num_nodes)] for i in range(num_nodes)]
        # # self.blocks_params[0][1] = [0, 0, 0, 0]
        # layers_n = len(self.blocks_params)
        # latency = [[0 for i in range(layers_n)] for i in range(layers_n)]
        # params_table = [[0 for i in range(layers_n)] for i in range(layers_n)]
        # IAs_table = [[0 for i in range(layers_n)] for i in range(layers_n)]
        # for i in range(0, layers_n):
        #     latency[i][i] = self.blocks_params[i][4]
        #     params_table[i][i] = self.blocks_params[i][3]
        #     IAs_table[i][i] = size2memory(self.blocks_params[i][1])
        #     for j in range(i+1, layers_n):
        #         total_params = 0
        #         IAs = 0
        #         for idx in range(i, j+1):
        #             total_params += self.blocks_params[idx][3]
        #             IAs = max(IAs, size2memory(self.blocks_params[idx][1]))
        #         params_table[i][j] = total_params
        #         IAs_table[i][j] = IAs
        #         if total_params + IAs + 7 > self.balance_point:
        #             for idx in range(i, j+1):
        #                 latency[i][j] += self.blocks_params[idx][5]
        #         else:
        #             for idx in range(i, j+1):
        #                 latency[i][j] += self.blocks_params[idx][4]
        #
        # if smart_flag:
        #     upper_bound = latency[-1][-1]/2
        #     orgin = []
        #     for i in range(layers_n):
        #         orgin.append(latency[i][i])
        #     print("{:.3f}".format(max(orgin)))
        #     peak = max(orgin)
        #     # print(max(orgin)-176)
        #     flag = [0 for i in range(layers_n)]
        #     for i in range(5):
        #         idx = 1
        #         f = 1
        #         while idx < len(orgin):
        #             if orgin[idx] + orgin[idx-1] < peak:
        #                 new_val = orgin[idx] + orgin[idx-1]
        #                 orgin.pop(idx)
        #                 orgin.pop(idx-1)
        #                 orgin.insert(idx-1, new_val)
        #                 flag[f] = 1
        #                 # print(idx)
        #             elif (orgin[idx] + orgin[idx-1] - peak) / peak < 0.1:
        #                 peak = orgin[idx] + orgin[idx-1]
        #                 orgin.pop(idx)
        #                 orgin.pop(idx - 1)
        #                 orgin.insert(idx - 1, peak)
        #                 flag[f] = 1
        #                 # print(idx)
        #             else:
        #                 # flag[f] = 2
        #                 idx += 1
        #             f += 1
        #     print(flag)
        #     orgin = [float('{:.3f}'.format(i)) for i in orgin]
        #     print('len ', len(orgin))
        #     print(orgin)
        #     print(max(orgin)*len(orgin))
        #     flag.append(-1)
        #     self.strategy = flag
        # else:
        #     for i in range(layers_n):
        #         print(latency[i][i])
        #     partition_flag = [[0 for i in range(layers_n)] for i in range(layers_n)]
        #     func = [0 for i in range(layers_n+1)]
        #     block_latency = [0 for i in range(layers_n + 1)]
        #     num_blocks = [0 for i in range(layers_n + 1)]
        #     func[1] = latency[0][0]
        #     num_blocks[1] = 1
        #     block_latency[1] = latency[0][0]
        #     for i in range(0, layers_n):
        #         point_type = 0
        #         min_func = 9999
        #         bl = block_latency[i+1]
        #         nb = num_blocks[i+1]
        #         partition_point = -1
        #         for j in range(0, i+1):
        #             trans = 15 * size2memory(self.blocks_params[j][1]) if j > 0 else 0
        #             # old_trans = func[j] - block_latency[j]
        #             # 1. new block 2.combination
        #             if max(latency[j][i], trans) > block_latency[j] and min_func > (max(latency[j][i], trans)) * (num_blocks[j] + 1):
        #                 min_func = max(latency[j][i], trans) * (num_blocks[j] + 1)
        #                 bl = max(latency[j][i], trans)
        #                 nb = num_blocks[j] + 1
        #                 partition_point = j
        #             if max(latency[j][i], trans) < block_latency[j] and min_func > (block_latency[j]) * (num_blocks[j] + 1):
        #                 min_func = (block_latency[j]) * (num_blocks[j] + 1)
        #                 bl = block_latency[j]
        #                 nb = num_blocks[j] + 1
        #                 partition_point = j
        #         func[i+1] = min_func
        #         num_blocks[i+1] = nb
        #         block_latency[i+1] = bl
        #         print('n={} partition point:{} block latency:{:.3f} num_blocks:{} min_func:{:.3f}'.format(i, partition_point, bl, nb, min_func))
        #         partition_flag[i][partition_point] = 1
        #     self.strategy = partition_flag
        #     partition_flag = np.array(partition_flag)
        #     latency = np.array(latency)
        #     l = 0
        #     for i in range(0, layers_n):
        #         l += self.blocks_params[i][4]
        #     print(l)
        #     print(func[-1])

    def save_graph_json(self, filename='model_graph.json'):
        with open(filename, 'w') as f:
            maxSizePerFPGA = 47185920
            maxFPGAs = 20
            maxCPUs = 20
            nodes = [dict(id=n,
                          supportedOnFpga=0,
                          cpuLatency=self.topo.nodes[n]['abnormal_latency'],
                          fpgaLatency=self.topo.nodes[n]['normal_latency'],
                          isBackwardNode=0,
                          size=int(self.topo.nodes[n]['params']) * 1024 * 1024) for n in self.topo.nodes()]
            edges = [dict(sourceId=u,
                          destId=v,
                          cost=size2memory(self.topo.nodes[u]['output_shape'])) for u, v in self.topo.edges()]
            json.dump(dict(maxSizePerFPGA=maxSizePerFPGA,
                           maxFPGAs=maxFPGAs,
                           maxCPUs=maxCPUs,
                           nodes=nodes,
                           edges=edges), f)

    def load_graph_json(self, filename='model_graph.json'):
        G = nx.DiGraph()
        d = json.load(open(filename))
        G.add_nodes_from(d['nodes'])
        G.add_edges_from(d['edges'])
        self.topo = G

    def generate_block_model(self, build_dir='model/'):
        for idx, bp in enumerate(self.blocks_params):
            model = bp[0]
            input_size = bp[1]
            _torch2onnx(model, torch.rand(input_size))
            path = osp.join(build_dir, str(idx))
            if not osp.exists(path):
                os.makedirs(path)
            _onnx2tvm(torch.rand(input_size), build_dir=path)

    def generate_pipeline_model(self, build_dir='model/'):
        model = self.blocks_params[0][0]
        input_size = self.blocks_params[0][1]
        idx = 0
        trans = 0
        for i, flag in enumerate(self.strategy):
            if i == 0:
                continue
            if flag == 0:
                path = osp.join(build_dir, str(idx))
                if not osp.exists(path):
                    os.makedirs(path)
                trans += 15 * size2memory(input_size)
                # _torch2onnx(model, torch.rand(input_size))
                # # _onnx2tvm(torch.rand(input_size), build_dir=path)
                # _onnx2tvm(torch.rand(input_size), build_dir='./')
                idx += 1
                # print('Writing model into ' + path)
                print('transmission latency', 15 * size2memory(input_size))
                # print('Block latency: ', _calculate_latency(str(input_size[0]) + '/' + str(input_size[1]) + '/' +
                #                                             str(input_size[2]) + '/' + str(input_size[3]),
                #                                             heap_size=0x30))
                model = self.blocks_params[i][0]
                input_size = self.blocks_params[i][1]
            elif flag == -1:
                path = osp.join(build_dir, str(idx))
                if not osp.exists(path):
                    os.makedirs(path)
                trans += 15 * size2memory(input_size)
                # _torch2onnx(model, torch.rand(input_size))
                # # _onnx2tvm(torch.rand(input_size), build_dir=path)
                # _onnx2tvm(torch.rand(input_size), build_dir='./')
                idx += 1
                # print('Writing model into ' + path)
                print('transmission latency', 15 * size2memory(input_size))
                # print('Block latency: ', _calculate_latency(str(input_size[0]) + '/' + str(input_size[1]) + '/' +
                #                                             str(input_size[2]) + '/' + str(input_size[3]),
                #                                             heap_size=0x30))
            else:
                model = nn.Sequential(model, self.blocks_params[i][0])

        print('total transmission latency', trans)

    def generate_model(self, build_dir='model/'):
        index_old = len(self.strategy) - 1
        modelset = []
        while index_old > 0:
            result = [i for i, j in enumerate(self.strategy[index_old]) if j == 1 or j == 2]
            if not result:
                return
            index_new = result[0] - 1
            model = self.blocks_params[index_new + 1][0]
            input_size = self.blocks_params[index_new + 1][1]
            latency = self.blocks_params[index_new + 1][4]
            for i in range(index_new + 2, index_old + 1):
                model = nn.Sequential(model, self.blocks_params[i][0])
                latency += self.blocks_params[i][4]
            modelset.append((model, input_size))
            print('latency: ', latency)
            index_old = index_new
            # print(index_new)
        idx = 0
        trans = 0
        for (model, input_size) in reversed(modelset):
            path = osp.join(build_dir, str(idx))
            idx += 1
            if not osp.exists(path):
                os.makedirs(path)
            trans += 15 * size2memory(input_size) + random.random() * 2
            _torch2onnx(model, torch.rand(input_size))
            # _onnx2tvm(torch.rand(input_size), build_dir=path)
            _onnx2tvm(torch.rand(input_size), build_dir='./')
            print('Writing model into ' + path)
            print('Block latency: ', _calculate_latency(str(input_size[0]) + '/' + str(input_size[1]) + '/' +
                                                        str(input_size[2]) + '/' + str(input_size[3]), heap_size=0x30))
            print('transmission latency', 15 * size2memory(input_size) + random.random() * 2)
        print('total transmission latency', trans)

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
            # print('partition')
            # self.partition()


if __name__ == '__main__':
    # import torchvision.models as models
    # model = models.vgg16(pretrained=False)
    # _, total_params = profile(model, (torch.rand((1, 3, 224, 224)),), verbose=False)
    # print("%s | %.3f MB" % ('model', float(total_params * 4. / (1024 ** 2.))))
    # total_ops, total_params = profile(model, (torch.randn((1, 3, 224, 224)),), verbose=False)
    # print("%s | %.3f MB | %.3fG GFLOPs" % ('model', float(total_params * 4. / (1024 ** 2.)), total_ops / (1000 ** 3)))

    # model = ResNet(Bottleneck, [3, 4, 6, 3])
    # ms = ModelSet(model, (1, 3, 224, 224), unit=[wrapper])
    # total_ops, total_params = profile(model, (torch.randn((1, 3, 224, 224)),), verbose=False)
    # print("%s | %.3f MB | %.3fG GFLOPs" % ('model', float(total_params * 4. / (1024 ** 2.)), total_ops / (1000 ** 3)))
    # ms.run()
    # with open('graph/resnet50.o', 'wb') as f:
    #     pickle.dump(ms, f)

    # look up for an old partition
    with open('/home/lifabing/sgx/best-partion/graph/resnet50.o', 'rb') as f:
        ms = pickle.load(f)
        ms.save_graph_json()
        # ms.partition()
        # ms.generate_pipeline_model()
        # ms.generate_model()
        # temp = list(ms.blocks_params[21])

    # _torch2onnx(model, torch.rand(1, 3, 224, 224))
    # _onnx2tvm(torch.rand(1, 3, 224, 224), build_dir='./')
