import os
import pickle
from auto_model_partition import *

import numpy as np
import matplotlib.pyplot as plt


def poly_fit(x, y, trans, degree=5):
    x = np.array(x)
    print('x is :\n', x)
    y = np.array(y)
    print('y is :\n', y)
    # 用3次多项式拟合
    f1 = np.polyfit(x, y, degree)
    print('f1 is :\n', f1)

    p1 = np.poly1d(f1)
    print('p1 is :\n', p1)

    # 也可使用yvals=np.polyval(f1, x)
    yvals = p1(x)  # 拟合y值
    print('yvals is :\n', yvals)
    print('large ', p1(18.464863))
    # 绘图
    plot1 = plt.plot(x, y, 's', label='original values')
    plot2 = plt.plot(x, yvals, 'r', label='polyfit values')
    plot3 = plt.scatter(x, trans, marker='*', label='transition cost')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc=4)  # 指定legend的位置右下角
    plt.title('parameter size --- loading latency')
    plt.show()


def model():
    model = mobilenet(1000)
    ms = ModelSet(model, (1, 3, 224, 224))
    print('running')
    ms.run(80)
    print('dumping')
    with open('latency.o', 'wb') as f:
        pickle.dump(ms, f)


def load(filename):
    with open(filename, 'rb') as f:
        ms = pickle.load(f)
    return ms


def graphize():
    params1 = [3.715, 20.461, 72.211, 6943.982, 18464.863]
    latency1 = [1.133, 1.805, 1.849, 16.616, 64.283]
    trans = []
    params = []
    latency = []
    ms = load('latency.o')
    for block in ms.blocks_params:
        trans.append(block[1][0] * block[1][1] * block[1][2] * block[1][3] * 4 /1024/1024 * 20)
        params.append(block[2] * 1024)
        latency.append(block[3])
        # print('{' + '{}, {}'.format(block[2], block[3]) + '}')
    ms = load('latency-resnet18.o')
    for block in ms.blocks_params:
        trans.append(block[1][0] * block[1][1] * block[1][2] * block[1][3] * 4 / 1024 / 1024 * 20)
        params.append(block[2] * 1024)
        latency.append(block[3])
    # params.extend(params1)
    # latency.extend(latency1)
    poly_fit(params, latency, trans)
    # model()


def pt_params_stat(pt):
    params = []
    latency = []
    trans = []
    for group in pt.look_up_table:
        pa = 0
        lt = list(group.values())[0]
        for index in group.keys():
            for idx in index:
                pa += pt.block_params[idx][2]
            idx = index[-1]
            trans.append(pt.block_params[idx][1][0] * pt.block_params[idx][1][1] * pt.block_params[idx][1][2] *
                         pt.block_params[idx][1][3] * 4 / 1024 / 1024 * 20)
        params.append(pa)
        latency.append(lt)
    # zipped = sorted(zip(params, latency, trans))
    # print(zipped)
    # params, latency, trans = zip(*zipped)
    # poly_fit(params, latency, trans)
    # print(sorted(zip(params, latency)))
    return params, latency, trans


if __name__ == '__main__':
    pt = load('loading-partition-mobilenet.o')
    params, latency, trans = pt_params_stat(pt)
    pt = load('loading-partition-resnet18.o')
    params1, latency1, trans1 = pt_params_stat(pt)
    params.extend(params1)
    latency.extend(latency1)
    trans.extend(trans1)
    zipped = sorted(zip(params, latency, trans))
    news_li = [zipped[0]]
    for i in zipped:
        nn = news_li.copy()
        p, _, _ = zip(*nn)
        if i[0] not in p:
            news_li.append(i)
    params, latency, trans = zip(*news_li)
    poly_fit(params, latency, trans)
    print(news_li)

    # ms = load('latency-mobilenet.o')
    # pt = Partition(ms.blocks_params)
    # del ms
    # pt.binary_partition()
    # with open('loading-partition-mobilenet.o', 'wb') as f:
    #     pickle.dump(pt, f)
    # pt.get_strategy()
    # with open('loading-partition-mobilenet.o', 'wb') as f:
    #     pickle.dump(pt, f)
