import torch
import torch.nn as nn

import torchvision.models as models
from torchsummary import summary
from thop import profile
import hiddenlayer as hl
from self_defined_nn import *


shape = [(1, 3, 224, 224), (1, 3, 150, 150), (1, 32, 75, 75)]
input_tensor = torch.randn(shape[1])
# model = VGG('VGG19')
# model = one_layer()
model = ResNet1(BasicBlock, 10)
# model = models.resnet50()
# model = mobilenet1()
model = models.inception_v3()
# torchsummary calculation
summary(model, (input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]))
# hl.build_graph(model, input_tensor)
# thop calculation
total_ops, total_params = profile(model, (input_tensor, ), verbose=False)
print("%s | %.3f MB | %.3fG GFLOPs" % ('model', float(total_params * 4. / (1024 ** 2.)), total_ops / (1000 ** 3)))

input_tensor = model(input_tensor)
# model = mobilenet2()
model = ResNet2(BasicBlock, 10)
summary(model, (input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]))
total_ops, total_params = profile(model, (input_tensor, ), verbose=False)
print("%s | %.3f MB | %.3fG GFLOPs" % ('model', float(total_params * 4. / (1024 ** 2.)), total_ops / (1000 ** 3)))

input_tensor = model(input_tensor)
# model = mobilenet3()
model = ResNet3(BasicBlock, 10)
summary(model, (input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]))
total_ops, total_params = profile(model, (input_tensor, ), verbose=False)
print("%s | %.3f MB | %.3fG GFLOPs" % ('model', float(total_params * 4. / (1024 ** 2.)), total_ops / (1000 ** 3)))

input_tensor = model(input_tensor)
# model = mobilenet4()
model = ResNet4(BasicBlock, 10)
summary(model, (input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]))
total_ops, total_params = profile(model, (input_tensor, ), verbose=False)
print("%s | %.3f MB | %.3fG GFLOPs" % ('model', float(total_params * 4. / (1024 ** 2.)), total_ops / (1000 ** 3)))

