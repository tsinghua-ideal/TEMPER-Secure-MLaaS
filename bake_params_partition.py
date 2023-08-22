import self_defined_nn as snn
from auto_model_partition import *
import torchvision.models as models


if __name__ == '__main__':
    # model = snn.get_vgg('D', False)
    # model = snn.ResNet(snn.Bottleneck, [3, 8, 36, 3], num_classes=1000)
    # model = snn.mobilenet(1000)
    model = snn.DenseNet(32, (6, 12, 48, 32), 64)
    # model = snn.Inception3(1000, aux_logits=False)
    ms = ModelSet(model, (1, 3, 224, 224), unit=[snn.conv_block, snn.Dense_Classifier, snn.DenseBlock, snn.Transition])
    # ms = ModelSet(model, (1, 3, 224, 224), unit=[snn.FirstBasicConv2d, snn.InceptionA, snn.InceptionB, snn.InceptionC, 
    # snn.InceptionD, snn.InceptionE, snn.InceptionAux, snn.inception_classifier, snn.inception_pool])
    # ms = ModelSet(model, (1, 3, 224, 224))
    ms._stat_layer_params()
    ms.generate_block_model('/home/lifabing/sgx/lasagna/sgx/lib/plus/densenet201')
    # ms.generate_vessels_model('/home/lifabing/sgx/lasagna/sgx/lib/plus/Inception3')