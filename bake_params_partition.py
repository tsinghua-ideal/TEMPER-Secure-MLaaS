import self_defined_nn
from auto_model_partition import *
import torchvision.models as models


if __name__ == '__main__':
    model = self_defined_nn.get_vgg('D', False)
    # model = self_defined_nn.ResNet(self_defined_nn.Bottleneck, [3, 8, 36, 3], num_classes=1000)
    # model = self_defined_nn.mobilenet(1000)
    ms = ModelSet(model, (1, 3, 224, 224))
    ms._stat_layer_params()
    ms.generate_vessels_model('/home/lifabing/sgx/lasagna/sgx/lib/plus/vgg16')