import torch
import torch.nn as nn


class conv_block(nn.Module):
    def __init__(self, input_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), pool=None, bn_flag=True):
        super(conv_block, self).__init__()
        conv = nn.Conv2d(input_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                         groups=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        if not bn_flag:
            bn = None
        relu = nn.ReLU6(inplace=True)
        # if pool is None:
        #     self.features = nn.Sequential(conv, bn, relu)
        # else:
        #     self.features = nn.Sequential(conv, bn, relu, pool)
        layers = [conv, bn, relu, pool]
        for i in layers:
            if i is None:
                layers.remove(i)
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        return self.features(x)


class separable_conv_block(nn.Module):
    def __init__(self, depthwise_channels, pointwise_channels, kernel_size=(3, 3), downsample=False, padding=(1, 1)):
        super(separable_conv_block, self).__init__()
        """Helper function to get a separable conv block"""
        if downsample:
            strides = (2, 2)
        else:
            strides = (1, 1)
        # depthwise convolution + bn + relu
        conv1 = nn.Conv2d(
            in_channels=depthwise_channels,
            out_channels=pointwise_channels,
            groups=depthwise_channels,
            kernel_size=kernel_size,
            stride=strides,
            padding=padding)
        bn1 = nn.BatchNorm2d(pointwise_channels)
        act1 = nn.ReLU6(inplace=True)
        # pointwise convolution + bn + relu
        conv2 = nn.Conv2d(
            pointwise_channels,
            out_channels=pointwise_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0), )
        bn2 = nn.BatchNorm2d(pointwise_channels)
        act2 = nn.ReLU6(inplace=True)
        self.features = nn.Sequential(conv1, bn1, act1, conv2, bn2, act2)

    def forward(self, x):
        return self.features(x)


class Classifier(nn.Module):
    def __init__(self, input_planes, num_classes, pool_stride=(1, 1)):
        super(Classifier, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(pool_stride)
        self.fc = nn.Linear(input_planes, num_classes)

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class one_layer(nn.Module):
    def __init__(self, num_classes=10):
        super(one_layer, self).__init__()
        self.conv1 = self.conv_block(3, 32, stride=(2, 2))
        self.conv2 = self.conv_block(32, 64, stride=(2, 2))
        # self.conv3 = self.conv_block(64, 128, stride=(2, 2))
        # self.conv4 = self.conv_block(128, 256, stride=(2, 2))
        # self.conv5 = self.conv_block(256, 512, stride=(2, 2))
        # self.conv6 = self.conv_block(512, 1024, stride=(2, 2))
        # self.conv7 = self.conv_block(1024, 2048, stride=(2, 2))
        # self.classifier = nn.Linear(1024, num_classes)

    # 模型计算时的前向过程，也就是按照这个过程进行计算
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        # out = self.conv3(out)
        # out = self.conv4(out)
        # out = self.conv5(out)
        # out = self.conv6(out)
        # out = self.conv7(out)
        return out

    def conv_block(self, input_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        conv = nn.Conv2d(input_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, groups=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU6(inplace=True)
        return nn.Sequential(conv, bn, relu)


class mobilenet1(nn.Module):
    def __init__(self, num_classes=10):
        super(mobilenet1, self).__init__()
        self.conv1 = conv_block(3, 32, stride=(2, 2))
        self.s1 = separable_conv_block(32, 64)
        # self.s2 = separable_conv_block(64, 128, downsample=True)
        # self.s3 = separable_conv_block(128, 128)
        # self.classifier = nn.Linear(1024, num_classes)

    # 模型计算时的前向过程，也就是按照这个过程进行计算
    def forward(self, x):
        out = self.conv1(x)
        out = self.s1(out)
        # out = self.s2(out)
        # out = self.s3(out)
        # out = self.conv5(out)
        # out = self.conv6(out)
        # out = self.conv7(out)
        return out


class mobilenet2(nn.Module):
    def __init__(self, num_classes=10):
        super(mobilenet2, self).__init__()
        self.s4 = separable_conv_block(64, 128, downsample=True)
        self.s5 = separable_conv_block(128, 128)
        self.s1 = separable_conv_block(128, 256, downsample=True)
        self.s2 = separable_conv_block(256, 256)
        self.s3 = separable_conv_block(256, 512, downsample=True)
        # self.classifier = nn.Linear(1024, num_classes)

    # 模型计算时的前向过程，也就是按照这个过程进行计算
    def forward(self, x):
        out = self.s4(x)
        out = self.s5(out)
        out = self.s1(out)
        out = self.s2(out)
        out = self.s3(out)
        # out = self.conv5(out)
        # out = self.conv6(out)
        # out = self.conv7(out)
        return out


class mobilenet3(nn.Module):
    def __init__(self, num_classes=10):
        super(mobilenet3, self).__init__()
        features = []
        for i in range(7, 12):
            features.append(separable_conv_block(512, 512))
        self.features = nn.Sequential(*features)
        # self.classifier = nn.Linear(1024, num_classes)

    # 模型计算时的前向过程，也就是按照这个过程进行计算
    def forward(self, x):
        out = self.features(x)
        return out


class mobilenet4(nn.Module):
    def __init__(self, num_classes=10):
        super(mobilenet4, self).__init__()
        self.s1 = separable_conv_block(512, 1024, downsample=True)
        self.s2 = separable_conv_block(1024, 1024)
        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes),
        )

    # 模型计算时的前向过程，也就是按照这个过程进行计算
    def forward(self, x):
        out = self.s1(x)
        out = self.s2(out)
        out = out.mean([2, 3])
        out = self.classifier(out)
        # out = self.conv5(out)
        # out = self.conv6(out)
        # out = self.conv7(out)
        return out


class mobilenet(nn.Module):
    def __init__(self, num_classes=10):
        super(mobilenet, self).__init__()
        self.conv1 = conv_block(3, 32, stride=(2, 2))
        self.s1 = separable_conv_block(32, 64)
        self.s2 = separable_conv_block(64, 128, downsample=True)
        self.s3 = separable_conv_block(128, 128)
        self.s4 = separable_conv_block(128, 256, downsample=True)
        self.s5 = separable_conv_block(256, 256)
        self.s6 = separable_conv_block(256, 512, downsample=True)
        features = []
        for i in range(7, 12):
            features.append(separable_conv_block(512, 512))
        self.features = nn.Sequential(*features)
        self.s12 = separable_conv_block(512, 1024, downsample=True)
        self.s13 = separable_conv_block(1024, 1024)
        # self.pool = nn.AvgPool2d(7)
        # building classifier
        # self.classifier = nn.Sequential(
        #     nn.Dropout(0.2),
        #     nn.Linear(1024, num_classes),
        # )

    # 模型计算时的前向过程，也就是按照这个过程进行计算
    def forward(self, x):
        out = self.conv1(x)
        out = self.s1(out)
        out = self.s2(out)
        out = self.s3(out)
        out = self.s4(out)
        out = self.s5(out)
        out = self.s6(out)
        out = self.features(out)
        out = self.s12(out)
        out = self.s13(out)
        # out = out.mean([2, 3])
        # out = self.pool(out)
        out = out.view(-1, 1024)
        # out = self.classifier(out)
        return out
    # def separable_conv_block(self, depthwise_channels, pointwise_channels,
    #                          kernel_size=(3, 3), downsample=False, padding=(1, 1)):
    #     """Helper function to get a separable conv block"""
    #     if downsample:
    #         strides = (2, 2)
    #     else:
    #         strides = (1, 1)
    #     # depthwise convolution + bn + relu
    #     conv1 = nn.Conv2d(
    #         depthwise_channels,
    #         groups=pointwise_channels,
    #         kernel_size=kernel_size,
    #         stride=strides,
    #         padding=padding)
    #     bn1 = nn.BatchNorm2d(pointwise_channels)
    #     act1 = nn.ReLU6(inplace=True)
    #     # pointwise convolution + bn + relu
    #     conv2 = nn.Conv2d(
    #         pointwise_channels,
    #         kernel_size=(1, 1),
    #         stride=(1, 1),
    #         padding=(0, 0),)
    #     bn2 = nn.BatchNorm2d(pointwise_channels)
    #     act2 = nn.ReLU6(inplace=True)
    #     return nn.Sequential(conv1, bn1, act1, conv2, bn2, act2)


class ResNet1(nn.Module):

    def __init__(self, block, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet1, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        # self.bn1 = norm_layer(self.inplanes)
        # self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv_block(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                pool=nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, 2)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


class ResNet2(nn.Module):

    def __init__(self, block, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.layer2 = self._make_layer(block, 128, 2, stride=2,
                                       dilate=replace_stride_with_dilation[0])

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.layer2(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


class ResNet3(nn.Module):

    def __init__(self, block, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet3, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 128
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.layer3 = self._make_layer(block, 256, 2, stride=2,
                                       dilate=replace_stride_with_dilation[1])

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.layer3(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


class ResNet4(nn.Module):

    def __init__(self, block, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet4, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 256
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.layer4 = self._make_layer(block, 512, 2, stride=2,
                                       dilate=replace_stride_with_dilation[2])
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.classifier = Classifier(512 * block.expansion, num_classes, (1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        x = self.classifier(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet18, self).__init__()
        block = BasicBlock
        self.r1 = ResNet1(block)
        self.r2 = ResNet2(block)
        self.r3 = ResNet3(block)
        self.r4 = ResNet4(block, num_classes=num_classes)

    def forward(self, x):
        x = self.r1(x)
        x = self.r2(x)
        x = self.r3(x)
        x = self.r4(x)
        return x


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        # self.bn1 = norm_layer(self.inplanes)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv1 = conv_block(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                pool=nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.classifier = Classifier(512 * block.expansion, num_classes, (1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        x = self.classifier(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


class vgg_pool(nn.Module):
    def __init__(self):
        super(vgg_pool, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(x)
        return x


class vgg_classifier(nn.Module):
    def __init__(self, num_classes=1000):
        super(vgg_classifier, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = vgg_classifier(num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            # layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            layers += [vgg_pool()]
        else:
            # conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                # layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                layers += [conv_block(in_channels, v, kernel_size=3, stride=1, padding=1)]
            else:
                # layers += [conv2d, nn.ReLU(inplace=True)]
                layers += [conv_block(in_channels, v, kernel_size=3, stride=1, padding=1, bn_flag=False)]
            in_channels = v
    return nn.Sequential(*layers)

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def get_vgg(cfg, batch_norm, **kwargs):
    kwargs['init_weights'] = True
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    return model