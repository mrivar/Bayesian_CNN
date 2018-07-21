import torch
import torch.nn.functional as F
import torch.nn as nn
from .BBBlayers import BBBConv2d, BBBLinearFactorial, FlattenLayer
from collections import OrderedDict


# Bayesian LeNet
class BBBLeNet(nn.Module):
    def __init__(self, outputs, inputs):
        super(BBBLeNet, self).__init__()
        self.conv1 = BBBConv2d(inputs, 6, 5, stride=1)
        self.soft1 = nn.Softplus()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = BBBConv2d(6, 16, 5, stride=1)
        self.soft2 = nn.Softplus()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = FlattenLayer(5 * 5 * 16)
        self.fc1 = BBBLinearFactorial(5 * 5 * 16, 120)
        self.soft3 = nn.Softplus()

        self.fc2 = BBBLinearFactorial(120, 84)
        self.soft4 = nn.Softplus()

        self.fc3 = BBBLinearFactorial(84, outputs)

        layers = [self.conv1, self.soft1, self.pool1, self.conv2, self.soft2, self.pool2,
                  self.flatten, self.fc1, self.soft3, self.fc2, self.soft4, self.fc3]

        self.layers = nn.ModuleList(layers)

    def probforward(self, x):
        kl = 0
        for layer in self.layers:
            if hasattr(layer, 'convprobforward') and callable(layer.convprobforward):
                x, _kl, = layer.convprobforward(x)
                kl += _kl

            elif hasattr(layer, 'fcprobforward') and callable(layer.fcprobforward):
                x, _kl, = layer.fcprobforward(x)
                kl += _kl
            else:
                x = layer(x)
        logits = x
        print('logits', logits)
        return logits, kl


# Bayesian 3Conv3FC
class BBB3Conv3FC(nn.Module):
    """
    To train on CIFAR-10:
    https://arxiv.org/pdf/1207.0580.pdf
    """
    def __init__(self, outputs, inputs):
        super(BBB3Conv3FC, self).__init__()
        self.conv1 = BBBConv2d(inputs, 32, 5, stride=1, padding=2)
        self.soft1 = nn.Softplus()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = BBBConv2d(32, 64, 5, stride=1, padding=2)
        self.soft2 = nn.Softplus()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = BBBConv2d(64, 128, 5, stride=1, padding=1)
        self.soft3 = nn.Softplus()
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.flatten = FlattenLayer(2 * 2 * 128)
        self.fc1 = BBBLinearFactorial(2 * 2 * 128, 1000)
        self.soft5 = nn.Softplus()

        self.fc2 = BBBLinearFactorial(1000, 1000)
        self.soft6 = nn.Softplus()

        self.fc3 = BBBLinearFactorial(1000, outputs)

        layers = [self.conv1, self.soft1, self.pool1, self.conv2, self.soft2, self.pool2,
                  self.conv3, self.soft3, self.pool3, self.flatten, self.fc1, self.soft5,
                  self.fc2, self.soft6, self.fc3]

        self.layers = nn.ModuleList(layers)

    def probforward(self, x):
        kl = 0
        for layer in self.layers:
            if hasattr(layer, 'convprobforward') and callable(layer.convprobforward):
                x, _kl, = layer.convprobforward(x)
                kl += _kl

            elif hasattr(layer, 'fcprobforward') and callable(layer.fcprobforward):
                x, _kl, = layer.fcprobforward(x)
                kl += _kl
            else:
                x = layer(x)
        logits = x
        print('logits', logits)
        return logits, kl


# Bayesian ELU-Network
class BBBELUN1(nn.Module):
    """
    To train on CIFAR-100:
    https://arxiv.org/pdf/1511.07289.pdf
    """
    def __init__(self, outputs, inputs):
        super(BBBELUN1, self).__init__()
        self.conv1 = BBBConv2d(inputs, 384, 3, stride=1)
        self.soft1 = nn.Softplus()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = BBBConv2d(384, 384, 1, stride=1)
        self.soft2 = nn.Softplus()
        self.conv3 = BBBConv2d(384, 384, 2, stride=1)
        self.soft3 = nn.Softplus()
        self.conv4 = BBBConv2d(384, 640, 2, stride=1)
        self.soft4 = nn.Softplus()
        self.conv5 = BBBConv2d(640, 640, 2, stride=1)
        self.soft5 = nn.Softplus()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv6 = BBBConv2d(640, 640, 1, stride=1)
        self.soft6 = nn.Softplus()
        self.conv7 = BBBConv2d(640, 768, 2, stride=1)
        self.soft7 = nn.Softplus()
        self.conv8 = BBBConv2d(640, 768, 2, stride=1)
        self.soft8 = nn.Softplus()
        self.conv9 = BBBConv2d(640, 768, 2, stride=1)
        self.soft9 = nn.Softplus()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv10 = BBBConv2d(768, 768, 1, stride=1)
        self.soft10 = nn.Softplus()
        self.conv11 = BBBConv2d(768, 896, 2, stride=1)
        self.soft11 = nn.Softplus()
        self.conv12 = BBBConv2d(896, 896, 2, stride=1)
        self.soft12 = nn.Softplus()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv13 = BBBConv2d(896, 896, 3, stride=1)
        self.soft13 = nn.Softplus()
        self.conv14 = BBBConv2d(896, 1024, 2, stride=1)
        self.soft14 = nn.Softplus()
        self.conv15 = BBBConv2d(896, 1024, 2, stride=1)
        self.soft15 = nn.Softplus()
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv16 = BBBConv2d(1024, 1024, 1, stride=1)
        self.soft16 = nn.Softplus()
        self.conv17 = BBBConv2d(1024, 1152, 2, stride=1)
        self.soft17 = nn.Softplus()
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv18 = BBBConv2d(1152, 1152, 2, stride=1)
        self.soft18 = nn.Softplus()
        self.pool7 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv19 = BBBConv2d(1152, outputs, 1, stride=1)

        layers = [self.conv1, self.soft1, self.pool1, self.conv2, self.soft2, self.conv3, self.soft3, self.conv4,
                  self.soft4, self.pool2, self.conv5, self.soft5, self.conv6, self.soft6, self.conv7, self.soft7,
                  self.conv8, self.soft8, self.pool3, self.conv9, self.soft9, self.conv10, self.soft10, self.conv11,
                  self.soft11, self.pool4, self.conv12, self.soft12, self.conv13, self.soft13, self.conv14,
                  self.soft14, self.pool5, self.conv15, self.soft15, self.pool5, self.conv16, self.soft16,
                  self.conv17, self.soft17, self.pool6, self.conv18, self.soft18, self.pool7, self.conv19]

        self.layers = nn.ModuleList(layers)

    def probforward(self, x):
        kl = 0
        for layer in self.layers:
            if hasattr(layer, 'convprobforward') and callable(layer.convprobforward):
                x, _kl, = layer.convprobforward(x)
                kl += _kl

            elif hasattr(layer, 'fcprobforward') and callable(layer.fcprobforward):
                x, _kl, = layer.fcprobforward(x)
                kl += _kl
            else:
                x = layer(x)
        logits = x
        print('logits', logits)
        return logits, kl


# Bayesian ELU-Network
class BBBELUN2(nn.Module):
    """
    To train on ImageNet:
    https://arxiv.org/pdf/1511.07289.pdf
    """
    def __init__(self, outputs, inputs):
        super(BBBELUN2, self).__init__()
        self.conv1 = BBBConv2d(inputs, 96, 6, stride=1)
        self.soft1 = nn.Softplus()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = BBBConv2d(96, 512, 3, stride=1)
        self.soft2 = nn.Softplus()
        self.conv3 = BBBConv2d(512, 512, 3, stride=1)
        self.soft3 = nn.Softplus()
        self.conv4 = BBBConv2d(512, 512, 3, stride=1)
        self.soft4 = nn.Softplus()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = BBBConv2d(512, 768, 3, stride=1)
        self.soft5 = nn.Softplus()
        self.conv6 = BBBConv2d(768, 768, 3, stride=1)
        self.soft6 = nn.Softplus()
        self.conv7 = BBBConv2d(768, 768, 2, stride=1)
        self.soft7 = nn.Softplus()
        self.conv8 = BBBConv2d(768, 768, 2, stride=1)
        self.soft8 = nn.Softplus()
        self.conv9 = BBBConv2d(768, 768, 1, stride=1)
        self.soft9 = nn.Softplus()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv10 = BBBConv2d(768, 1024, 3, stride=1)
        self.soft10 = nn.Softplus()
        self.conv11 = BBBConv2d(1024, 1024, 3, stride=1)
        self.soft11 = nn.Softplus()
        self.conv12 = BBBConv2d(1024, 1024, 3, stride=1)
        self.soft12 = nn.Softplus()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = FlattenLayer(2 * 2 * 128)
        self.fc1 = BBBLinearFactorial(2 * 2 * 128, 4096)
        self.soft13 = nn.Softplus()

        self.fc2 = BBBLinearFactorial(4096, 4096)
        self.soft14 = nn.Softplus()

        self.fc3 = BBBLinearFactorial(4096, outputs)

        layers = [self.conv1, self.soft1, self.pool1, self.conv2, self.soft2, self.conv3, self.soft3, self.conv4,
                  self.soft4, self.pool2, self.conv5, self.soft5, self.conv6, self.soft6, self.conv7, self.soft7,
                  self.conv8, self.soft8, self.conv9, self.soft9, self.pool3, self.conv10, self.soft10, self.conv11,
                  self.soft11, self.conv12, self.soft12,  self.pool4, self.flatten, self.fc1, self.soft13, self.fc2,
                  self.soft14, self.fc3]

        self.layers = nn.ModuleList(layers)

    def probforward(self, x):
        kl = 0
        for layer in self.layers:
            if hasattr(layer, 'convprobforward') and callable(layer.convprobforward):
                x, _kl, = layer.convprobforward(x)
                kl += _kl

            elif hasattr(layer, 'fcprobforward') and callable(layer.fcprobforward):
                x, _kl, = layer.fcprobforward(x)
                kl += _kl
            else:
                x = layer(x)
        logits = x
        print('logits', logits)
        return logits, kl



"""
forget everything from here
"""
# Bayesian SqueezeNet
class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = BBBConv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.Softplus()
        self.expand1x1 = BBBConv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.Softplus()
        self.expand3x3 = BBBConv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.Softplus()

        firelayers = [self.squeeze, self.squeeze_activation, self.expand1x1, self.expand1x1_activation,
                      self.expand3x3, self.expand3x3_activation]

        self.firelayers = nn.ModuleList(firelayers)

    def fireprobforward(self, x):
        kl = 0
        for firelayer in self.firelayers:
            if hasattr(firelayer, 'convprobforward') and callable(firelayer.convprobforward):
                x, _kl, = firelayer.convprobforward(x)
                kl += _kl

            elif hasattr(firelayer, 'fcprobforward') and callable(firelayer.fcprobforward):
                x, _kl, = firelayer.fcprobforward(x)
                kl += _kl
            else:
                x = firelayer(x)
        logits = x
        print('logits', logits)
        return logits, kl, torch.cat([
                    self.expand1x1_activation(self.expand1x1(x)),
                    self.expand3x3_activation(self.expand3x3(x))], 1)


class BBBSqueezeNet(nn.Module):
    def __init__(self, inputs, outputs):
        super(BBBSqueezeNet, self).__init__()
        self.conv1 = BBBConv2d(inputs, 64, kernel_size=3, stride=2)
        self.soft1 = nn.Softplus()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.fire1 = Fire(64, 64, 64, 64)
        """
        self.squeeze1 = BBBConv2d(64, 16, kernel_size=1)
        self.squeeze_activation1 = nn.Softplus()
        self.expand1x1_1 = BBBConv2d(16, 64, kernel_size=1)
        self.expand1x1_activation1 = nn.Softplus()
        self.expand3x3_1 = BBBConv2d(16, 64,
                                     kernel_size=3, padding=1)
        self.expand3x3_activation1 = nn.Softplus()
        """
        self.fire2 = Fire(128, 16, 64, 64)
        """
        self.squeeze2 = BBBConv2d(128, 16, kernel_size=1)
        self.squeeze_activation2 = nn.Softplus()
        self.expand1x1_2 = BBBConv2d(16, 64, kernel_size=1)
        self.expand1x1_activation2 = nn.Softplus()
        """
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.fire3 = Fire(128, 32, 128, 128)
        """
        self.squeeze3 = BBBConv2d(128, 32, kernel_size=1)
        self.squeeze_activation3 = nn.Softplus()
        self.expand1x1_3 = BBBConv2d(32, 128, kernel_size=1)
        self.expand1x1_activation3 = nn.Softplus()
        """
        self.fire4 = Fire(256, 32, 128, 128)
        """
        self.squeeze4 = BBBConv2d(256, 32, kernel_size=1)
        self.squeeze_activation4 = nn.Softplus()
        self.expand1x1_4 = BBBConv2d(32, 128, kernel_size=1)
        self.expand1x1_activation4 = nn.Softplus()
        """
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.fire5 = Fire(256, 48, 192, 192)
        """
        self.squeeze5 = BBBConv2d(256, 48, kernel_size=1)
        self.squeeze_activation5 = nn.Softplus()
        self.expand1x1_5 = BBBConv2d(48, 192, kernel_size=1)
        self.expand1x1_activation5 = nn.Softplus()
        """
        self.fire6 = Fire(384, 48, 192, 192)
        """
        self.squeeze6 = BBBConv2d(384, 48, kernel_size=1)
        self.squeeze_activation6 = nn.Softplus()
        self.expand1x1_6 = BBBConv2d(48, 192, kernel_size=1)
        self.expand1x1_activation6 = nn.Softplus()
        """
        self.fire7 = Fire(384, 64, 256, 256)
        """
        self.squeeze7 = BBBConv2d(384, 64, kernel_size=1)
        self.squeeze_activation7 = nn.Softplus()
        self.expand1x1_7 = BBBConv2d(64, 256, kernel_size=1)
        self.expand1x1_activation7 = nn.Softplus()
        """
        self.fire8 = Fire(512, 64, 256, 256)
        """
        self.squeeze8 = BBBConv2d(512, 64, kernel_size=1)
        self.squeeze_activation8 = nn.Softplus()
        self.expand1x1_8 = BBBConv2d(64, 256, kernel_size=1)
        self.expand1x1_activation8 = nn.Softplus()
        """
        # Final convolution
        self.drop1 = nn.Dropout()
        self.final_conv = BBBConv2d(512, outputs, kernel_size=1)
        self.softmax = nn.Softmax()
        #self.soft2 = nn.Softplus(),
        #self.pool4 = nn.AvgPool2d(13, stride=1)

        layers = [self.conv1, self.soft1, self.pool1,
                  self.fire1,
                  #self.squeeze1, self.squeeze_activation1, self.expand1x1_1, self.expand1x1_activation1,
                  self.fire2,
                  #self.squeeze2, self.squeeze_activation2, self.expand1x1_2, self.expand1x1_activation2,
                  self.pool2,
                  self.fire3,
                  #self.squeeze3, self.squeeze_activation3, self.expand1x1_3, self.expand1x1_activation3,
                  self.fire4,
                  #self.squeeze4, self.squeeze_activation4, self.expand1x1_4, self.expand1x1_activation4,
                  self.pool3,
                  self.fire5,
                  #self.squeeze5, self.squeeze_activation5, self.expand1x1_5, self.expand1x1_activation5,
                  self.fire6,
                  #self.squeeze6, self.squeeze_activation6, self.expand1x1_6, self.expand1x1_activation6,
                  self.fire7,
                  #self.squeeze7, self.squeeze_activation7, self.expand1x1_7, self.expand1x1_activation7,
                  self.fire8,
                  #self.squeeze8, self.squeeze_activation8, self.expand1x1_8, self.expand1x1_activation8,
                  self.drop1, self.final_conv, self.softmax
                  #self.soft2, self.pool4
                  ]

        self.layers = nn.ModuleList(layers)

    def probforward(self, x):
        kl = 0
        for layer in self.layers:
            if hasattr(layer, 'convprobforward') and callable(layer.convprobforward):
                x, _kl, = layer.convprobforward(x)
                kl += _kl
                print('x Size', x.size())

            elif hasattr(layer, 'fcprobforward') and callable(layer.fcprobforward):
                x, _kl, = layer.fcprobforward(x)
                kl += _kl

            elif hasattr(layer, 'fireprobforward') and callable(layer.fireprobforward):
                x, _kl, = layer.fireprobforward(x)
                kl += _kl
                print('x Size', x.size())
            else:
                x = layer(x)
                print('x Size', x.size())

        logits = x
        print('logits', logits)
        return logits, kl


# Bayesian DenseNet
class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('soft1', nn.Softplus()),
        self.add_module('conv1', BBBConv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('soft2', nn.Softplus()),
        self.add_module('conv2', BBBConv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('soft', nn.Softplus())
        self.add_module('conv', BBBConv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class BBBDenseNet(nn.Module):

    """
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, inputs, outputs, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0):

        super(BBBDenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', BBBConv2d(inputs, num_init_features, kernel_size=7, stride=2, padding=3)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('soft0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = BBBLinearFactorial(num_features, outputs)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out

    def probforward(self, x):
        kl = 0
        for layer in self.layers:
            if hasattr(layer, 'convprobforward') and callable(layer.convprobforward):
                x, _kl, = layer.convprobforward(x)
                kl += _kl

            elif hasattr(layer, 'fcprobforward') and callable(layer.fcprobforward):
                x, _kl, = layer.fcprobforward(x)
                kl += _kl
            else:
                x = layer(x)
        logits = x
        print('logits', logits)
        return logits, kl