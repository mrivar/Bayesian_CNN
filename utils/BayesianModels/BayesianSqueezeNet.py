import torch
import torch.nn as nn
from utils.BBBlayers import BBBConv2d, BBBLinearFactorial, FlattenLayer


class BBBFire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(BBBFire, self).__init__()
        self.inplanes = inplanes

        self.squeeze = BBBConv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.Softplus()
        self.expand1x1 = BBBConv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.Softplus()
        self.expand3x3 = BBBConv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.Softplus()

        layers = [self.squeeze, self.squeeze_activation, self.expand1x1,self.expand1x1_activation, self.expand3x3,
                  self.expand3x3_activation]

        self.layers = nn.ModuleList(layers)

    def probforward(self, x):
        'Forward pass with Bayesian weights'
        kl = 0
        for layer in self.layers:
            if hasattr(layer, 'convprobforward') and callable(layer.convprobforward):
                logits, _kl, = layer.convprobforward(x)
                kl += _kl

            elif hasattr(layer, 'fcprobforward') and callable(layer.fcprobforward):
                logits, _kl, = layer.fcprobforward(x)
                kl += _kl
            else:
                logits = layer(x)
        print('logits', logits)
        return logits, kl, torch.cat([self.expand1x1_activation(self.expand1x1(x)),
                                      self.expand3x3_activation(self.expand3x3(x))
                                      ], 1)


class BBBSqueezeNet(nn.Module):

    def __init__(self, outputs, inputs):
        super(BBBSqueezeNet, self).__init__()
        self.outputs = outputs

        self.conv1 = BBBConv2d(inputs, 64, kernel_size=3, stride=2)
        self.soft1 = nn.Softplus()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.fire1 = BBBFire(64, 16, 64, 64)
        self.fire2 = BBBFire(128, 16, 64, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.fire3 = BBBFire(128, 32, 128, 128)
        self.fire4 = BBBFire(256, 32, 128, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.fire5 = BBBFire(256, 48, 192, 192)
        self.fire6 = BBBFire(384, 48, 192, 192)
        self.fire7 = BBBFire(384, 64, 256, 256)
        self.fire8 = BBBFire(512, 64, 256, 256)

        self.drop1 = nn.Dropout(p=0.5)
        self.conv2 = BBBConv2d(512, self.outputs, kernel_size=1)
        self.soft2 = nn.Softplus()
        self.pool4 = nn.AvgPool2d(13, stride=1)

        layers = [self.conv1, self.soft1, self.pool1,self.fire1, self.fire2, self.pool2, self.fire3, self.fire4,
                  self.pool3, self.fire5, self.fire6, self.fire7, self.fire8,self.drop1, self.conv2, self.soft2,
                  self.pool4]

        self.layers = nn.ModuleList(layers)

    def probforward(self, x):
        'Forward pass with Bayesian weights'
        kl = 0
        for layer in self.layers:
            if hasattr(layer, 'convprobforward') and callable(layer.convprobforward):
                logits, _kl, = layer.convprobforward(x)
                kl += _kl

            elif hasattr(layer, 'fcprobforward') and callable(layer.fcprobforward):
                logits, _kl, = layer.fcprobforward(x)
                kl += _kl
            else:
                logits = layer(x)
        print('logits', logits)
        return logits, kl
