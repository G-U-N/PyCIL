'''
For MEMO implementations of CIFAR-ResNet
Reference:
https://github.com/khurramjaved96/incremental-learning/blob/autoencoders/model/resnet32.py
'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class DownsampleA(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleA, self).__init__()
        assert stride == 2
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.avg(x)
        return torch.cat((x, x.mul(0)), 1)

class ResNetBasicblock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResNetBasicblock, self).__init__()

        self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(planes)

        self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(planes)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        basicblock = self.conv_a(x)
        basicblock = self.bn_a(basicblock)
        basicblock = F.relu(basicblock, inplace=True)

        basicblock = self.conv_b(basicblock)
        basicblock = self.bn_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(x)

        return F.relu(residual + basicblock, inplace=True)



class GeneralizedResNet_cifar(nn.Module):
    def __init__(self, block, depth, channels=3):
        super(GeneralizedResNet_cifar, self).__init__()
        assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
        layer_blocks = (depth - 2) // 6
        self.conv_1_3x3 = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(16)

        self.inplanes = 16 
        self.stage_1 = self._make_layer(block, 16, layer_blocks, 1)
        self.stage_2 = self._make_layer(block, 32, layer_blocks, 2)

        self.out_dim = 64 * block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_1_3x3(x)  # [bs, 16, 32, 32]
        x = F.relu(self.bn_1(x), inplace=True)

        x_1 = self.stage_1(x)  # [bs, 16, 32, 32]
        x_2 = self.stage_2(x_1)  # [bs, 32, 16, 16]
        return x_2
    
class SpecializedResNet_cifar(nn.Module):
    def __init__(self, block, depth, inplanes=32, feature_dim=64):
        super(SpecializedResNet_cifar, self).__init__()
        self.inplanes = inplanes
        self.feature_dim = feature_dim
        layer_blocks = (depth - 2) // 6
        self.final_stage = self._make_layer(block, 64, layer_blocks, 2)
        self.avgpool = nn.AvgPool2d(8)
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()
    
    def _make_layer(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    
    def forward(self, base_feature_map):
        final_feature_map = self.final_stage(base_feature_map)
        pooled = self.avgpool(final_feature_map)
        features = pooled.view(pooled.size(0), -1) #bs x 64
        return features

#For cifar & MEMO
def get_resnet8_a2fc():
    basenet = GeneralizedResNet_cifar(ResNetBasicblock,8)
    adaptivenet = SpecializedResNet_cifar(ResNetBasicblock,8)
    return basenet,adaptivenet

def get_resnet14_a2fc():
    basenet = GeneralizedResNet_cifar(ResNetBasicblock,14)
    adaptivenet = SpecializedResNet_cifar(ResNetBasicblock,14)
    return basenet,adaptivenet

def get_resnet20_a2fc():
    basenet = GeneralizedResNet_cifar(ResNetBasicblock,20)
    adaptivenet = SpecializedResNet_cifar(ResNetBasicblock,20)
    return basenet,adaptivenet

def get_resnet26_a2fc():
    basenet = GeneralizedResNet_cifar(ResNetBasicblock,26)
    adaptivenet = SpecializedResNet_cifar(ResNetBasicblock,26)
    return basenet,adaptivenet

def get_resnet32_a2fc():
    basenet = GeneralizedResNet_cifar(ResNetBasicblock,32)
    adaptivenet = SpecializedResNet_cifar(ResNetBasicblock,32)
    return basenet,adaptivenet


