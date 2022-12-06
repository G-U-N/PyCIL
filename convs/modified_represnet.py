import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

__all__ = ['ResNet', 'resnet18_rep', 'resnet34_rep' ]


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)

class conv_block(nn.Module):

    def __init__(self, in_planes, planes, mode, stride=1):
        super(conv_block, self).__init__()
        self.conv = conv3x3(in_planes, planes, stride)
        self.mode = mode
        if mode == 'parallel_adapters':
            self.adapter = conv1x1(in_planes, planes, stride)

    
    def re_init_conv(self):
        nn.init.kaiming_normal_(self.adapter.weight, mode='fan_out', nonlinearity='relu')
        return 
    def forward(self, x):
        y = self.conv(x)
        if self.mode == 'parallel_adapters':
            y = y + self.adapter(x)

        return y


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, mode, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv_block(inplanes, planes, mode, stride)
        self.norm1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_block(planes, planes, mode)
        self.norm2 = nn.BatchNorm2d(planes)
        self.mode = mode

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=100, args = None):
        self.inplanes = 64
        super(ResNet, self).__init__()
        assert args is not None
        self.mode = args["mode"]
        
        if 'cifar' in args["dataset"]:
            self.conv1 = nn.Sequential(nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(self.inplanes), nn.ReLU(inplace=True))
            print("use cifar")
        elif 'imagenet' in args["dataset"]:
            if args["init_cls"] == args["increment"]:
                self.conv1 = nn.Sequential(
                    nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.BatchNorm2d(self.inplanes),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                )
            else:
                # Following PODNET implmentation
                self.conv1 = nn.Sequential(
                    nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(self.inplanes),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                )


        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.feature = nn.AvgPool2d(4, stride=1)
        self.out_dim = 512


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
            )
        layers = []
        layers.append(block(self.inplanes, planes, self.mode, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.mode))

        return nn.Sequential(*layers)

    def switch(self, mode='normal'):
        for name, module in self.named_modules():
            if hasattr(module, 'mode'):
                module.mode = mode
    def re_init_params(self):
        for name, module in self.named_modules():
            if hasattr(module, 're_init_conv'):
                module.re_init_conv()
    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        dim = x.size()[-1]
        pool = nn.AvgPool2d(dim, stride=1)
        x = pool(x)
        x = x.view(x.size(0), -1)
        return {"features": x}


def resnet18_rep(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet18'])
        now_state_dict        = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet34_rep(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet34'])
        now_state_dict        = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model