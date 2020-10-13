import torch
import torch.nn as nn
"""vgg in pytorch
[1] Karen Simonyan, Andrew Zisserman
    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VGG(nn.Module):

    def __init__(self, features):
        super().__init__()
        self.features = features
        self.out_dim = 512
        '''
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )
        '''
    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        # output = self.classifier(output)

        return output


def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for i in range(len(cfg)-2):
        # print(cfg[i])
        if cfg[i] == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, cfg[i], kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(cfg[i])]

        layers += [nn.ReLU(inplace=True)]
        input_channel = cfg[i]

    # Last conv layer without ReLU
    # print(cfg[-2])
    layers += [nn.Conv2d(input_channel, cfg[-2], kernel_size=3, padding=1)]
    if batch_norm:
        layers += [nn.BatchNorm2d(cfg[-2])]
    input_channel = cfg[-2]

    # Last MaxPool2d
    # print(cfg[-1])
    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

    return nn.Sequential(*layers)


def vgg11_bn():
    print("modified VGG11 network")
    return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13_bn():
    print("modified VGG13 network")
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16_bn():
    print("modified VGG16 network")
    return VGG(make_layers(cfg['D'], batch_norm=True))


def vgg19_bn():
    print("modified VGG19 network")
    return VGG(make_layers(cfg['E'], batch_norm=True))


'''
model = vgg19_bn()
print(model(torch.randn((1, 3, 32, 32))))
'''
