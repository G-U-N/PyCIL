'''
For MEMO implementations of ImageNet-ConvNet
Reference:
https://github.com/wangkiw/ICLR23-MEMO/blob/main/convs/conv_imagenet.py
'''
import torch.nn as nn
import torch

# for imagenet
def first_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
    
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class ConvNet(nn.Module):
    def __init__(self, x_dim=3, hid_dim=128, z_dim=512):
        super().__init__()
        self.block1 = first_block(x_dim, hid_dim)
        self.block2 = conv_block(hid_dim, hid_dim)
        self.block3 = conv_block(hid_dim, hid_dim)
        self.block4 = conv_block(hid_dim, z_dim)
        self.avgpool = nn.AvgPool2d(7)
        self.out_dim = 512

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.avgpool(x)
        features = x.view(x.shape[0], -1)
        
        return {
            "features": features
        }

class GeneralizedConvNet(nn.Module):
    def __init__(self, x_dim=3, hid_dim=128, z_dim=512):
        super().__init__()
        self.block1 = first_block(x_dim, hid_dim)
        self.block2 = conv_block(hid_dim, hid_dim)
        self.block3 = conv_block(hid_dim, hid_dim)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

class SpecializedConvNet(nn.Module):
    def __init__(self, hid_dim=128,z_dim=512):
        super().__init__()
        self.block4 = conv_block(hid_dim, z_dim)
        self.avgpool = nn.AvgPool2d(7)
        self.feature_dim = 512
        
    def forward(self, x):
        x = self.block4(x)
        x = self.avgpool(x)
        features = x.view(x.shape[0], -1)
        return features
    
def conv4():
    model = ConvNet()
    return model

def conv_a2fc_imagenet():
    _base = GeneralizedConvNet()
    _adaptive_net = SpecializedConvNet()
    return _base, _adaptive_net