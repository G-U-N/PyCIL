'''
For MEMO implementations of CIFAR-ConvNet
Reference:
https://github.com/wangkiw/ICLR23-MEMO/blob/main/convs/conv_cifar.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

# for cifar
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class ConvNet2(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.out_dim = 64
        self.avgpool = nn.AvgPool2d(8)
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.avgpool(x)
        features = x.view(x.shape[0], -1)
        return {
            "features":features
        }
        
class GeneralizedConvNet2(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
        )

    def forward(self, x):
        base_features = self.encoder(x)
        return base_features
    
class SpecializedConvNet2(nn.Module):
    def __init__(self,hid_dim=64,z_dim=64):
        super().__init__()
        self.feature_dim = 64
        self.avgpool = nn.AvgPool2d(8)
        self.AdaptiveBlock = conv_block(hid_dim,z_dim)
    
    def forward(self,x):
        base_features = self.AdaptiveBlock(x)
        pooled = self.avgpool(base_features)
        features = pooled.view(pooled.size(0),-1)
        return features

def conv2():
    return ConvNet2()

def get_conv_a2fc():
    basenet = GeneralizedConvNet2()
    adaptivenet = SpecializedConvNet2()
    return basenet,adaptivenet

if __name__ == '__main__':
    a, b = get_conv_a2fc()
    _base = sum(p.numel() for p in a.parameters())
    _adap = sum(p.numel() for p in b.parameters())
    print(f"conv :{_base+_adap}")
    
    conv2 = conv2()
    conv2_sum = sum(p.numel() for p in conv2.parameters())
    print(f"conv2 :{conv2_sum}")