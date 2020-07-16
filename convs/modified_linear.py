'''
Reference:
https://github.com/hshustc/CVPR19_Incremental_Learning/blob/master/cifar100-class-incremental/modified_linear.py
'''
import math
import torch
from torch import nn
from torch.nn import functional as F


class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out = self.sigma * out
        return out


class SplitCosineLinear(nn.Module):
    def __init__(self, in_features, out_features1, out_features2, sigma=True):
        super(SplitCosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features1 + out_features2
        self.fc1 = CosineLinear(in_features, out_features1, False)
        self.fc2 = CosineLinear(in_features, out_features2, False)
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
            self.sigma.data.fill_(1)
        else:
            self.register_parameter('sigma', None)

        self.old_scores_hook = self.fc1.register_forward_hook(self.get_old_scores_before_scale_fn)
        self.new_scores_hook = self.fc2.register_forward_hook(self.get_new_scores_before_scale_fn)

    def get_old_scores_before_scale_fn(self, module, inputs, outputs):
        self.old_scores = outputs

    def get_new_scores_before_scale_fn(self, module, inputs, outputs):
        self.new_scores = outputs

    def get_old_scores(self):
        return self.old_scores

    def get_new_scores(self):
        return self.new_scores

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.fc2(x)

        out = torch.cat((out1, out2), dim=1)  # concatenate along the channel
        if self.sigma is not None:
            out = self.sigma * out

        return out
