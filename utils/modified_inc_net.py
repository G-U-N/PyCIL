import copy
from torch import nn
from utils import factory
from convs.modified_linear import SplitCosineLinear, CosineLinear


class ModifiedIncrementalNet(nn.Module):

    def __init__(self, convnet_type, pretrained):
        super(ModifiedIncrementalNet, self).__init__()

        self.convnet = factory.get_convnet(convnet_type, pretrained)
        self.fc = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def get_features_fn(self, module, inputs, outputs):
        self.features = inputs[0]

    def get_features(self):
        return self.features

    def extract_vector(self, x):
        return self.convnet(x)

    def forward(self, x):
        x = self.convnet(x)
        logits = self.fc(x)

        return logits

    def update_fc(self, nb_classes, task_num):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            self.features_hook.remove()
            if task_num == 1:
                fc.fc1.weight.data = self.fc.weight.data
                fc.sigma.data = self.fc.sigma.data
            else:
                prev_out_features1 = self.fc.fc1.out_features
                fc.fc1.weight.data[:prev_out_features1] = self.fc.fc1.weight.data
                fc.fc1.weight.data[prev_out_features1:] = self.fc.fc2.weight.data
                fc.sigma.data = self.fc.sigma.data

        del self.fc
        self.fc = fc
        self.features_hook = self.fc.register_forward_hook(self.get_features_fn)

    def generate_fc(self, in_dim, out_dim):
        if self.fc is None:
            fc = CosineLinear(in_dim, out_dim)
        else:
            prev_out_features = self.fc.out_features
            fc = SplitCosineLinear(in_dim, prev_out_features, out_dim - prev_out_features)

        return fc

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self
