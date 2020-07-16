from convs.cifar_resnet import resnet32
from convs.modified_cifar_resnet import resnet32 as cosine_resnet32
from convs.modified_resnet import resnet50 as cosine_resnet50
from convs.resnet import resnet50
from models.icarl import iCaRL
from models.end2end import End2End
from models.dr import DR
from models.ucir import UCIR


def get_convnet(convnet_type, pretrained=False):
    name = convnet_type.lower()
    if name == 'resnet32':
        return resnet32()
    elif name == 'resnet50':
        return resnet50(pretrained=pretrained)
    elif name == 'cosine_resnet32':
        return cosine_resnet32()
    elif name == 'cosine_resnet50':
        return cosine_resnet50()
    else:
        raise NotImplementedError('Unknown type {}'.format(convnet_type))


def get_model(model_name, args):
    name = model_name.lower()
    if name == 'icarl':
        return iCaRL(args)
    elif name == 'end2end':
        return End2End(args)
    elif name == 'dr':
        return DR(args)
    elif name == 'ucir':
        return UCIR(args)
