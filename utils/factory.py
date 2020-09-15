from models.icarl import iCaRL
from models.end2end import End2End
from models.dr import DR
from models.ucir import UCIR


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
