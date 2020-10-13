import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters


def train(args):
    seed_list = copy.deepcopy(args['seed'])
    device = copy.deepcopy(args['device'])

    for seed in seed_list:
        args['seed'] = seed
        args['device'] = device
        _train(args)


def _train(args):
    logfilename = '{}_{}_{}_{}_{}_{}_{}'.format(args['prefix'], args['seed'], args['model_name'], args['convnet_type'],
                                                args['dataset'], args['init_cls'], args['increment'])
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s] => %(message)s',
        handlers=[
            logging.FileHandler(filename=logfilename + '.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.info('Seed: {}'.format(args['seed']))
    logging.info('Model: {}'.format(args['model_name']))
    logging.info('Convnet: {}'.format(args['convnet_type']))
    logging.info('Dataset: {}'.format(args['dataset']))
    _set_device(args)
    data_manager = DataManager(args['dataset'], args['shuffle'], args['seed'], args['init_cls'], args['increment'])
    model = factory.get_model(args['model_name'], args)

    cnn_curve, ncm_curve = {'top1': [], 'top5': []}, {'top1': [], 'top5': []}
    for task in range(data_manager.nb_tasks):
        logging.info('All params: {}'.format(count_parameters(model._network)))
        logging.info('Trainable params: {}'.format(count_parameters(model._network, True)))
        model.incremental_train(data_manager)
        cnn_accy, ncm_accy = model.eval_task()
        model.after_task()

        logging.info('CNN: {}'.format(cnn_accy['grouped']))
        logging.info('NCM: {}'.format(ncm_accy['grouped']))

        cnn_curve['top1'].append(cnn_accy['top1'])
        cnn_curve['top5'].append(cnn_accy['top5'])

        ncm_curve['top1'].append(ncm_accy['top1'])
        ncm_curve['top5'].append(ncm_accy['top5'])

        logging.info('CNN top1 curve: {}'.format(cnn_curve['top1']))
        logging.info('CNN top5 curve: {}'.format(cnn_curve['top5']))
        logging.info('NCM top1 curve: {}'.format(ncm_curve['top1']))
        logging.info('NCM top5 curve: {}\n'.format(ncm_curve['top5']))


def _set_device(args):
    device_type = args['device']

    if device_type == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(device_type))

    args['device'] = device
