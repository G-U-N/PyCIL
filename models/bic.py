import logging
import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import IncrementalNetWithBias

# CIFAR100, ResNet32, 10 base
epochs = 250
lrate = 0.1
milestones = [100, 150, 200]
lrate_decay = 0.1
batch_size = 128
memory_size = 2000
split_ratio = 0.1
T = 2


class BiC(BaseLearner):
    def __init__(self, args):
        super().__init__()
        self._network = IncrementalNetWithBias(args['convnet_type'], False, bias_correction=True)
        self._device = args['device']
        self._class_means = None

    def after_task(self):
        # self.save_checkpoint(logfilename)
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        # Loader
        if self._cur_task >= 1:
            train_dset, val_dset = data_manager.get_dataset_with_split(np.arange(self._known_classes,
                                                                                 self._total_classes),
                                                                       source='train', mode='train',
                                                                       appendent=self._get_memory(),
                                                                       val_samples_per_class=int(
                                                                           split_ratio*memory_size/self._known_classes))
            self.val_loader = DataLoader(val_dset, batch_size=batch_size, shuffle=True, num_workers=4)
            logging.info('Stage1 dset: {}, Stage2 dset: {}'.format(len(train_dset), len(val_dset)))
            self.lamda = self._known_classes / self._total_classes
            logging.info('Lambda: {:.3f}'.format(self.lamda))
        else:
            train_dset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                  source='train', mode='train',
                                                  appendent=self._get_memory())
        test_dset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')

        self.train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=4)
        self.test_loader = DataLoader(test_dset, batch_size=batch_size, shuffle=False, num_workers=4)

        # Procedure
        self._stage1_training(self.train_loader, self.test_loader)
        if self._cur_task >= 1:
            self._stage2_bias_correction(self.val_loader, self.test_loader)

        # Exemplars
        self._reduce_exemplar(data_manager, memory_size//self._total_classes)
        self._construct_exemplar(data_manager, memory_size//self._total_classes)

    def _run(self, train_loader, test_loader, optimizer, scheduler, stage):
        for epoch in range(1, epochs+1):
            self._network.train()
            losses = 0.
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)['logits']

                if stage == 'training':
                    clf_loss = F.cross_entropy(logits, targets)
                    if self._old_network is not None:
                        old_logits = self._old_network(inputs)['logits'].detach()
                        hat_pai_k = F.softmax(old_logits / T, dim=1)
                        log_pai_k = F.log_softmax(logits[:, :self._known_classes] / T, dim=1)
                        distill_loss = -torch.mean(torch.sum(hat_pai_k * log_pai_k, dim=1))
                        loss = distill_loss * self.lamda + clf_loss * (1 - self.lamda)
                    else:
                        loss = clf_loss
                elif stage == 'bias_correction':
                    loss = F.cross_entropy(torch.softmax(logits, dim=1), targets)
                else:
                    raise NotImplementedError()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

            scheduler.step()
            train_acc = self._compute_accuracy(self._network, train_loader)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = '{} => Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.3f}, Test_accy {:.3f}'.format(
                stage, self._cur_task, epoch, epochs, losses/len(train_loader), train_acc, test_acc)
            logging.info(info)

    def _stage1_training(self, train_loader, test_loader):
        '''
        if self._cur_task == 0:
            loaded_dict = torch.load('./dict_0.pkl')
            self._network.load_state_dict(loaded_dict['model_state_dict'])
            self._network.to(self._device)
            return
        '''

        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        # Freeze bias layer and train stage1 layer
        ignored_params = list(map(id, self._network.bias_layers.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, self._network.parameters())
        network_params = [{'params': base_params, 'lr': lrate, 'weight_decay': 2e-4},
                          {'params': self._network.bias_layers.parameters(), 'lr': 0, 'weight_decay': 0}]
        optimizer = optim.SGD(network_params, lr=lrate, momentum=0.9, weight_decay=2e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)

        self._log_bias_params()
        self._run(train_loader, test_loader, optimizer, scheduler, stage='training')
        self._log_bias_params()

    def _stage2_bias_correction(self, val_loader, test_loader):
        self._network.to(self._device)

        # Freeze stage1 layer and train bias layer
        network_params = [{'params': self._network.bias_layers[-1].parameters(), 'lr': 0.001, 'weight_decay': 2e-4}]
        optimizer = optim.SGD(network_params, lr=lrate, momentum=0.9, weight_decay=2e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)

        self._log_bias_params()
        self._run(val_loader, test_loader, optimizer, scheduler, stage='bias_correction')
        self._log_bias_params()

    def _log_bias_params(self):
        logging.info('Parameters of bias layer:')
        params = self._network.get_bias_params()
        for i, param in enumerate(params):
            logging.info('{} => {:.3f}, {:.3f}'.format(i, param[0], param[1]))
