import math
import logging
import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import CosineIncrementalNet
from utils.toolkit import tensor2numpy

# CIFAR100, ResNet32, 50 base
epochs = 160
lrate = 0.1
milestones = [80, 120]
lrate_decay = 0.1
batch_size = 128
lambda_c_base = 5
lambda_f_base = 1
nb_proxy = 10
weight_decay = 5e-4
num_workers = 4


class PODNet(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        self._network = CosineIncrementalNet(args['convnet_type'], pretrained=False, nb_proxy=10)
        self._class_means = None

    def after_task(self):
        # self.save_checkpoint('podnet')
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes, self._cur_task)
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        # Loader
        train_dset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                              mode='train', appendent=self._get_memory())
        test_dset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.test_loader = DataLoader(test_dset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # Procedure
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)

    def _train(self, train_loader, test_loader):
        '''
        if self._cur_task == 0:
            loaded_dict = torch.load('./podnet_0.pkl')
            self._network.load_state_dict(loaded_dict['model_state_dict'])
            self._network.to(self._device)
            return
        '''
        # Adaptive factor
        # Adaptive lambda = base * factor
        # According to the official code: factor = total_clases / task_size
        # Slightly different from the implementation in UCIR
        # But the effect is negligible
        if self._cur_task == 0:
            self.factor = 0
        else:
            self.factor = math.sqrt(self._total_classes / (self._total_classes - self._known_classes))
        logging.info('Adaptive factor: {}'.format(self.factor))

        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        # Fix the embedding of old classes
        if self._cur_task == 0:
            network_params = self._network.parameters()
        else:
            ignored_params = list(map(id, self._network.fc.fc1.parameters()))
            base_params = filter(lambda p: id(p) not in ignored_params, self._network.parameters())
            network_params = [{'params': base_params, 'lr': lrate, 'weight_decay': weight_decay},
                              {'params': self._network.fc.fc1.parameters(), 'lr': 0, 'weight_decay': 0}]
        optimizer = optim.SGD(network_params, lr=lrate, momentum=0.9, weight_decay=weight_decay)
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs)

        self._run(train_loader, test_loader, optimizer, scheduler)

    def _run(self, train_loader, test_loader, optimizer, scheduler):
        for epoch in range(1, epochs+1):
            self._network.train()
            lsc_losses = 0.  # CE loss
            spatial_losses = 0.  # width + height
            flat_losses = 0.  # embedding
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs = self._network(inputs)
                logits = outputs['logits']
                features = outputs['features']
                fmaps = outputs['fmaps']
                lsc_loss = F.cross_entropy(logits, targets)

                spatial_loss = 0.
                flat_loss = 0.
                if self._old_network is not None:
                    with torch.no_grad():
                        old_outputs = self._old_network(inputs)
                    old_features = old_outputs['features']
                    old_fmaps = old_outputs['fmaps']
                    flat_loss = F.cosine_embedding_loss(features, old_features.detach(),
                                                        torch.ones(inputs.shape[0]).to(
                                                            self._device)) * self.factor * lambda_f_base
                    spatial_loss = pod_spatial_loss(fmaps, old_fmaps) * self.factor * lambda_c_base

                loss = lsc_loss + flat_loss + spatial_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # record
                lsc_losses += lsc_loss.item()
                spatial_losses += spatial_loss.item() if self._cur_task != 0 else spatial_loss
                flat_losses += flat_loss.item() if self._cur_task != 0 else flat_loss

                # acc
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info1 = 'Task {}, Epoch {}/{} => '.format(self._cur_task, epoch, epochs)
            info2 = 'LSC_loss {:.2f}, Spatial_loss {:.2f}, Flat_loss {:.2f}, Train_acc {:.2f}, Test_acc {:.2f}'.format(
                lsc_losses/(i+1), spatial_losses/(i+1), flat_losses/(i+1), train_acc, test_acc)
            logging.info(info1 + info2)


def pod_spatial_loss(old_fmaps, fmaps, normalize=True):
    '''
    a, b: list of [bs, c, w, h]
    '''
    loss = torch.tensor(0.).to(fmaps[0].device)
    for i, (a, b) in enumerate(zip(old_fmaps, fmaps)):
        assert a.shape == b.shape, 'Shape error'

        a = torch.pow(a, 2)
        b = torch.pow(b, 2)

        a_h = a.sum(dim=3).view(a.shape[0], -1)  # [bs, c*w]
        b_h = b.sum(dim=3).view(b.shape[0], -1)  # [bs, c*w]
        a_w = a.sum(dim=2).view(a.shape[0], -1)  # [bs, c*h]
        b_w = b.sum(dim=2).view(b.shape[0], -1)  # [bs, c*h]

        a = torch.cat([a_h, a_w], dim=-1)
        b = torch.cat([b_h, b_w], dim=-1)

        if normalize:
            a = F.normalize(a, dim=1, p=2)
            b = F.normalize(b, dim=1, p=2)

        layer_loss = torch.mean(torch.frobenius_norm(a - b, dim=-1))
        loss += layer_loss

    return loss / len(fmaps)
