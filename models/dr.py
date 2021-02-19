import logging
import numpy as np
import torch
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy

EPSILON = 1e-8

# CIFAR100, ResNet32
epochs_expert = 100
lrate_expert = 0.1
milestones_expert = [50, 70]
lrate_decay_expert = 0.1

epochs = 100
lrate = 0.5
milestones = [50, 70]
lrate_decay = 0.2

batch_size = 128
T1 = 2
T2 = 2
weight_decay = 1e-5
num_workers = 4


class DR(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args['convnet_type'], False)

        self.convnet_type = args['convnet_type']
        self.expert = None

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self.task_size = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + self.task_size
        self._network.update_fc(self._total_classes)
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        # Loader
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train', appendent=self._get_memory())
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        expert_train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                        source='train', mode='train')
        self.expert_train_loader = DataLoader(expert_train_dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)
        expert_test_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                       source='test', mode='test')
        self.expert_test_loader = DataLoader(expert_test_dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)

        # Procedure
        logging.info('Training the expert CNN...')
        self._train_expert(self.expert_train_loader, self.expert_test_loader)
        if self._cur_task == 0:
            self._network = self.expert.copy()
        else:
            self.expert = self.expert.freeze()
            logging.info('Training the updated CNN...')
            self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)
        optimizer = optim.SGD(self._network.parameters(), lr=lrate, momentum=0.9, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=lrate_decay)

        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)['logits']
                exp_logits = self.expert(inputs)['logits']
                old_logits = self._old_network(inputs)['logits']

                # Distillation
                dist_term = _KD_loss(logits[:, self._known_classes:], exp_logits, T1)
                # Retrospection
                retr_term = _KD_loss(logits[:, :self._known_classes], old_logits, T2)

                loss = dist_term + retr_term
                losses += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # acc
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            # train_acc = self._compute_accuracy(self._network, train_loader)
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = 'Updated CNN => Epoch {}/{}, Loss {:.3f}, Train accy {:.2f}, Test accy {:.2f}'.format(
                epoch+1, epochs, losses/len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)

        logging.info(info)

    def _train_expert(self, train_loader, test_loader):
        self.expert = IncrementalNet(self.convnet_type, False)
        self.expert.update_fc(self.task_size)
        self.expert.to(self._device)
        optimizer = optim.SGD(self.expert.parameters(), lr=lrate_expert, momentum=0.9, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones_expert, gamma=lrate_decay_expert)

        prog_bar = tqdm(range(epochs_expert))
        for _, epoch in enumerate(prog_bar):
            self.expert.train()
            losses = 0.
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), (targets - self._known_classes).to(self._device)
                logits = self.expert(inputs)['logits']

                loss = F.cross_entropy(logits, targets)
                losses += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()
            train_acc = self._compute_accuracy(self.expert, train_loader, self._known_classes)
            test_acc = self._compute_accuracy(self.expert, test_loader, self._known_classes)
            info = 'Expert CNN => Epoch {}/{}, Loss {:.3f}, Train accy {:.2f}, Test accy {:.2f}'.format(
                epoch+1, epochs_expert, losses/len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)

        logging.info(info)

    def _compute_accuracy(self, model, loader, offset=0):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            targets -= offset
            with torch.no_grad():
                outputs = model(inputs)['logits']
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct)*100 / total, decimals=2)


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred/T, dim=1)
    soft = torch.softmax(soft/T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / soft.shape[0]
