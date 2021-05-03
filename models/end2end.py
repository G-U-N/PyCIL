import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy

EPSILON = 1e-8

# ImageNet1000, ResNet18


# CIFAR100, ResNet32
epochs_init = 90
lrate_init = 0.1
milestones_init = [50, 60]

epochs = 60
lrate = 0.1
milestones = [40, 50]

epochs_finetune = 40
lrate_finetune = 0.01
milestones_finetune = [10, 20]

lrate_decay = 0.1
batch_size = 128
T = 2
weight_decay = 1e-3
num_workers = 4


class End2End(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args['convnet_type'], False)
        self._seen_classes = []

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self.task_size = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + self.task_size
        self._network.update_fc(self._total_classes)
        self._seen_classes.append(self.task_size)
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        # Loader
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train', appendent=self._get_memory())
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # Procedure
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(data_manager, self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, data_manager, train_loader, test_loader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)
        if self._cur_task == 0:
            optimizer = optim.SGD(self._network.parameters(), lr=lrate_init, momentum=0.9, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones_init,
                                                       gamma=lrate_decay)
            self._is_finetuning = False
            self._run(self.train_loader, self.test_loader, epochs_init, optimizer, scheduler, 'Training')
            return

        # New + exemplars
        optimizer = optim.SGD(self._network.parameters(), lr=lrate, momentum=0.9, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
        self._is_finetuning = False
        self._run(self.train_loader, self.test_loader, epochs, optimizer, scheduler, 'Training')

        # Finetune
        if self._fixed_memory:
            finetune_samples_per_class = self._memory_per_class
            self._construct_exemplar_unified(data_manager, finetune_samples_per_class)
        else:
            finetune_samples_per_class = self._memory_size//self._known_classes
            self._reduce_exemplar(data_manager, finetune_samples_per_class)
            self._construct_exemplar(data_manager, finetune_samples_per_class)

        if len(self._multiple_gpus) > 1:
            self._old_network = self._network.module.copy().freeze()
        else:
            self._old_network = self._network.copy().freeze()
        finetune_train_dataset = data_manager.get_dataset([], source='train', mode='train',
                                                          appendent=self._get_memory())
        finetune_train_loader = DataLoader(finetune_train_dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=num_workers)
        # Update all weights or only the weights of FC layer?
        # According to the experiment results, fine-tuning all weights is slightly better.
        optimizer = optim.SGD(self._network.parameters(), lr=lrate_finetune, momentum=0.9, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones_finetune,
                                                   gamma=lrate_decay)
        self._is_finetuning = True
        self._run(finetune_train_loader, self.test_loader, epochs_finetune, optimizer, scheduler, 'Finetuning')

        # Remove the temporary exemplars of new classes
        if self._fixed_memory:
            self._data_memory = self._data_memory[:-self._memory_per_class*self.task_size]
            self._targets_memory = self._targets_memory[:-self._memory_per_class*self.task_size]
            # Check
            assert len(np.setdiff1d(self._targets_memory, np.arange(0, self._known_classes))) == 0, 'Exemplar error!'

    def _run(self, train_loader, test_loader, epochs_, optimizer, scheduler, process):
        prog_bar = tqdm(range(epochs_))
        for _, epoch in enumerate(prog_bar, start=1):
            self._network.train()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)['logits']

                # CELoss
                clf_loss = F.cross_entropy(logits, targets)

                if self._cur_task == 0:
                    distill_loss = torch.zeros(1, device=self._device)
                else:
                    finetuning_task = (self._cur_task + 1) if self._is_finetuning else self._cur_task
                    distill_loss = 0.
                    old_logits = self._old_network(inputs)['logits']
                    for i in range(1, finetuning_task+1):
                        lo = sum(self._seen_classes[:i-1])
                        hi = sum(self._seen_classes[:i])

                        task_prob_new = F.softmax(logits[:, lo:hi], dim=1)
                        task_prob_old = F.softmax(old_logits[:, lo:hi], dim=1)

                        task_prob_new = task_prob_new ** (1 / T)
                        task_prob_old = task_prob_old ** (1 / T)

                        task_prob_new = task_prob_new / task_prob_new.sum(1).view(-1, 1)
                        task_prob_old = task_prob_old / task_prob_old.sum(1).view(-1, 1)

                        distill_loss += F.binary_cross_entropy(task_prob_new, task_prob_old)

                    distill_loss *= 1 / finetuning_task

                loss = clf_loss + distill_loss
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
            info1 = '{} => '.format(process)
            info2 = 'Task {}, Epoch {}/{}, Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task, epoch+1, epochs_, losses/len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info1 + info2)

        logging.info(info1 + info2)
