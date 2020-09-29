import logging
import numpy as np
import torch
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet
from models.base import BaseLearner
from utils.toolkit import accuracy

EPSILON = 1e-8

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
memory_size = 2000
T = 2


class End2End(BaseLearner):

    def __init__(self, args):
        super().__init__()
        self._network = IncrementalNet(args['convnet_type'], False)
        self._device = args['device']
        self._seen_classes = []

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes

    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        accy = accuracy(y_pred, y_true, self._known_classes)

        return accy

    def incremental_train(self, data_manager):
        self._cur_task += 1
        task_size = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + task_size
        self._network.update_fc(self._total_classes)
        self._seen_classes.append(task_size)
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        # Loader
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train', appendent=self._get_memory())
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # Procedure
        self._train(data_manager, self.train_loader, self.test_loader)
        self._reduce_exemplar(data_manager, memory_size//self._total_classes)
        self._construct_exemplar(data_manager, memory_size//self._total_classes)

    def _train(self, data_manager, train_loader, test_loader):
        self._network.to(self._device)
        if self._cur_task == 0:
            optimizer = optim.SGD(self._network.parameters(), lr=lrate_init, momentum=0.9, weight_decay=1e-3)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones_init,
                                                       gamma=lrate_decay)
            self._is_finetuning = False
            self._run(self.train_loader, self.test_loader, epochs_init, optimizer, scheduler, 'Training')
            return

        # New + exemplars
        optimizer = optim.SGD(self._network.parameters(), lr=lrate, momentum=0.9, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
        self._is_finetuning = False
        self._run(self.train_loader, self.test_loader, epochs, optimizer, scheduler, 'Training')

        # Finetune
        samples_per_class = memory_size//self._known_classes
        self._reduce_exemplar(data_manager, samples_per_class)
        self._construct_exemplar(data_manager, samples_per_class)
        self._old_network = self._network.copy().freeze()
        finetune_train_dataset = data_manager.get_dataset([], source='train', mode='train',
                                                          appendent=self._get_memory())
        finetune_train_loader = DataLoader(finetune_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        optimizer = optim.SGD(self._network.parameters(), lr=lrate_finetune, momentum=0.9, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones_finetune,
                                                   gamma=lrate_decay)
        self._is_finetuning = True
        self._run(finetune_train_loader, self.test_loader, epochs_finetune, optimizer, scheduler, 'Finetuning')

    def _run(self, train_loader, test_loader, epochs_, optimizer, scheduler, process):
        prog_bar = tqdm(range(epochs_))
        for _, epoch in enumerate(prog_bar, start=1):
            self._network.train()
            losses = 0.
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)

                # CELoss
                clf_loss = F.cross_entropy(logits, targets)

                if self._cur_task == 0:
                    distill_loss = torch.zeros(1, device=self._device)
                else:
                    finetuning_task = (self._cur_task + 1) if self._is_finetuning else self._cur_task
                    distill_loss = 0.
                    old_logits = self._old_network(inputs)
                    for i in range(1, finetuning_task+1):
                        lo = sum(self._seen_classes[:i-1])
                        hi = sum(self._seen_classes[:i])
                        distill_loss += F.binary_cross_entropy(
                            F.softmax(logits[:, lo:hi] / T, dim=1),
                            F.softmax(old_logits[:, lo:hi] / T, dim=1)
                        )

                loss = clf_loss + distill_loss
                losses += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()
            train_acc = self._compute_accuracy(self._network, train_loader)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info1 = '{} => '.format(process)
            info2 = 'Task {}, Epoch {}/{}, Loss {:.3f}, Train_accy {:.3f}, Test_accy {:.3f}'.format(
                self._cur_task, epoch+1, epochs_, losses/len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info1 + info2)

        logging.info(info1 + info2)
