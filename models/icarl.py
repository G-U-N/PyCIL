import copy
import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import target2onehot, accuracy

EPSILON = 1e-8

# CIFAR100, ResNet32
epochs = 70
lrate = 2.0
milestones = [49, 63]
lrate_decay = 0.2
batch_size = 64
memory_size = 2000


class iCaRL(BaseLearner):

    def __init__(self, args):
        super().__init__()
        self._network = IncrementalNet(args['convnet_type'], False)
        self._device = args['device']

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes

    def eval_task(self):
        y_pred, y_true = self._eval_ncm(self.test_loader, self._class_means)
        accy = accuracy(y_pred, y_true, self._known_classes)

        return accy

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        # Loader
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train', appendent=self._get_memory())
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # Procedure
        self._train(self.train_loader, self.test_loader)
        self._reduce_exemplar(data_manager, memory_size//self._total_classes)
        self._construct_exemplar(data_manager, memory_size//self._total_classes)

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        optimizer = optim.SGD(self._network.parameters(), lr=lrate, momentum=0.9, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
        self._update_representation(train_loader, test_loader, optimizer, scheduler)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)
                onehots = target2onehot(targets, self._total_classes)

                if self._old_network is None:
                    loss = F.binary_cross_entropy_with_logits(logits, onehots)
                else:
                    old_onehots = torch.sigmoid(self._old_network(inputs).detach())
                    new_onehots = onehots.clone()
                    new_onehots[:, :self._known_classes] = old_onehots
                    loss = F.binary_cross_entropy_with_logits(logits, new_onehots)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

            scheduler.step()
            train_acc = self._compute_accuracy(self._network, train_loader)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.3f}, Test_accy {:.3f}'.format(
                self._cur_task, epoch+1, epochs, losses/len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)

        logging.info(info)

    def _reduce_exemplar(self, data_manager, m):
        logging.info('Reducing exemplars...({} per classes)'.format(m))
        dummy_data, dummy_targets = copy.deepcopy(self._data_memory), copy.deepcopy(self._targets_memory)
        self._class_means = np.zeros((self._total_classes, self._network.feature_dim))
        self._data_memory, self._targets_memory = [], []

        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]
            dd = [dummy_data[i] for i in mask][:m]
            dt = dummy_targets[mask][:m]
            self._data_memory = self._data_memory + dd
            self._targets_memory.append(dt)

            # Exemplar mean
            idx_dataset = data_manager.get_dataset([], source='train', mode='test', appendent=(dd, dt))
            idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

        '''
        if len(self._data_memory) == 0:
            self._data_memory = []
            # self._data_memory = np.array(self._data_memory)
        else:
            self._data_memory = np.concatenate((self._data_memory))
        '''

    def _construct_exemplar(self, data_manager, m):
        logging.info('Constructing exemplars...({} per classes)'.format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                  mode='test', ret_data=True)
            idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]
            for k in range(1, m+1):
                S = np.sum(exemplar_vectors, axis=0)  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                exemplar_vectors.append(vectors[i])
                selected_exemplars.append(data[i])

            exemplar_targets = np.full(m, class_idx)
            '''
            selected_exemplars = np.array(selected_exemplars)  # [n, H, W, C]
            dummy = []
            for sample in self._data_memory:
                dummy.append(sample)
            for sample in selected_exemplars:
                dummy.append(sample)
            self._data_memory = np.array(dummy)
            '''
            self._data_memory = self._data_memory + selected_exemplars
            self._targets_memory.append(exemplar_targets)

            # Exemplar mean
            idx_dataset = data_manager.get_dataset([], source='train', mode='test',
                                                   appendent=(selected_exemplars, exemplar_targets))
            idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

        # self._data_memory = np.array(self._data_memory)
        self._targets_memory = np.concatenate(self._targets_memory)
