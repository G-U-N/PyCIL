import copy
import logging
import numpy as np
import torch
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet
from models.base import BaseLearner
from utils.toolkit import accuracy, tensor2numpy

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
memory_size = 2000
T1 = 2
T2 = 2


class DR(BaseLearner):
    def __init__(self, args):
        super().__init__()
        self._network = IncrementalNet(args['convnet_type'], False)
        self._device = args['device']

        self.convnet_type = args['convnet_type']
        self.expert = None

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes

    def eval_task(self):
        y_pred, y_true = self._eval_ncm(self.test_loader, self._class_means)
        accy = accuracy(y_pred, y_true, self._known_classes)

        return accy

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self.task_size = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + self.task_size
        self._network.update_fc(self._total_classes)
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        # Loader
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train', appendent=self._get_memory())
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        expert_train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                        source='train', mode='train')
        self.expert_train_loader = DataLoader(expert_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        expert_test_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                       source='test', mode='test')
        self.expert_test_loader = DataLoader(expert_test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # Procedure
        logging.info('Training the expert CNN...')
        self._train_expert(self.expert_train_loader, self.expert_test_loader)
        if self._cur_task == 0:
            self._network = self.expert.copy()
        else:
            self.expert = self.expert.freeze()
            logging.info('Training the updated CNN...')
            self._train(self.train_loader, self.test_loader)
        self._reduce_exemplar(data_manager, memory_size//self._total_classes)
        self._construct_exemplar(data_manager, memory_size//self._total_classes)

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        optimizer = optim.SGD(self._network.parameters(), lr=lrate, momentum=0.9, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=lrate_decay)

        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)
                exp_logits = self.expert(inputs)
                old_logits = self._old_network(inputs)

                # Distillation
                dist_term = _KD_loss(logits[:, self._known_classes:], exp_logits, T1)
                # Retrospection
                retr_term = _KD_loss(logits[:, :self._known_classes], old_logits, T2)

                loss = dist_term + retr_term
                losses += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()
            train_acc = self._compute_accuracy(self._network, train_loader)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = 'Updated CNN => Epoch {}/{}, Loss {:.3f}, Train accy {:.3f}, Test accy {:.3f}'.format(
                epoch+1, epochs, losses/len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)

        logging.info(info)

    def _train_expert(self, train_loader, test_loader):
        self.expert = IncrementalNet(self.convnet_type, False)
        self.expert.update_fc(self.task_size)
        self.expert.to(self._device)
        optimizer = optim.SGD(self.expert.parameters(), lr=lrate_expert, momentum=0.9, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones_expert, gamma=lrate_decay_expert)

        prog_bar = tqdm(range(epochs_expert))
        for _, epoch in enumerate(prog_bar):
            self.expert.train()
            losses = 0.
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), (targets - self._known_classes).to(self._device)
                logits = self.expert(inputs)

                loss = F.cross_entropy(logits, targets)
                losses += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()
            train_acc = self._compute_accuracy(self.expert, train_loader, self._known_classes)
            test_acc = self._compute_accuracy(self.expert, test_loader, self._known_classes)
            info = 'Expert CNN => Epoch {}/{}, Loss {:.3f}, Train accy {:.3f}, Test accy {:.3f}'.format(
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
                outputs = model(inputs)
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) / total, decimals=3)

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

                vectors = np.delete(vectors, i, axis=0)  # Remove it to avoid duplicative selection
                data = np.delete(data, i, axis=0)  # Remove it to avoid duplicative selection

            # uniques = np.unique(selected_exemplars, axis=0)
            # print('Unique elements: {}'.format(len(uniques)))
            exemplar_targets = np.full(m, class_idx)
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

        self._targets_memory = np.concatenate(self._targets_memory)


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred/T, dim=1)
    soft = torch.softmax(soft/T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / soft.shape[0]
