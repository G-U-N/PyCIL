import math
import logging
import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.modified_inc_net import ModifiedIncrementalNet
from utils.toolkit import accuracy, tensor2numpy, target2onehot

EPSILON = 1e-8

# CIFAR100, ResNet32
epochs = 160
lrate = 0.1
milestones = [80, 120]
lrate_decay = 0.1
batch_size = 128
samples_per_class = 20  # Save 20 samples per class!
lamda_base = 5
K = 2
margin = 0.5


class UCIR(BaseLearner):
    def __init__(self, args):
        super().__init__()
        self._network = ModifiedIncrementalNet(args['convnet_type'], True)
        self._device = args['device']
        self._class_means = None

    def save_checkpoint(self):
        self._network.cpu()
        save_dict = {
            'tasks': self._cur_task,
            'model_state_dict': self._network.state_dict(),
            'data_memory': self._data_memory,
            'targets_memory': self._targets_memory,
        }
        torch.save(save_dict, 'dict_{}.pkl'.format(self._cur_task))

    def after_task(self):
        # self.save_checkpoint()
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes

    def eval_task(self):
        y_pred, y_true = self._eval_ncm(self.test_loader, self._class_means)
        accy = accuracy(y_pred, y_true, self._known_classes)

        return accy

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes, self._cur_task)
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        # Loader
        train_dset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                              mode='train', appendent=self._get_memory())
        test_dset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=4)
        self.test_loader = DataLoader(test_dset, batch_size=batch_size, shuffle=False, num_workers=4)

        # Procedure
        self._train(self.train_loader, self.test_loader)
        self._construct_exemplar(data_manager, samples_per_class)

    def _train(self, train_loader, test_loader):
        '''
        if self._cur_task == 0:
            loaded_dict = torch.load('./dict_0.pkl')
            self._network.load_state_dict(loaded_dict['model_state_dict'])
            self._network.to(self._device)
            return
        '''

        # Adaptive lambda
        # The definition of adaptive lambda in paper and the official code repository is different.
        # Here we use the definition in official code repository.
        if self._cur_task == 0:
            self.lamda = 0
        else:
            self.lamda = lamda_base * math.sqrt(self._known_classes / (self._total_classes - self._known_classes))
        logging.info('Adaptive lambda: {}'.format(self.lamda))

        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        # Fix the embedding of old classes
        if self._cur_task == 0:
            network_params = self._network.parameters()
        else:
            ignored_params = list(map(id, self._network.fc.fc1.parameters()))
            base_params = filter(lambda p: id(p) not in ignored_params, self._network.parameters())
            network_params = [{'params': base_params, 'lr': lrate, 'weight_decay': 5e-5},
                              {'params': self._network.fc.fc1.parameters(), 'lr': 0, 'weight_decay': 0}]
        optimizer = optim.SGD(network_params, lr=lrate, momentum=0.9, weight_decay=5e-5)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)

        self._run(train_loader, test_loader, optimizer, scheduler)

    def _run(self, train_loader, test_loader, optimizer, scheduler):
        for epoch in range(1, epochs+1):
            self._network.train()
            ce_losses = 0.
            lf_losses = 0.
            is_losses = 0.
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)  # Final outputs after scaling  (bs, nb_classes)
                features = self._network.get_features()  # Features before fc layer  (bs, 64)
                ce_loss = F.cross_entropy(logits, targets)  # Cross entropy loss

                lf_loss = 0.  # Less forgetting loss
                is_loss = 0.  # Inter-class speration loss
                if self._old_network is not None:
                    old_logits = self._old_network(inputs)  # Final outputs after scaling
                    old_features = self._old_network.get_features()  # Features before fc layer
                    lf_loss = F.cosine_embedding_loss(features, old_features.detach(),
                                                      torch.ones(inputs.shape[0]).to(self._device)) * self.lamda

                    scores = self._network.fc.get_new_scores()  # Scores before scaling  (bs, nb_new)
                    old_scores = self._network.fc.get_old_scores()  # Scores before scaling  (bs, nb_old)
                    old_classes_mask = np.where(tensor2numpy(targets) < self._known_classes)[0]
                    if len(old_classes_mask) != 0:
                        scores = scores[old_classes_mask]  # (n, nb_new)
                        old_scores = old_scores[old_classes_mask]  # (n, nb_old)

                        # Ground truth targets
                        gt_targets = targets[old_classes_mask]  # (n)
                        old_bool_onehot = target2onehot(gt_targets, self._known_classes).type(torch.bool)
                        anchor_positive = torch.masked_select(old_scores, old_bool_onehot)  # (n)
                        anchor_positive = anchor_positive.view(-1, 1).repeat(1, K)  # (n, K)

                        # Top K hard
                        anchor_hard_negative = scores.topk(K, dim=1)[0]  # (n, K)

                        is_loss = F.margin_ranking_loss(anchor_positive, anchor_hard_negative,
                                                        torch.ones(K).to(self._device), margin=margin)

                loss = ce_loss + lf_loss + is_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                ce_losses += ce_loss.item()
                lf_losses += lf_loss.item() if self._cur_task != 0 else lf_loss
                is_losses += is_loss.item() if self._cur_task != 0 and len(old_classes_mask) != 0 else is_loss

            scheduler.step()
            train_acc = self._compute_accuracy(self._network, train_loader)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info1 = 'Task {}, Epoch {}/{} => '.format(self._cur_task, epoch, epochs)
            info2 = 'CE_loss {:.3f}, LF_loss {:.3f}, IS_loss {:.3f}, Train_accy {:.3f}, Test_accy {:.3f}'.format(
                ce_losses/(i+1), lf_losses/(i+1), is_losses/(i+1), train_acc, test_acc)
            logging.info(info1 + info2)

    def _construct_exemplar(self, data_manager, m):
        logging.info('Constructing exemplars for new classes...({} per classes)'.format(m))
        _class_means = np.zeros((self._total_classes, self._network.feature_dim))

        # Calculate the means of old classes with newly trained network
        for class_idx in range(self._known_classes):
            mask = np.where(self._targets_memory == class_idx)[0]
            class_data = [self._data_memory[i] for i in mask]
            class_targets = self._targets_memory[mask]

            class_dset = data_manager.get_dataset([], source='train', mode='test',
                                                  appendent=(class_data, class_targets))
            class_loader = DataLoader(class_dset, batch_size=batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        # Construct exemplars for new classes and calculate the means
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, class_dset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                 mode='test', ret_data=True)
            class_loader = DataLoader(class_dset, batch_size=batch_size, shuffle=False, num_workers=4)

            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []
            for k in range(1, m+1):
                S = np.sum(exemplar_vectors, axis=0)  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

                exemplar_vectors.append(vectors[i])
                selected_exemplars.append(data[i])

                vectors = np.delete(vectors, i, axis=0)  # Remove it to avoid duplicative selection
                data = np.delete(data, i, axis=0)  # Remove it to avoid duplicative selection

            exemplar_targets = np.full(m, class_idx).tolist()
            self._data_memory = self._data_memory + selected_exemplars
            self._targets_memory = np.concatenate((self._targets_memory, exemplar_targets)).astype(int)

            # Exemplar mean
            exemplar_dset = data_manager.get_dataset([], source='train', mode='test',
                                                     appendent=(selected_exemplars, exemplar_targets))
            exemplar_loader = DataLoader(exemplar_dset, batch_size=batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(exemplar_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        self._class_means = _class_means
