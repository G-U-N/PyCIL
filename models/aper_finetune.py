"""
Re-implementation of APER-Finetune (https://arxiv.org/abs/2303.07338) without pre-trained weights.
Note: this method was initially designed for PTMs, whereas it has been slightly modified here to adapt to the train-from-scratch setting. 
Please refer to the original implementation (https://github.com/zhoudw-zdw/RevisitingCIL) if you are using pre-trained weights.
"""

import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet, SimpleCosineIncrementalNet, MultiBranchCosineIncrementalNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy
import copy


num_workers = 8


class APER_FINETUNE(BaseLearner):
    def __init__(self, args):
        super().__init__(args)

        self._network = SimpleCosineIncrementalNet(args, False)
        self.batch_size = args.get("batch_size", 128)
        self.init_lr = args.get("init_lr", 0.01)
        self.finetune_lr = args.get("finetune_lr", 0.001)

        self.init_weight_decay = args.get("init_weight_decay", 0.0005)
        self.weight_decay = args.get("weight_decay", 0.005)
        self.min_lr = args.get('min_lr', 1e-8)
        self.args = args

        self.trained_epoch = args.get('trained_epoch', 50)
        self.tuned_epoch = args.get('tuned_epoch', 20)
        self.trained_model = None

    def after_task(self):
        self._known_classes = self._total_classes

    def replace_fc(self, trainloader, model, args):
        # replace fc.weight with the embedding average of train data
        model = model.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_, data, label) = batch
                data = data.cuda()
                label = label.cuda()
                embedding = model(data)['features']
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        class_list = np.unique(self.train_dataset.labels)
        proto_list = []
        for class_index in class_list:
            # print('Replacing...',class_index)
            data_index = (label_list == class_index).nonzero().squeeze(-1)
            embedding = embedding_list[data_index]
            proto = embedding.mean(0)
            self._network.fc.weight.data[class_index] = proto
        return model

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train",
                                                 mode="train", )
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                              source="train", mode="test", )
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=self.batch_size,
                                                    shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader, self.train_loader_for_protonet)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader, train_loader_for_protonet):

        self._network.to(self._device)

        if self._cur_task == 0:
            if self.args['optimizer'] == 'sgd':
                optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.init_lr,
                                      weight_decay=self.init_weight_decay)
            elif self.args['optimizer'] == 'adam':
                optimizer = optim.AdamW(self._network.parameters(), lr=self.init_lr,
                                        weight_decay=self.init_weight_decay)

            total_params = sum(p.numel() for p in self._network.parameters())
            print(f'{total_params:,} total parameters.')
            total_trainable_params = sum(
                p.numel() for p in self._network.parameters() if p.requires_grad)
            print(f'{total_trainable_params:,} training parameters.')

            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trained_epoch,
                                                             eta_min=self.min_lr)
            self._init_train(train_loader, test_loader, optimizer, scheduler, self.trained_epoch)
            self.replace_fc(train_loader_for_protonet, self._network, None)
            self.trained_model = copy.deepcopy(self._network.cpu())

            self.construct_dual_branch_network()

            return

        elif self._cur_task == 1:
            self._network = SimpleCosineIncrementalNet(self.args, False)
            self._network.regenerate_fc(self.args['init_cls'])  # to be compatible with trained_model
            self._network.to(self._device)
            msg = self._network.load_state_dict(self.trained_model.state_dict(), strict=False)
            logging.info('INFO -- state dict loaded', msg)
            self._network.regenerate_fc(self.args['increment'])
            logging.info('Fully finetuning ...')

            if self.args['optimizer'] == 'sgd':
                optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.finetune_lr,
                                      weight_decay=self.weight_decay)
            elif self.args['optimizer'] == 'adam':
                optimizer = optim.AdamW(self._network.parameters(), lr=self.finetune_lr, weight_decay=self.weight_decay)

            total_trainable_params = sum(
                p.numel() for p in self._network.parameters() if p.requires_grad)
            print(f'{total_trainable_params:,} training parameters.')

            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.tuned_epoch,
                                                             eta_min=self.min_lr)
            self._init_train(train_loader, test_loader, optimizer, scheduler, self.tuned_epoch)
            self.construct_dual_branch_network()

        self.replace_fc(train_loader_for_protonet, self._network, None)

    def construct_dual_branch_network(self):
        if self._cur_task == 0:
            network = MultiBranchCosineIncrementalNet(self.args, False)
            network.construct_dual_branch_network(self._network, self._network, self.args['init_cls'])
        else:
            network = MultiBranchCosineIncrementalNet(self.args, False)
            self.trained_model = self.trained_model.to(self._device)
            network.construct_dual_branch_network(self.trained_model, self._network,
                                                  self.args['init_cls'] + self.args['increment'])

        network.fc.weight.data[:self.args['init_cls'], :] = self.trained_model.fc.weight.data.repeat(1, 2)
        self._network = network.to(self._device)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler, epc):
        prog_bar = tqdm(range(epc))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                if self._cur_task == 1:
                    targets -= self.args['init_cls']
                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                # print('preds', preds)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                epc,
                losses / len(train_loader),
                train_acc,
                # test_acc,
            )
            prog_bar.set_description(info)

        logging.info(info)
