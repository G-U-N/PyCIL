import copy
import logging
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.foster import FOSTER
from utils.toolkit import count_parameters, tensor2numpy, accuracy
from utils.inc_net import IncrementalNet
from scipy.spatial.distance import cdist
from models.base import BaseLearner
from models.icarl import iCaRL
from tqdm import tqdm
import torch.optim as optim


EPSILON = 1e-8
batch_size = 128
weight_decay = 2e-4
num_workers = 8


class RMMBase(BaseLearner):
    def __init__(self, args):
        self._args = args
        self._m_rate_list = args.get("m_rate_list", [])
        self._c_rate_list = args.get("c_rate_list", [])

    @property
    def samples_per_class(self):
        return int(self.memory_size // self._total_classes)

    @property
    def memory_size(self):
        if self._args["dataset"] == "cifar100":
            img_per_cls = 500
        else:
            img_per_cls = 1300

        if self._m_rate_list[self._cur_task] != 0:
            print(self._total_classes)
            self._memory_size = min(int(self._total_classes*img_per_cls-1),self._args["memory_size"] + int(
                self._m_rate_list[self._cur_task]
                * self._args["increment"]
                * img_per_cls
            ))
        return self._memory_size

    @property
    def new_memory_size(self):
        if self._args["dataset"] == "cifar100":
            img_per_cls = 500
        else:
            img_per_cls = 1300
        return int(
            (1 - self._m_rate_list[self._cur_task])
            * self._args["increment"]
            * img_per_cls
        )

    def build_rehearsal_memory(self, data_manager, per_class):
        self._reduce_exemplar(data_manager, per_class)
        self._construct_exemplar(data_manager, per_class)

    def _construct_exemplar(self, data_manager, m):
        if self._args["dataset"] == "cifar100":
            img_per_cls = 500
        else:
            img_per_cls = 1300
        ns = [
            min(img_per_cls,int(m * (1 - self._c_rate_list[self._cur_task]))),
            min(img_per_cls,int(m * (1 + self._c_rate_list[self._cur_task]))),
        ]
        logging.info(
            "Constructing exemplars...({} or {} per classes)".format(ns[0], ns[1])
        )

        all_cls_entropies = []
        ms = []
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            with torch.no_grad():
                cidx_cls_entropies = []
                for idx, (_, inputs, targets) in enumerate(idx_loader):
                    inputs, targets = inputs.to(self._device), targets.to(self._device)
                    logits = self._network(inputs)["logits"]
                    cross_entropy = (
                        F.cross_entropy(logits, targets, reduction="none")
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    cidx_cls_entropies.append(cross_entropy)
                # print(cidx_cls_entropies)
                cidx_cls_entropies = np.mean(np.concatenate(cidx_cls_entropies))
            all_cls_entropies.append(cidx_cls_entropies)
        entropy_median = np.median(all_cls_entropies)
        for the_entropy in all_cls_entropies:
            if the_entropy > entropy_median:
                ms.append(ns[0])
            else:
                ms.append(ns[1])

        logging.info(f"ms: {ms}")
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)
            # Select
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]
            for k in range(1, ms[class_idx - self._known_classes] + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(ms[class_idx - self._known_classes], class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean


class RMM_iCaRL(
    RMMBase, iCaRL
):  # RMM Base is supposed to be prior to the orginal method.
    def __init__(self, args):
        RMMBase.__init__(self, args)
        iCaRL.__init__(self, args)

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
            m_rate=self._m_rate_list[self._cur_task] if self._cur_task > 0 else None,
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module


class RMM_FOSTER(RMMBase, FOSTER):
    def __init__(self, args):
        RMMBase.__init__(self, args)
        FOSTER.__init__(self, args)

    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1
        if self._cur_task > 1:
            self._network = self._snet
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        self._network_module_ptr = self._network
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        if self._cur_task > 0:
            for p in self._network.convnets[0].parameters():
                p.requires_grad = False
            for p in self._network.oldfc.parameters():
                p.requires_grad = False

        logging.info("All params: {}".format(count_parameters(self._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(self._network, True))
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
            m_rate=self._m_rate_list[self._cur_task] if self._cur_task > 0 else None,
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args["batch_size"],
            shuffle=True,
            num_workers=self.args["num_workers"],
            pin_memory=True,
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.args["batch_size"],
            shuffle=False,
            num_workers=self.args["num_workers"],
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
