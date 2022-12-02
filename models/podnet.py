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

epochs = 160
lrate = 0.1
ft_epochs = 20
ft_lrate = 0.005
batch_size = 128
lambda_c_base = 5
lambda_f_base = 1
nb_proxy = 10
weight_decay = 5e-4
num_workers = 4

"""
Distillation losses: POD-flat (lambda_f=1) + POD-spatial (lambda_c=5)
NME results are shown.
The reproduced results are not in line with the reported results.
Maybe I missed something...
+--------------------+--------------------+--------------------+--------------------+
|     Classifier     |       Steps        |    Reported (%)    |   Reproduced (%)   |
+--------------------+--------------------+--------------------+--------------------+
|    Cosine (k=1)    |         50         |       56.69        |       55.49        |
+--------------------+--------------------+--------------------+--------------------+
|    LSC-CE (k=10)   |         50         |       59.86        |       55.69        |
+--------------------+--------------------+--------------------+--------------------+
|   LSC-NCA (k=10)   |         50         |       61.40        |       56.50        |
+--------------------+--------------------+--------------------+--------------------+
|    LSC-CE (k=10)   |         25         |       -----        |       59.16        |
+--------------------+--------------------+--------------------+--------------------+
|   LSC-NCA (k=10)   |         25         |       62.71        |       59.79        |
+--------------------+--------------------+--------------------+--------------------+
|    LSC-CE (k=10)   |         10         |       -----        |       62.59        |
+--------------------+--------------------+--------------------+--------------------+
|   LSC-NCA (k=10)   |         10         |       64.03        |       62.81        |
+--------------------+--------------------+--------------------+--------------------+
|    LSC-CE (k=10)   |         5          |       -----        |       64.16        |
+--------------------+--------------------+--------------------+--------------------+
|   LSC-NCA (k=10)   |         5          |       64.48        |       64.37        |
+--------------------+--------------------+--------------------+--------------------+
"""


class PODNet(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = CosineIncrementalNet(
            args, pretrained=False, nb_proxy=nb_proxy
        )
        self._class_means = None

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self.task_size = self._total_classes - self._known_classes
        self._network.update_fc(self._total_classes, self._cur_task)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        train_dset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        test_dset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.train_loader = DataLoader(
            train_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        self.test_loader = DataLoader(
            test_dset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        self._train(data_manager, self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)

    def _train(self, data_manager, train_loader, test_loader):
        if self._cur_task == 0:
            self.factor = 0
        else:
            self.factor = math.sqrt(
                self._total_classes / (self._total_classes - self._known_classes)
            )
        logging.info("Adaptive factor: {}".format(self.factor))

        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        if self._cur_task == 0:
            network_params = self._network.parameters()
        else:
            ignored_params = list(map(id, self._network.fc.fc1.parameters()))
            base_params = filter(
                lambda p: id(p) not in ignored_params, self._network.parameters()
            )
            network_params = [
                {"params": base_params, "lr": lrate, "weight_decay": weight_decay},
                {
                    "params": self._network.fc.fc1.parameters(),
                    "lr": 0,
                    "weight_decay": 0,
                },
            ]
        optimizer = optim.SGD(
            network_params, lr=lrate, momentum=0.9, weight_decay=weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=epochs
        )
        self._run(train_loader, test_loader, optimizer, scheduler, epochs)

        if self._cur_task == 0:
            return
        logging.info(
            "Finetune the network (classifier part) with the undersampled dataset!"
        )
        if self._fixed_memory:
            finetune_samples_per_class = self._memory_per_class
            self._construct_exemplar_unified(data_manager, finetune_samples_per_class)
        else:
            finetune_samples_per_class = self._memory_size // self._known_classes
            self._reduce_exemplar(data_manager, finetune_samples_per_class)
            self._construct_exemplar(data_manager, finetune_samples_per_class)

        finetune_train_dataset = data_manager.get_dataset(
            [], source="train", mode="train", appendent=self._get_memory()
        )
        finetune_train_loader = DataLoader(
            finetune_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        logging.info(
            "The size of finetune dataset: {}".format(len(finetune_train_dataset))
        )

        ignored_params = list(map(id, self._network.fc.fc1.parameters()))
        base_params = filter(
            lambda p: id(p) not in ignored_params, self._network.parameters()
        )
        network_params = [
            {"params": base_params, "lr": ft_lrate, "weight_decay": weight_decay},
            {"params": self._network.fc.fc1.parameters(), "lr": 0, "weight_decay": 0},
        ]
        optimizer = optim.SGD(
            network_params, lr=ft_lrate, momentum=0.9, weight_decay=weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=ft_epochs
        )
        self._run(finetune_train_loader, test_loader, optimizer, scheduler, ft_epochs)

        if self._fixed_memory:
            self._data_memory = self._data_memory[
                : -self._memory_per_class * self.task_size
            ]
            self._targets_memory = self._targets_memory[
                : -self._memory_per_class * self.task_size
            ]
            assert (
                len(
                    np.setdiff1d(
                        self._targets_memory, np.arange(0, self._known_classes)
                    )
                )
                == 0
            ), "Exemplar error!"

    def _run(self, train_loader, test_loader, optimizer, scheduler, epk):
        for epoch in range(1, epk + 1):
            self._network.train()
            lsc_losses = 0.0
            spatial_losses = 0.0
            flat_losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs = self._network(inputs)
                logits = outputs["logits"]
                features = outputs["features"]
                fmaps = outputs["fmaps"]
                lsc_loss = nca(logits, targets)

                spatial_loss = 0.0
                flat_loss = 0.0
                if self._old_network is not None:
                    with torch.no_grad():
                        old_outputs = self._old_network(inputs)
                    old_features = old_outputs["features"]
                    old_fmaps = old_outputs["fmaps"]
                    flat_loss = (
                        F.cosine_embedding_loss(
                            features,
                            old_features.detach(),
                            torch.ones(inputs.shape[0]).to(self._device),
                        )
                        * self.factor
                        * lambda_f_base
                    )
                    spatial_loss = (
                        pod_spatial_loss(fmaps, old_fmaps) * self.factor * lambda_c_base
                    )

                loss = lsc_loss + flat_loss + spatial_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                lsc_losses += lsc_loss.item()
                spatial_losses += (
                    spatial_loss.item() if self._cur_task != 0 else spatial_loss
                )
                flat_losses += flat_loss.item() if self._cur_task != 0 else flat_loss

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler is not None:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info1 = "Task {}, Epoch {}/{} (LR {:.5f}) => ".format(
                self._cur_task, epoch, epk, optimizer.param_groups[0]["lr"]
            )
            info2 = "LSC_loss {:.2f}, Spatial_loss {:.2f}, Flat_loss {:.2f}, Train_acc {:.2f}, Test_acc {:.2f}".format(
                lsc_losses / (i + 1),
                spatial_losses / (i + 1),
                flat_losses / (i + 1),
                train_acc,
                test_acc,
            )
            logging.info(info1 + info2)


def pod_spatial_loss(old_fmaps, fmaps, normalize=True):
    """
    a, b: list of [bs, c, w, h]
    """
    loss = torch.tensor(0.0).to(fmaps[0].device)
    for i, (a, b) in enumerate(zip(old_fmaps, fmaps)):
        assert a.shape == b.shape, "Shape error"

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


def nca(
    similarities,
    targets,
    class_weights=None,
    focal_gamma=None,
    scale=1.0,
    margin=0.6,
    exclude_pos_denominator=True,
    hinge_proxynca=False,
    memory_flags=None,
):
    margins = torch.zeros_like(similarities)
    margins[torch.arange(margins.shape[0]), targets] = margin
    similarities = scale * (similarities - margin)

    if exclude_pos_denominator:
        similarities = similarities - similarities.max(1)[0].view(-1, 1)

        disable_pos = torch.zeros_like(similarities)
        disable_pos[torch.arange(len(similarities)), targets] = similarities[
            torch.arange(len(similarities)), targets
        ]

        numerator = similarities[torch.arange(similarities.shape[0]), targets]
        denominator = similarities - disable_pos

        losses = numerator - torch.log(torch.exp(denominator).sum(-1))
        if class_weights is not None:
            losses = class_weights[targets] * losses

        losses = -losses
        if hinge_proxynca:
            losses = torch.clamp(losses, min=0.0)

        loss = torch.mean(losses)
        return loss

    return F.cross_entropy(
        similarities, targets, weight=class_weights, reduction="mean"
    )
