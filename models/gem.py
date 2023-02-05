import logging
import numpy as np
from torch._C import device
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.inc_net import CosineIncrementalNet
from utils.toolkit import target2onehot, tensor2numpy
try:
    from quadprog import solve_qp
except:
    pass


EPSILON = 1e-8


init_epoch = 200
init_lr = 0.1
init_milestones = [60, 120, 170]
init_lr_decay = 0.1
init_weight_decay = 0.0005


epochs = 100
lrate = 0.1
milestones = [30, 60, 80]
lrate_decay = 0.1
batch_size = 128
weight_decay = 2e-4
num_workers = 4


class GEM(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)
        self.previous_data = None
        self.previous_label = None

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

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
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        if self._cur_task > 0:
            previous_dataset = data_manager.get_dataset(
                [], source="train", mode="train", appendent=self._get_memory()
            )

            self.previous_data = []
            self.previous_label = []
            for i in previous_dataset:
                _, data_, label_ = i
                self.previous_data.append(data_)
                self.previous_label.append(label_)
            self.previous_data = torch.stack(self.previous_data)
            self.previous_label = torch.tensor(self.previous_label)
        # Procedure
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        if self._cur_task == 0:
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=init_lr,
                weight_decay=init_weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = optim.SGD(
                self._network.parameters(),
                lr=lrate,
                momentum=0.9,
                weight_decay=weight_decay,
            )  # 1e-5
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=milestones, gamma=lrate_decay
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(init_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                )

            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(epochs))
        grad_numels = []
        for params in self._network.parameters():
            grad_numels.append(params.data.numel())
        G = torch.zeros((sum(grad_numels), self._cur_task + 1)).to(self._device)

        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                incremental_step = self._total_classes - self._known_classes
                for k in range(0, self._cur_task):
                    optimizer.zero_grad()
                    mask = torch.where(
                        (self.previous_label >= k * incremental_step)
                        & (self.previous_label < (k + 1) * incremental_step)
                    )[0]
                    data_ = self.previous_data[mask].to(self._device)
                    label_ = self.previous_label[mask].to(self._device)
                    pred_ = self._network(data_)["logits"]
                    pred_[:, : k * incremental_step].data.fill_(-10e10)
                    pred_[:, (k + 1) * incremental_step :].data.fill_(-10e10)
                    loss_ = F.cross_entropy(pred_, label_)
                    loss_.backward()

                    j = 0
                    for params in self._network.parameters():
                        if params is not None:
                            if j == 0:
                                stpt = 0
                            else:
                                stpt = sum(grad_numels[:j])

                            endpt = sum(grad_numels[: j + 1])
                            G[stpt:endpt, k].data.copy_(params.grad.data.view(-1))
                            j += 1

                    optimizer.zero_grad()

                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]
                logits[:, : self._known_classes].data.fill_(-10e10)
                loss_clf = F.cross_entropy(logits, targets)

                loss = loss_clf

                optimizer.zero_grad()
                loss.backward()

                j = 0
                for params in self._network.parameters():
                    if params is not None:
                        if j == 0:
                            stpt = 0
                        else:
                            stpt = sum(grad_numels[:j])

                        endpt = sum(grad_numels[: j + 1])
                        G[stpt:endpt, self._cur_task].data.copy_(
                            params.grad.data.view(-1)
                        )
                        j += 1

                dotprod = torch.mm(
                    G[:, self._cur_task].unsqueeze(0), G[:, : self._cur_task]
                )

                if (dotprod < 0).sum() > 0:

                    old_grad = G[:, : self._cur_task].cpu().t().double().numpy()
                    cur_grad = G[:, self._cur_task].cpu().contiguous().double().numpy()

                    C = old_grad @ old_grad.T
                    p = old_grad @ cur_grad
                    A = np.eye(old_grad.shape[0])
                    b = np.zeros(old_grad.shape[0])

                    v = solve_qp(C, -p, A, b)[0]

                    new_grad = old_grad.T @ v + cur_grad
                    new_grad = torch.tensor(new_grad).float().to(self._device)

                    new_dotprod = torch.mm(
                        new_grad.unsqueeze(0), G[:, : self._cur_task]
                    )
                    if (new_dotprod < -0.01).sum() > 0:
                        assert 0
                    j = 0
                    for params in self._network.parameters():
                        if params is not None:
                            if j == 0:
                                stpt = 0
                            else:
                                stpt = sum(grad_numels[:j])

                            endpt = sum(grad_numels[: j + 1])
                            params.grad.data.copy_(
                                new_grad[stpt:endpt]
                                .contiguous()
                                .view(params.grad.data.size())
                            )
                            j += 1

                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)
