# -*- coding: utf-8 -*-
"""
Proper implementation of the ACIL [1].

This implementation refers to the official implementation https://github.com/ZHUANGHP/Analytic-continual-learning.

References:
[1] Zhuang, Huiping, et al.
    "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection."
    Advances in Neural Information Processing Systems 35 (2022): 11602-11614.
"""

import torch
import logging
import numpy as np
from tqdm import tqdm
from os import path, makedirs
from utils.inc_net import ACILNet
from models.base import BaseLearner
from typing import Dict, Any, Optional, Sized
from torch.utils.data import DataLoader, Sampler
from utils.data_manager import DataManager, DummyDataset
from tqdm.contrib.logging import logging_redirect_tqdm


__all__ = [
    "ACIL",
]


class _Extract(torch.nn.Module):
    def __init__(self, name: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.name = name

    def forward(self, X: Dict[str, Any]) -> torch.Tensor:
        return X[self.name]


class InplaceRepeatSampler(Sampler):
    def __init__(self, data_source: Sized, num_repeats: int = 1):
        self.data_source = data_source
        self.num_repeats = num_repeats

    def __iter__(self):
        for i in range(len(self.data_source)):
            for _ in range(self.num_repeats):
                yield i

    def __len__(self):
        return len(self.data_source) * self.num_repeats


class ACIL(BaseLearner):
    """
    Training process of the ACIL [1].

    This implementation refers to the official implementation https://github.com/ZHUANGHP/Analytic-continual-learning.

    References:
    [1] Zhuang, Huiping, et al.
        "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection."
        Advances in Neural Information Processing Systems 35 (2022): 11602-11614.
    """
    def __init__(self, args: Dict[str, Any]) -> None:
        args.update(args["configurations"][args["dataset"]])

        if "memory_size" not in args:
            args["memory_size"] = 0
        elif args["memory_size"] != 0:
            raise ValueError(
                f"{self.__class__.__name__} is an exemplar-free method,"
                "so the `memory_size` must be 0."
            )
        super().__init__(args)
        self.parse_args(args)

        makedirs(self.save_path, exist_ok=True)

        """ Create the network """
        self.create_network()

    def parse_args(self, args: Dict[str, Any]) -> None:
        """ Base training hyper-parameters
        For small datasets like CIFAR-100 without powerful image augmentation provided here,
        we suggest using the MultiStepLR scheduler to get a more generalizable model.
        """
        self.num_workers: int = args.get("num_workers", 8)

        self.train_eval_freq: int = args.get("train_eval_freq", 1)
        # Batch size
        self.init_batch_size: int = args.get("init_batch_size", 256)
        # Learning rate information
        self.lr_info: Dict[str, Any] = args["scheduler"]
        # 5e-4 for CIFAR and 5e-5 for ImageNet
        self.weight_decay: float = args.get("init_weight_decay", 1e-4)

        """ Incremental learning hyper-parameters"""
        # Bigger batch size leads faster learning speed, >= 4096 for ImageNet.
        self.IL_batch_size: int = args.get("IL_batch_size", self.init_batch_size)
        # 8192 for CIFAR-100, and 16384 for ImageNet
        self.buffer_size: int = args["buffer_size"]
        # Regularization term of the regression
        self.gamma: float = args["gamma"]
        # Inplace repeat sampler for the training data loader during incremental learning
        self.inplace_repeat: int = args.get("inplace_repeat", 1)

        # Set the log path to save the base training backbone
        self.seed: int = args["seed"]
        self.conv_type: str = args["convnet_type"]
        self.save_path = (
            f"logs/{args['model_name']}/{args['dataset']}/{args['init_cls']}"
        )

    def create_network(self) -> None:
        self._network = ACILNet(
            self.args,
            buffer_size=self.buffer_size,
            pretrained=False,
            gamma=self.gamma,
            device=self._device,
        )

    def incremental_train(self, data_manager: DataManager) -> None:
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        test_dataset: DummyDataset = data_manager.get_dataset(
            np.arange(0, self._total_classes),
            source="test",
            mode="test",
            ret_data=False,
        )  # type: ignore

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.init_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=(self.num_workers == 0),
        )

        if self._cur_task == 0:
            assert self._known_classes == 0
            # You can specify the base weight in configuration file to skip the base training process.
            # This is helpful when you want to compare the performance of different methods fairly.
            if self.args.get("base_weight", None) is not None:
                base_weight_path = self.args["base_weight"]
                assert path.isfile(base_weight_path), "The base weight is not found."
                logging.info(
                    f"Loading the base model from the provided weight: {base_weight_path}. "
                    f"The base training process is skipped."
                )
                self._network.convnet.load_state_dict(torch.load(base_weight_path))
                self._network.to(self._device)
            else:
                train_dataset_init: DummyDataset = data_manager.get_dataset(
                    np.arange(0, self._total_classes),
                    source="train",
                    mode="train",
                    ret_data=False,
                )  # type: ignore
                train_loader_init = DataLoader(
                    train_dataset_init,
                    batch_size=self.init_batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                    persistent_workers=True,
                )
                self._init_train(train_loader_init, self.test_loader)
            self._network.generate_buffer()
            self._network.generate_fc()
        self._network.to(self._device)

        train_dataset: DummyDataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="test",
            ret_data=False,
        )  # type: ignore

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.IL_batch_size,
            num_workers=self.num_workers,
            sampler=InplaceRepeatSampler(train_dataset, self.inplace_repeat),
        )

        self._train(
            train_loader,
            desc="Base Re-align" if self._cur_task == 0 else "Incremental Learning",
        )

    @torch.no_grad()
    def _train(
        self, train_loader: DataLoader, desc: str = "Incremental Learning"
    ) -> None:
        self._network.eval()
        self._network.update_fc(self._total_classes)
        for _, X, y in tqdm(train_loader, desc=desc):
            X: torch.Tensor = X.to(self._device, non_blocking=True)
            y: torch.Tensor = y.to(self._device, non_blocking=True)
            self._network.fit(X, y)
        self._network.after_task()

    def _init_train(self, train_loader: DataLoader, test_loader: DataLoader) -> None:
        model = torch.nn.Sequential(
            self._network.convnet,
            _Extract("features"),
            torch.nn.Linear(self._network.feature_dim, self._total_classes, bias=False),
        ).to(self._device)
        if len(self._multiple_gpus) > 1:
            model = torch.nn.DataParallel(model, self._multiple_gpus)

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.lr_info["init_lr"],
            momentum=0.9,
            weight_decay=self.weight_decay,
        )

        criterion = torch.nn.CrossEntropyLoss().to(self._device)

        # Scheduler with linear warmup
        scheduler_type = self.lr_info["type"]
        if scheduler_type == "MultiStep":
            # For the CIFAR-100 dataset provided
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.lr_info["milestones"],
                gamma=self.lr_info["decay"],
            )
        elif scheduler_type == "CosineAnnealing":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.lr_info["init_epochs"], eta_min=1e-6
            )
        else:
            raise ValueError(f"Unsupported LR scheduler type: {scheduler_type}")

        if self.lr_info.get("warmup", 0) > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-3,
                total_iters=self.lr_info["warmup"],
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, [warmup_scheduler, scheduler], [self.lr_info["warmup"]]
            )

        init_epochs = self.lr_info["init_epochs"]

        total_batches = init_epochs * (len(train_loader) + len(test_loader)) + (
            init_epochs // self.train_eval_freq
        ) * len(train_loader)
        process_bar = tqdm(total=total_batches, desc="Base Training", unit="batches")

        for epoch in range(init_epochs):
            model.train()
            for _, X, y in train_loader:
                X: torch.Tensor = X.to(self._device, non_blocking=True)
                y: torch.Tensor = y.to(self._device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                y_hat = model(X)
                loss: torch.Tensor = criterion(y_hat, y)
                loss.backward()
                optimizer.step()
                process_bar.update(1)

            model.eval()
            if (epoch + 1) % self.train_eval_freq == 0:
                train_metrics = _evaluate(
                    model, train_loader, process_bar, self._device
                )
                with logging_redirect_tqdm():
                    logging.info(
                        f"Epoch {epoch + 1}/{init_epochs} - "
                        f"Train Loss: {train_metrics['loss']:.4f}, "
                        f"Train Acc@1: {train_metrics['acc@1'] * 100:.3f}%, "
                        f"Train Acc@5: {train_metrics['acc@5'] * 100:.3f}%, "
                        f"LR: {scheduler.get_last_lr()[0]}"
                    )

            test_metrics = _evaluate(model, test_loader, process_bar, self._device)
            with logging_redirect_tqdm():
                logging.info(
                    f"Epoch {epoch + 1}/{init_epochs} - "
                    f"Test Loss: {test_metrics['loss']:.4f}, "
                    f"Test Acc@1: {test_metrics['acc@1'] * 100:.3f}%, "
                    f"Test Acc@5: {test_metrics['acc@5'] * 100:.3f}%"
                )

            scheduler.step()
        self._network.eval()
        saving_file = path.join(
            self.save_path,
            f"{self.conv_type}_{self.seed}_{round(test_metrics['acc@1'] * 10000)}.pth",
        )
        torch.save(self._network.convnet.state_dict(), saving_file)
        self._network.freeze()

    def after_task(self) -> None:
        self._known_classes = self._total_classes
        self._network.after_task()


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    progress_bar: Optional[tqdm] = None,
    device=None,
) -> Dict[str, float]:
    """Evaluate the model on the given data loader."""
    model.eval()
    acc1_cnt, acc5_cnt = 0, 0
    sample_cnt = 0
    total_loss = 0.0

    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    for _, X, y in loader:
        X: torch.Tensor = X.to(device, non_blocking=True)
        y: torch.Tensor = y.to(device, non_blocking=True)

        logits: torch.Tensor = model(X)
        acc1_cnt += (logits.argmax(dim=1) == y).sum().item()
        acc5_cnt += (logits.topk(5, dim=1).indices == y[:, None]).sum().item()
        sample_cnt += y.size(0)
        total_loss += float(criterion(logits, y).item())

        if progress_bar is not None:
            progress_bar.update(1)

    return {
        "acc@1": acc1_cnt / sample_cnt,
        "acc@5": acc5_cnt / sample_cnt,
        "loss": total_loss / sample_cnt,
    }
