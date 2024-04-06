# -*- coding: utf-8 -*-
"""
Implementation of the GKEAL [1].

The GKEAL is a CIL method specially proposed for the few-shot CIL.
But the implementation here is just a simplified version for common CIL settings.
Compared with the method proposed in the paper, we do not perform image augmentation here.
Each sample will only be learned once by default.

This implementation refers to the official implementation https://github.com/ZHUANGHP/Analytic-continual-learning.

References:
[1] Zhuang, Huiping, et al.
    "GKEAL: Gaussian Kernel Embedded Analytic Learning for Few-Shot Class Incremental Task."
    Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
"""

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from .acil import ACIL, ACILNet
from convs.buffer import GaussianKernel
from typing import Dict, Any


class GKEALNet(ACILNet):
    def __init__(
        self,
        args: Dict[str, Any],
        buffer_size: int = 8192,
        gamma: float = 0.1,
        sigma: float = 10,
        pretrained: bool = False,
        device=None,
        dtype=torch.double,
    ) -> None:
        self.sigma = sigma
        super().__init__(
            args,
            buffer_size=buffer_size,
            gamma=gamma,
            pretrained=pretrained,
            device=device,
            dtype=dtype,
        )

    def generate_buffer(self) -> None:
        # The gaussian kernel buffer is late initialized.
        self._buffer_initialized = False
        self.buffer = GaussianKernel(
            torch.zeros((self.buffer_size, self.feature_dim)),
            self.sigma,
            self.device,
            self.dtype,
        )

    @torch.no_grad()
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        assert self._buffer_initialized
        super().fit(X, Y)


class GKEAL(ACIL):
    def create_network(self) -> None:
        # The width-adjusting parameter β controls the width of the Gaussian kernels.
        # There is a comfortable range for σ at around [5, 15] for CIFAR-100 and ImageNet-1k
        # that gives good results, where β = 1 / (2σ²).
        self._network = GKEALNet(
            self.args,
            buffer_size=self.buffer_size,
            gamma=self.gamma,
            sigma=self.args["sigma"],
            pretrained=False,
            device=self._device,
        )

    @torch.no_grad()
    def _train(
        self, train_loader: DataLoader, desc: str = "Incremental Learning"
    ) -> None:
        torch.cuda.empty_cache()
        if self._network._buffer_initialized:
            return super()._train(train_loader, desc)

        self._network.update_fc(self._total_classes)
        total_X = []
        total_y = []
        for _, X, y in tqdm(train_loader, desc="Selecting center vectors"):
            X: torch.Tensor = X.to(self._device, non_blocking=True)
            y: torch.Tensor = y.to(self._device, non_blocking=True)
            X = self._network.convnet(X)["features"]
            total_X.append(X)
            total_y.append(y)

        X_all = torch.cat(total_X)
        self._network.buffer.init(X_all, self.buffer_size)
        torch.cuda.empty_cache()
        for X, y in tqdm(zip(total_X, total_y), total=len(total_X), desc=desc):
            X = self._network.buffer(X)
            Y = torch.nn.functional.one_hot(y, self._total_classes)
            self._network.fc.fit(X, Y)
        self._network.fc.after_task()

        self._network._buffer_initialized = True
