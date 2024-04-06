# -*- coding: utf-8 -*-
"""
Proper implementation of the DS-AL [1].

This implementation refers to the official implementation https://github.com/ZHUANGHP/Analytic-continual-learning.

References:
[1] Zhuang, Huiping, et al.
    "DS-AL: A Dual-Stream Analytic Learning for Exemplar-Free Class-Incremental Learning."
    Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 15. 2024.
"""
import torch

from .acil import ACIL, ACILNet
from typing import Callable, Dict, Any
from convs.buffer import RandomBuffer
from convs.analytic_linear import RecursiveLinear
from convs.buffer import activation_t


class DSALNet(ACILNet):
    def __init__(
        self,
        args: Dict[str, Any],
        buffer_size: int = 8192,
        gamma_main: float = 1e-3,
        gamma_comp: float = 1e-3,
        C: float = 1,
        activation_main: activation_t = torch.relu,
        activation_comp: activation_t = torch.tanh,
        pretrained: bool = False,
        device=None,
        dtype=torch.double,
    ) -> None:
        self.C = C
        self.gamma_comp = gamma_comp
        self.activation_main = activation_main
        self.activation_comp = activation_comp
        super().__init__(args, buffer_size, gamma_main, pretrained, device, dtype)

    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        X = self.buffer(self.convnet(X)["features"])
        X_main = self.fc(self.activation_main(X))
        X_comp = self.fc_comp(self.activation_comp(X))
        return {"logits": X_main + self.C * X_comp}

    @torch.no_grad()
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        num_classes = max(self.fc.out_features, int(y.max().item()) + 1)
        Y_main = torch.nn.functional.one_hot(y, num_classes=num_classes)
        X = self.buffer(self.convnet(X)["features"])

        # Train the main stream
        X_main = self.activation_main(X)
        self.fc.fit(X_main, Y_main)
        self.fc.after_task()

        # Previous label cleansing (PLC)
        Y_comp = Y_main - self.fc(X_main)
        Y_comp[:, : -self.increment_size] = 0

        # Train the compensation stream
        X_comp = self.activation_comp(X)
        self.fc_comp.fit(X_comp, Y_comp)

    @torch.no_grad()
    def after_task(self) -> None:
        self.fc.after_task()
        self.fc_comp.after_task()

    def generate_buffer(self) -> None:
        self.buffer = RandomBuffer(
            self.feature_dim,
            self.buffer_size,
            activation=None,
            device=self.device,
            dtype=self.dtype,
        )

    def generate_fc(self, *_) -> None:
        # Main stream
        self.fc = RecursiveLinear(
            self.buffer_size,
            self.gamma,
            bias=False,
            device=self.device,
            dtype=self.dtype,
        )

        # Compensation stream
        self.fc_comp = RecursiveLinear(
            self.buffer_size,
            self.gamma_comp,
            bias=False,
            device=self.device,
            dtype=self.dtype,
        )

    def update_fc(self, nb_classes) -> None:
        self.increment_size = nb_classes - self.fc.out_features
        self.fc.update_fc(nb_classes)
        self.fc_comp.update_fc(nb_classes)


class DSAL(ACIL):
    def create_network(self) -> None:
        # We recommend using the grid search to find the best compensation ratio `C` in the interval [0, 2].
        # The best value is 0.6 for the CIFAR-100, while the best value for the ImageNet-1k is 1.5.
        self._network = DSALNet(
            self.args,
            self.buffer_size,
            self.gamma,
            self.args["gamma_comp"],
            self.args["compensation_ratio"],
        )
