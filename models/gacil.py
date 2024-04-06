# -*- coding: utf-8 -*-
"""
Proper implementation of the G-ACIL [1].

This implementation refers to the official implementation https://github.com/ZHUANGHP/Analytic-continual-learning.

References:
[1] Zhuang, Huiping, et al.
    "G-ACIL: Analytic Learning for Exemplar-Free Generalized Class Incremental Learning"
    arXiv preprint arXiv:2403.15706 (2024).
"""

import torch
from .acil import ACIL, ACILNet
from convs.analytic_linear import GeneralizedRecursiveLinear

__all__ = [
    "GACIL",
    "GACILNet",
]


class GACILNet(ACILNet):
    def generate_fc(self, *_):
        self.fc = GeneralizedRecursiveLinear(
            self.buffer_size,
            self.gamma,
            bias=False,
            device=self.device,
            dtype=self.dtype,
        )

    def update_fc(self, *_):
        """
        The G-ACIL is a generalized version of the ACIL for general CIL setting.
        Its output layer will be automatically updated.
        Thus you have no need to call this `update_fc` function.
        """
        pass

    @torch.no_grad()
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        X = self.convnet(X)["features"]
        X = self.buffer(X)
        Y: torch.Tensor = torch.nn.functional.one_hot(y)
        self.fc.fit(X, Y)


class GACIL(ACIL):
    def create_network(self) -> None:
        self._network = GACILNet(
            self.args,
            buffer_size=self.buffer_size,
            pretrained=False,
            gamma=self.gamma,
            device=self._device,
        )
