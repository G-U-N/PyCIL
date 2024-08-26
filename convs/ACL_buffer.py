# -*- coding: utf-8 -*-
"""
Buffer layers for the analytic continual learning (ACL) [1-3].

This implementation refers to the official implementation https://github.com/ZHUANGHP/Analytic-continual-learning.

References:
[1] Zhuang, Huiping, et al.
    "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection."
    Advances in Neural Information Processing Systems 35 (2022): 11602-11614.
[2] Zhuang, Huiping, et al.
    "GKEAL: Gaussian Kernel Embedded Analytic Learning for Few-Shot Class Incremental Task."
    Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
[3] Zhuang, Huiping, et al.
    "DS-AL: A Dual-Stream Analytic Learning for Exemplar-Free Class-Incremental Learning."
    Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 15. 2024.
"""

import torch
from typing import Optional, Union, Callable
from abc import ABCMeta, abstractmethod

__all__ = [
    "Buffer",
    "RandomBuffer",
    "activation_t",
]

activation_t = Union[Callable[[torch.Tensor], torch.Tensor], torch.nn.Module]


class Buffer(torch.nn.Module, metaclass=ABCMeta):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class RandomBuffer(torch.nn.Linear, Buffer):
    """
    Random buffer layer for the ACIL [1] and DS-AL [2].

    This implementation refers to the official implementation https://github.com/ZHUANGHP/Analytic-continual-learning.

    References:
    [1] Zhuang, Huiping, et al.
        "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection."
        Advances in Neural Information Processing Systems 35 (2022): 11602-11614.
    [2] Zhuang, Huiping, et al.
        "DS-AL: A Dual-Stream Analytic Learning for Exemplar-Free Class-Incremental Learning."
        Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 15. 2024.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device=None,
        dtype=torch.float,
        activation: Optional[activation_t] = torch.relu_,
    ) -> None:
        super(torch.nn.Linear, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.activation: activation_t = (
            torch.nn.Identity() if activation is None else activation
        )

        W = torch.empty((out_features, in_features), **factory_kwargs)
        b = torch.empty(out_features, **factory_kwargs) if bias else None

        # Using buffer instead of parameter
        self.register_buffer("weight", W)
        self.register_buffer("bias", b)

        # Random Initialization
        self.reset_parameters()

    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.weight)
        return self.activation(super().forward(X))
