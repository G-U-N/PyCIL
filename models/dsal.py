# -*- coding: utf-8 -*-
"""
Proper implementation of the DS-AL [1].

This implementation refers to the official implementation https://github.com/ZHUANGHP/Analytic-continual-learning.

References:
[1] Zhuang, Huiping, et al.
    "DS-AL: A Dual-Stream Analytic Learning for Exemplar-Free Class-Incremental Learning."
    Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 15. 2024.
"""

from .acil import ACIL
from utils.inc_net import DSALNet


class DSAL(ACIL):
    """
    Training process of the DS-AL [1].

    This implementation refers to the official implementation https://github.com/ZHUANGHP/Analytic-continual-learning.

    References:
    [1] Zhuang, Huiping, et al.
        "DS-AL: A Dual-Stream Analytic Learning for Exemplar-Free Class-Incremental Learning."
        Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 15. 2024.
    """

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
