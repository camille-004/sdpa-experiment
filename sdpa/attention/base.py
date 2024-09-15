from abc import ABC, abstractmethod

import numpy as np

from sdpa.utils.types import AttentionOutput


class BaseAttention(ABC):
    @abstractmethod
    def forward(
        self, query: np.ndarray, key: np.ndarray, value: np.ndarray
    ) -> AttentionOutput:
        pass
