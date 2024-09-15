from __future__ import annotations

import numpy as np

from sdpa.attention.base import BaseAttention
from sdpa.utils.decorators import validate_input
from sdpa.utils.math_utils import softmax
from sdpa.utils.types import AttentionOutput


class ScaledDotProductAttention(BaseAttention):
    def __init__(self, d_k: int) -> None:
        self._d_k = d_k

    @property
    def d_k(self) -> int:
        return self._d_k

    @staticmethod
    def _scaled_dot_product(query: np.ndarray, key: np.ndarray) -> np.ndarray:
        # query, key: (batch_size, seq_length, d_k)
        scores = np.matmul(query, key.transpose(0, 2, 1)) / np.sqrt(
            query.shape[-1]
        )
        return scores

    @validate_input
    def forward(
        self, query: np.ndarray, key: np.ndarray, value: np.ndarray
    ) -> AttentionOutput:
        # query, key, value: (batch_size, seq_length, d_k)
        scores = self._scaled_dot_product(
            query, key
        )  # (batch_size, seq_length, seq_length)
        weights = softmax(scores)
        output = np.matmul(weights, value)  # (batch_size, seq_length, d_k)
        return AttentionOutput(output, weights)
