from __future__ import annotations

import numpy as np

from sdpa.attention.base import BaseAttention
from sdpa.utils.decorators import validate_input
from sdpa.utils.types import AttentionOutput


class ScaledDotProductAttention(BaseAttention):
    def __init__(self, d_model: int) -> None:
        self._d_model = d_model

    @property
    def d_model(self) -> int:
        return self._d_model

    @staticmethod
    def _scaled_dot_product(query: np.ndarray, key: np.ndarray) -> np.ndarray:
        return np.dot(query, key.T) / np.sqrt(query.shape[-1])

    @validate_input
    def forward(
        self, query: np.ndarray, key: np.ndarray, value: np.ndarray
    ) -> AttentionOutput:
        scores = self._scaled_dot_product(query, key)
        weights = self._softmax(scores)
        output = np.dot(weights, value)
        return AttentionOutput(output, weights)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
