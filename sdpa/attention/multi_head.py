import numpy as np

from sdpa.attention.base import BaseAttention
from sdpa.attention.scaled_dot_product import ScaledDotProductAttention
from sdpa.utils.types import AttentionOutput


class MultiHeadAttention(BaseAttention):
    def __init__(self, d_model: int, num_heads: int) -> None:
        assert (
            d_model % num_heads == 0
        ), "d_model must be divisible by num_heads"
        self._d_model = d_model
        self._num_heads = num_heads
        self._d_k = d_model // num_heads
        self.d_v = self._d_k

        self.W_Q = np.random.randn(d_model, d_model)
        self.W_K = np.random.randn(d_model, d_model)
        self.W_V = np.random.randn(d_model, d_model)
        self.W_O = np.random.randn(d_model, d_model)

        self.sdpa = ScaledDotProductAttention(self._d_k)

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def d_k(self) -> int:
        return self._d_k

    def forward(
        self, query: np.ndarray, key: np.ndarray, value: np.ndarray
    ) -> AttentionOutput:
        batch_size, seq_length_query, _ = query.shape
        _, seq_length_key, _ = key.shape

        Q = np.matmul(query, self.W_Q)  # (batch_size, seq_length, d_model)
        K = np.matmul(key, self.W_K)
        V = np.matmul(value, self.W_V)

        Q = Q.reshape(
            batch_size, seq_length_query, self.num_heads, self.d_k
        ).transpose(0, 2, 1, 3)
        K = K.reshape(
            batch_size, seq_length_key, self.num_heads, self.d_k
        ).transpose(0, 2, 1, 3)
        V = V.reshape(
            batch_size, seq_length_key, self.num_heads, self.d_k
        ).transpose(0, 2, 1, 3)

        Q = Q.reshape(-1, seq_length_query, self.d_k)
        K = K.reshape(-1, seq_length_key, self.d_k)
        V = V.reshape(-1, seq_length_key, self.d_k)

        attn_output = self.sdpa.forward(Q, K, V)
        context = attn_output.output
        weights = attn_output.weights

        context = context.reshape(
            batch_size, self.num_heads, seq_length_query, self.d_k
        )
        context = context.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_length_query, self.d_model
        )

        weights = weights.reshape(
            batch_size, self.num_heads, seq_length_query, seq_length_key
        )
        avg_weights = weights.mean(axis=1)

        output = np.matmul(context, self.W_O)

        return AttentionOutput(output=output, weights=avg_weights)
