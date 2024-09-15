import unittest

import numpy as np

from sdpa.attention.factory import AttentionFactory
from sdpa.attention.multi_head import MultiHeadAttention
from sdpa.attention.scaled_dot_product import ScaledDotProductAttention
from sdpa.utils.types import AttentionConfig


class TestAttentionFactor(unittest.TestCase):
    def test_create_scaled_dot_product_attention(self) -> None:
        config: AttentionConfig = {"d_k": 4}
        attention = AttentionFactory.create(config)

        self.assertIsInstance(attention, ScaledDotProductAttention)
        assert isinstance(attention, ScaledDotProductAttention)
        self.assertEqual(attention.d_k, 4)

    def test_create_multi_head_attention(self) -> None:
        config: AttentionConfig = {"d_model": 8, "num_heads": 2}
        attention = AttentionFactory.create(config)

        self.assertIsInstance(attention, MultiHeadAttention)
        assert isinstance(attention, MultiHeadAttention)
        self.assertEqual(attention.d_model, 8)
        self.assertEqual(attention.num_heads, 2)
        self.assertEqual(attention.d_k, 4)

    def test_scaled_dot_product_attention_functionality(self) -> None:
        config: AttentionConfig = {"d_k": 4}
        attention = AttentionFactory.create(config)

        query = np.array(
            [[[1.0, 0.0, 0.0, 0.0]]]
        )  # (batch_size=1, seq_length,1, d_k=4)
        key = np.array(
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]
        )  # (1, 2, 4)
        value = np.array([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2, 2)

        output = attention.forward(query, key, value)

        self.assertEqual(output.output.shape, (1, 1, 2))
        self.assertEqual(output.weights.shape, (1, 1, 2))

    def test_multi_head_attention_functionality(self) -> None:
        config: AttentionConfig = {"d_model": 8, "num_heads": 2}
        attention = AttentionFactory.create(config)

        query = np.array(
            [[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]
        )  # (1, 1, 8)
        key = np.array(
            [
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ]
        )
        value = np.array(
            [
                [
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
                ]
            ]
        )

        output = attention.forward(query, key, value)

        self.assertEqual(
            output.output.shape, (1, 1, 8)
        )  # Should match d_model
        self.assertEqual(output.weights.shape, (1, 1, 2))
