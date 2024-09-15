import unittest

import numpy as np

from sdpa.attention.factory import AttentionFactory
from sdpa.attention.scaled_dot_product import ScaledDotProductAttention
from sdpa.utils.types import AttentionConfig, AttentionType


class TestAttentionFactor(unittest.TestCase):
    def test_create_scaled_dot_product_attention(self) -> None:
        config: AttentionConfig = {"d_model": 4, "num_heads": 1}
        attention = AttentionFactory.create(
            AttentionType.SCALED_DOT_PRODUCT, config
        )

        self.assertIsInstance(attention, ScaledDotProductAttention)
        self.assertEqual(attention.d_model, 4)

    def test_create_unsupported_attention(self) -> None:
        config: AttentionConfig = {"d_model": 4, "num_heads": 1}
        with self.assertRaises(ValueError):
            AttentionFactory.create("UnsupportedType", config)

    def test_created_attention_functionality(self) -> None:
        config: AttentionConfig = {"d_model": 4, "num_heads": 1}
        attention = AttentionFactory.create(
            AttentionType.SCALED_DOT_PRODUCT, config
        )

        query = np.array([[1.0, 0.0, 0.0, 0.0]])
        key = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        value = np.array([[1.0, 2.0], [3.0, 4.0]])

        output = attention.forward(query, key, value)

        self.assertEqual(output.output.shape, (1, 2))
        self.assertEqual(output.weights.shape, (1, 2))
