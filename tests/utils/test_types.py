import unittest

import numpy as np

from sdpa.utils.types import (
    AttentionConfig,
    AttentionOutput,
    AttentionParams,
    AttentionType,
)


class TestTyoes(unittest.TestCase):
    def test_attention_output(self) -> None:
        output = np.array([[1.0, 2.0]])
        weights = np.array([[0.6, 0.4]])
        attention_output = AttentionOutput(output, weights)

        self.assertTrue(np.array_equal(attention_output.output, output))
        self.assertTrue(np.array_equal(attention_output.weights, weights))

    def test_attention_params(self) -> None:
        query = np.array([[1.0, 0.0, 0.0, 0.0]])
        key = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        value = np.array([[1.0, 2.0], [3.0, 4.0]])

        params = AttentionParams(query, key, value)

        self.assertTrue(np.array_equal(params.query, query))
        self.assertTrue(np.array_equal(params.key, key))
        self.assertTrue(np.array_equal(params.value, value))

    def test_attention_type(self) -> None:
        self.assertEqual(
            AttentionType.SCALED_DOT_PRODUCT.name, "SCALED_DOT_PRODUCT"
        )

    def test_attention_config(self) -> None:
        config: AttentionConfig = {"d_model": 4, "num_heads": 1}
        self.assertEqual(config["d_model"], 4)
        self.assertEqual(config["num_heads"], 1)


if __name__ == "__main__":
    unittest.main()
