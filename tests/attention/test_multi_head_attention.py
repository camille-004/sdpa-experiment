import unittest

import numpy as np

from sdpa.attention.multi_head import MultiHeadAttention


class TestMultiHeadAttention(unittest.TestCase):
    def setUp(self) -> None:
        self.d_model = 8
        self.num_heads = 2
        self.attention = MultiHeadAttention(
            d_model=self.d_model, num_heads=self.num_heads
        )

    def test_basic_forward(self) -> None:
        query = np.array([[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
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

        output = self.attention.forward(query, key, value)

        print(f"Basic forward - Output Shape: {output.output.shape}")
        print(f"Basic Forward - Weights Shape: {output.weights.shape}")
        print(f"Basic Forward - Output: {output.output}")
        print(f"Basic Forward - Weights: {output.weights}")

        self.assertEqual(output.output.shape, (1, 1, self.d_model))
        self.assertEqual(output.weights.shape, (1, 1, key.shape[1]))

        np.testing.assert_almost_equal(
            np.sum(output.weights, axis=-1), np.ones((1, 1))
        )

    def test_multi_query(self) -> None:
        query = np.array(
            [
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ]
        )
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

        output = self.attention.forward(query, key, value)

        print(f"Multi Query - Output Shape: {output.output.shape}")
        print(f"Multi Query - Weights Shape: {output.weights.shape}")
        print(f"Multi Query - Output: {output.output}")
        print(f"Multi Query - Weights: {output.weights}")

        self.assertEqual(output.output.shape, (1, 2, self.d_model))
        self.assertEqual(output.weights.shape, (1, 2, key.shape[1]))

        np.testing.assert_almost_equal(
            np.sum(output.weights, axis=-1), np.ones((1, 2))
        )

    def test_zero_query(self) -> None:
        query = np.zeros((1, 1, self.d_model))
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

        output = self.attention.forward(query, key, value)

        print(f"Zero Query - Output: {output.output}")
        print(f"Zero Query - Weights: {output.weights}")

        np.testing.assert_almost_equal(
            output.weights[0, 0, 0], output.weights[0, 0, 1]
        )

    def test_identical_keys(self) -> None:
        query = np.array([[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
        key = np.array(
            [
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
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

        output = self.attention.forward(query, key, value)

        print(f"Identical Keys - Output: {output.output}")
        print(f"Identical Keys - Weights: {output.weights}")

        np.testing.assert_almost_equal(
            output.weights[0, 0, 0], output.weights[0, 0, 1]
        )

    def test_numerical_stability(self) -> None:
        query = np.array([[[1e3] * self.d_model]])
        key = np.array([[[1e3] * self.d_model, [1e-3] * self.d_model]])
        value = np.array([[[1.0] * self.d_model, [2.0] * self.d_model]])

        output = self.attention.forward(query, key, value)

        print(f"Numerical Stability - Output: {output.output}")
        print(f"Numerical Stability - Weights: {output.weights}")

        self.assertFalse(np.any(np.isnan(output.weights)))
        self.assertFalse(np.any(np.isnan(output.output)))

    def test_attention_weights_sum_to_one(self) -> None:
        query = np.random.randn(2, 3, self.d_model)
        key = np.random.randn(2, 4, self.d_model)
        value = np.random.randn(2, 4, self.d_model)

        output = self.attention.forward(query, key, value)

        weights_sum = np.sum(output.weights, axis=-1)
        np.testing.assert_almost_equal(weights_sum, np.ones_like(weights_sum))
