import unittest

import numpy as np

from sdpa.attention.scaled_dot_product import ScaledDotProductAttention


class TestSDPA(unittest.TestCase):
    def setUp(self) -> None:
        self.d_model = 4
        self.attention = ScaledDotProductAttention(self.d_model)

    def test_basic_forward(self) -> None:
        query = np.array([[1.0, 0.0, 0.0, 0.0]])
        key = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        value = np.array([[1.0, 2.0], [3.0, 4.0]])

        output = self.attention.forward(query, key, value)

        print(
            f"Basic Forward - Score: {self.attention._scaled_dot_product(query, key)}"  # noqa
        )
        print(f"Basic Forward - Weights: {output.weights}")
        print(f"Basic Forward - Output: {output.output}")

        self.assertEqual(output.weights.shape, (1, 2))
        self.assertEqual(output.output.shape, (1, 2))

        np.testing.assert_almost_equal(np.sum(output.weights), 1.0)

        # First weight should be closer to 1 than the second weight
        self.assertGreater(output.weights[0, 0], output.weights[0, 1])

        expected_output = (
            output.weights[0, 0] * value[0] + output.weights[0, 1] * value[1]
        )
        np.testing.assert_array_almost_equal(output.output[0], expected_output)

    def test_multi_query(self) -> None:
        query = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        key = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        value = np.array([[1.0, 2.0], [3.0, 4.0]])

        output = self.attention.forward(query, key, value)

        print(
            f"Multi Query - Scores: {self.attention._scaled_dot_product(query, key)}"  # noqa
        )
        print(f"Multi Query - Weights: {output.weights}")
        print(f"Multi Query - Output: {output.output}")

        self.assertEqual(output.weights.shape, (2, 2))
        self.assertEqual(output.output.shape, (2, 2))
        np.testing.assert_almost_equal(
            np.sum(output.weights, axis=1), [1.0, 1.0]
        )

    def test_scaled_dot_product(self) -> None:
        query = np.array([[1.0, 1.0, 1.0, 1.0]])
        key = np.array([[1.0, 1.0, 1.0, 1.0]])

        result = self.attention._scaled_dot_product(query, key)
        expected = np.array([[2.0]])  # 4 / sqrt(4) = 2.0

        print(f"Scaled Dot Product - Result: {result}")
        print(f"Scaled Dot Product - Expected: {expected}")

        np.testing.assert_almost_equal(result, expected)

    def test_softmax(self) -> None:
        x = np.array([[1.0, 2.0, 3.0]])
        result = self.attention._softmax(x)

        print(f"Softmax - Input: {x}")
        print(f"Softmax - Result: {result}")

        self.assertEqual(result.shape, (1, 3))
        np.testing.assert_almost_equal(np.sum(result), 1.0)
        self.assertTrue(
            np.all(result[0, 1:] > result[0, :-1])
        )  # Monotonically increasing

    def test_zero_query(self) -> None:
        query = np.array([[0.0, 0.0, 0.0, 0.0]])
        key = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        value = np.array([[1.0, 2.0], [3.0, 4.0]])

        output = self.attention.forward(query, key, value)

        print(
            f"Zero Query - Scores: {self.attention._scaled_dot_product(query, key)}"  # noqa
        )
        print(f"Zero Query - Weights: {output.weights}")
        print(f"Zero Query - Output: {output.output}")

        np.testing.assert_almost_equal(
            output.weights[0, 0], output.weights[0, 1]
        )  # Equal attention (0.5, 0.5)

    def test_identical_keys(self) -> None:
        query = np.array([[1.0, 0.0, 0.0, 0.0]])
        key = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        value = np.array([[1.0, 2.0], [3.0, 4.0]])

        output = self.attention.forward(query, key, value)

        print(
            f"Identical Keys - Scores: {self.attention._scaled_dot_product(query, key)}"  # noqa
        )
        print(f"Identical Keys - Weights: {output.weights}")
        print(f"Identical Keys - Output: {output.output}")

        np.testing.assert_almost_equal(
            output.weights[0, 0], output.weights[0, 1]
        )  # Equal attention (0.5, 0.5)

    def test_numerical_stability(self) -> None:
        query = np.array([[1e3, 1e3, 1e3, 1e3]])
        key = np.array([[1e3, 1e3, 1e3, 1e3], [1e-3, 1e-3, 1e-3, 1e-3]])
        value = np.array([[1.0, 2.0], [3.0, 4.0]])

        output = self.attention.forward(query, key, value)

        print(
            f"Numerical Stability - Scores: {self.attention._scaled_dot_product(query, key)}"  # noqa
        )
        print(f"Numerical Stability - Weights: {output.weights}")
        print(f"Numerical Stability - Output: {output.output}")

        self.assertFalse(np.any(np.isnan(output.weights)))
        self.assertFalse(np.any(np.isnan(output.output)))


if __name__ == "__main__":
    unittest.main()
