import unittest

import numpy as np
from scipy import stats

from sdpa.utils.data_generation import generate_attention_data


class TestDataGeneration(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(42)

    def test_generate_attention_data_default(self) -> None:
        queries, keys, values = generate_attention_data()

        self.assertEqual(queries.shape, (1, 5, 4))
        self.assertEqual(keys.shape, (1, 5, 4))
        self.assertEqual(values.shape, (1, 5, 4))

    def test_generate_attention_data_custom(self) -> None:
        batch_size, seq_length, d_k, d_v = 2, 10, 8, 6
        queries, keys, values = generate_attention_data(
            batch_size, seq_length, d_k, d_v
        )

        self.assertEqual(queries.shape, (batch_size, seq_length, d_k))
        self.assertEqual(keys.shape, (batch_size, seq_length, d_k))
        self.assertEqual(values.shape, (batch_size, seq_length, d_v))

    def test_generate_attention_data_distribution(self) -> None:
        queries, keys, values = generate_attention_data(
            batch_size=100, seq_length=100, d_k=10, d_v=10
        )

        queries_flat = queries.flatten()
        keys_flat = keys.flatten()
        values_flat = values.flatten()

        def check_normal_distribution(data: np.ndarray, name: str) -> None:
            t_stat, p_val = stats.ttest_1samp(data, 0)
            self.assertGreater(
                p_val, 0.05, f"{name} mean significantly different from 0"
            )

            self.assertAlmostEqual(
                np.std(data),
                1.0,
                delta=0.1,
                msg=f"{name} std dev not close to 1",
            )

            _, p_val = stats.kstest(data, "norm")
            self.assertGreater(p_val, 0.05, f"{name} distribution not normal")

        check_normal_distribution(queries_flat, "Queries")
        check_normal_distribution(keys_flat, "Keys")
        check_normal_distribution(values_flat, "Values")


if __name__ == "__main__":
    unittest.main()
