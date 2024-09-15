import unittest

import numpy as np
from scipy import stats

from sdpa.utils.data_generation import generate_attention_data


class TestDataGeneration(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(42)

    def test_generate_attention_data_custom(self) -> None:
        batch_size, seq_length, d_k, num_heads = 2, 10, 8, 2
        queries, keys, values = generate_attention_data(
            batch_size=batch_size,
            seq_length=seq_length,
            d_k=d_k,
            num_heads=num_heads,
        )

        expected_dimension = d_k * num_heads
        self.assertEqual(
            queries.shape, (batch_size, seq_length, expected_dimension)
        )
        self.assertEqual(
            keys.shape, (batch_size, seq_length, expected_dimension)
        )
        self.assertEqual(
            values.shape, (batch_size, seq_length, expected_dimension)
        )

    def test_generate_attention_data_distribution(self) -> None:
        batch_size, seq_length, d_model, num_heads = 100, 100, 80, 8
        queries, keys, values = generate_attention_data(
            batch_size=batch_size,
            seq_length=seq_length,
            d_model=d_model,
            num_heads=num_heads,
        )
        queries_flat = queries.flatten()
        keys_flat = keys.flatten()
        values_flat = values.flatten()

        def check_normal_distribution(data: np.ndarray, name: str) -> None:
            t_stat, p_val = stats.ttest_1samp(data, 0)
            self.assertGreater(
                p_val, 0.01, f"{name} mean significantly different from 0"
            )

            self.assertAlmostEqual(
                np.std(data),
                1.0,
                delta=0.1,
                msg=f"{name} std dev not close to 1",
            )

            _, p_val = stats.kstest(data, "norm")
            self.assertGreater(p_val, 0.01, f"{name} distribution not normal")

        check_normal_distribution(queries_flat, "Queries")
        check_normal_distribution(keys_flat, "Keys")
        check_normal_distribution(values_flat, "Values")


if __name__ == "__main__":
    unittest.main()
