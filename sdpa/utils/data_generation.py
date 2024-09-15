import numpy as np


def generate_attention_data(
    batch_size: int = 1, seq_length: int = 5, d_k: int = 4, d_v: int = 4
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    queries = np.random.randn(batch_size, seq_length, d_k)
    keys = np.random.randn(batch_size, seq_length, d_k)
    values = np.random.randn(batch_size, seq_length, d_v)

    return queries, keys, values
