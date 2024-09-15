import numpy as np


def generate_attention_data(
    batch_size: int,
    seq_length: int,
    d_model: int | None = None,
    num_heads: int | None = None,
    d_k: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if d_model is not None:
        if num_heads is not None:
            if d_model % num_heads != 0:
                raise ValueError("d_model must be divisible by num_heads.")
            d_k = d_model // num_heads
        else:
            d_k = None
        dimension = d_model
    elif d_k is not None:
        if num_heads is not None:
            dimension = d_k * num_heads
        else:
            dimension = d_k
    else:
        raise ValueError("Either d_model or d_k must be specified.")

    queries = np.random.randn(batch_size, seq_length, dimension)
    keys = np.random.randn(batch_size, seq_length, dimension)
    values = np.random.randn(batch_size, seq_length, dimension)
    return queries, keys, values
