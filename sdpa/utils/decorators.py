from functools import wraps

import numpy as np


def validate_input(func):
    @wraps(func)
    def wrapper(self, query: np.ndarray, key: np.ndarray, value: np.ndarray):
        if query.shape[-1] != self.d_k or key.shape[-1] != self.d_k:
            raise ValueError(f"Input dimensions must match d_k ({self.d_k})")
        if key.shape[-2] != value.shape[-2]:
            raise ValueError(
                "Key and Value must have the same sequence length"
            )
        return func(self, query, key, value)

    return wrapper
