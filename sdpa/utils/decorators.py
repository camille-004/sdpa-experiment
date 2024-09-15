from functools import wraps

import numpy as np


def validate_input(func):
    @wraps(func)
    def wrapper(self, query: np.ndarray, key: np.ndarray, value: np.ndarray):
        if query.shape[-1] != self.d_model or key.shape[-1] != self.d_model:
            raise ValueError(
                f"Input dimensions must match d_model ({self.d_model})"
            )
        if key.shape[0] != value.shape[0]:
            raise ValueError(
                "Key and Value must have the same number of elemenets"
            )
        return func(self, query, key, value)

    return wrapper
