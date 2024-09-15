import numpy as np
from scipy.stats import entropy
from sklearn.decomposition import PCA


def calc_entropy(weights: np.ndarray) -> float:
    return entropy(weights.flatten())


def calc_focus(weights: np.ndarray) -> float:
    return np.max(weights, axis=1).mean()


def calc_sparsity(weights: np.ndarray, threshold: float = 0.01) -> float:
    return (weights < threshold).mean()


def calc_pca(
    weights: list[np.ndarray], num_components: int = 2
) -> tuple[PCA, np.ndarray]:
    weights_2d = np.concatenate([w.reshape(-1, w.shape[-1]) for w in weights])
    pca = PCA(n_components=num_components)
    return pca, pca.fit_transform(weights_2d)


def round_floats(value: float, decimals: int = 5) -> float:
    if isinstance(value, (int, float)):
        return round(value, decimals)
    else:
        return value


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    sum_e_x = np.sum(e_x, axis=axis, keepdims=True)
    return e_x / sum_e_x
