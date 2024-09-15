import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_attention_hm(
    weights: np.ndarray, scale: float, output_path: str
) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(weights, annot=True, cmap="viridis")
    plt.title(f"Attention Weights Heatmap (Scale: {scale})")
    plt.xlabel("Key Positions")
    plt.ylabel("Query Positions")
    plt.savefig(output_path)
    plt.close()
