import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # type: ignore
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from scipy.stats import entropy
from sklearn.decomposition import PCA

CMAP = "coolwarm"


def plot_attention_hm(
    weights: np.ndarray, scale: float, output_path: str
) -> None:
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        weights, annot=True, cmap=CMAP, cbar_kws={"label": "Attention Weight"}
    )
    plt.title(
        f"Attention Weights Heatmap (Scale: {scale})",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Key Positions", fontsize=12)
    plt.ylabel("Query Positions", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_attention_dist(
    weight_dict: dict[float, np.ndarray], output_path: str
) -> None:
    plt.figure(figsize=(10, 6))
    for scale, weights in weight_dict.items():
        sns.kdeplot(weights.flatten(), label=f"Scale: {scale}", fill=True)
    plt.title("Distribution of Attention Weights")
    plt.xlabel("Attention Weight")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(output_path)
    plt.close()


def plot_attention_entropy(
    weights: list[np.ndarray], scales: list[float], output_path: str
) -> None:
    entropies = [entropy(w.flatten()) for w in weights]

    max_entropies = [np.log(w.size) for w in weights]
    relative_entropies = [e / m for e, m in zip(entropies, max_entropies)]

    fig, ax1 = plt.subplots(figsize=(12, 10), sharex=True)

    color = "tab:blue"
    ax1.set_xlabel("Scale")
    ax1.set_ylabel("Absolute Entropy", color=color)
    line1 = ax1.plot(
        scales,
        entropies,
        color=color,
        marker="o",
        linestyle="-",
        label="Absolute Entropy",
    )
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()

    color = "tab:orange"
    ax2.set_ylabel("Relative Entropy", color=color)
    line2 = ax2.plot(
        scales,
        relative_entropies,
        color=color,
        marker="s",
        linestyle="-",
        label="Relative Entropy",
    )
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim(0, 1)

    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="lower left")

    plt.title("Attention Entropy vs Scale")

    ax1.grid(True, linestyle="--", alpha=0.7)

    for ax, data, color in [
        (ax1, entropies, "tab:blue"),
        (ax2, relative_entropies, "tab:orange"),
    ]:
        min_idx, max_idx = np.argmin(data), np.argmax(data)
        ax.annotate(
            f"Min: {data[min_idx]:.2f}",
            xy=(scales[min_idx], data[min_idx]),
            xytext=(5, 5),
            textcoords="offset points",
            color=color,
        )
        ax.annotate(
            f"Max: {data[max_idx]:.2f}",
            xy=(scales[max_idx], data[max_idx]),
            xytext=(5, 5),
            textcoords="offset points",
            color=color,
        )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_attention_focus(
    weights: list[np.ndarray], scales: list[float], output_path: str
) -> None:
    focus_metrics = [np.max(w, axis=1).mean() for w in weights]

    plt.figure(figsize=(10, 6))
    plt.plot(scales, focus_metrics, marker="o")
    plt.title("Attention Focus vs Scale")
    plt.xlabel("Scale")
    plt.ylabel("Average Max Attention Weight")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(output_path)
    plt.close()


def plot_attention_sparsity(
    weights: list[np.ndarray],
    scales: list[float],
    threshold: float,
    output_path: str,
) -> None:
    sparsity_metrics = [(w < threshold).mean() for w in weights]

    plt.figure(figsize=(10, 6))
    plt.plot(scales, sparsity_metrics, marker="o")
    plt.title(f"Attention Sparsity vs Scale (threshold: {threshold})")
    plt.xlabel("Scale")
    plt.ylabel("Proportion of Weights < Threshold")
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(output_path)
    plt.close()


def plot_attention_pca(
    weights: list[np.ndarray], scales: list[float], output_path: str
) -> None:
    pca = PCA(n_components=2)
    pca_results = [
        pca.fit_transform(w.reshape(w.shape[0], -1)) for w in weights
    ]

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.set_facecolor("white")
    ax.set_facecolor("#f0f0f0")

    cmap = plt.get_cmap(CMAP)
    norm = mcolors.Normalize(vmin=min(scales), vmax=max(scales))

    for i, result in enumerate(pca_results):
        color = cmap(norm(scales[i]))

        hull = ConvexHull(result)
        hull_points = result[hull.vertices]
        hull_path = Polygon(
            hull_points, facecolor=color, edgecolor=color, alpha=0.2, zorder=1
        )
        ax.add_patch(hull_path)

        for simplex in hull.simplices:
            ax.plot(
                result[simplex, 0],
                result[simplex, 1],
                color=color,
                alpha=0.7,
                linewidth=1.5,
                zorder=3,
            )

        ax.scatter(
            result[:, 0],
            result[:, 1],
            c=[color],
            s=60,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.5,
            label=f"Scale: {scales[i]:.2f}",
            zorder=4,
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(sm, ax=ax, label="Scale")
    cbar.set_ticks(np.linspace(min(scales), max(scales), 5))
    cbar.set_ticklabels(
        [f"{scale:.2f}" for scale in np.linspace(min(scales), max(scales), 5)]
    )
    cbar.ax.yaxis.label.set_fontsize(12)
    cbar.ax.tick_params(labelsize=10)

    ax.set_title("PCA of Attention Weights", fontsize=14, fontweight="bold")
    ax.set_xlabel("First Principal Component", fontsize=12)
    ax.set_ylabel("Second Principal Component", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_attention_focus_and_sparsity(
    weights: list[np.ndarray],
    scales: list[float],
    threshold: float,
    output_path: str,
) -> None:
    focus_metrics = [np.max(w, axis=1).mean() for w in weights]
    sparsity_metrics = [(w < threshold).mean() for w in weights]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color1 = "tab:blue"
    color2 = "tab:red"

    ax1.set_xlabel("Scale", fontsize=12)
    ax1.set_ylabel("Average Max Attention Weight", color=color1, fontsize=12)
    line1 = ax1.plot(
        scales,
        focus_metrics,
        color=color1,
        marker="o",
        linestyle="-",
        label="Focus",
    )
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.set_ylabel(
        f"Proportion of Weights < {threshold}", color=color2, fontsize=12
    )
    line2 = ax2.plot(
        scales,
        sparsity_metrics,
        color=color2,
        marker="s",
        linestyle="-",
        label="Sparsity",
    )
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, 1)

    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="center right", fontsize=10)

    plt.title(
        "Attention Focus and Sparsity vs. Scale",
        fontsize=14,
        fontweight="bold",
    )
    ax1.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
