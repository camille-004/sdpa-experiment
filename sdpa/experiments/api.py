import json
from pathlib import Path
from typing import Any

import numpy as np

from sdpa.experiments.config import load_config
from sdpa.experiments.results_manager import ResultsManager
from sdpa.experiments.runner import run_experiment
from sdpa.experiments.visualization import (
    plot_attention_dist,
    plot_attention_entropy,
    plot_attention_focus_and_sparsity,
    plot_attention_hm,
    plot_attention_pca,
)


class API:
    def __init__(self, config_path: str) -> None:
        self.config = load_config(config_path)
        self.results = ResultsManager()

    def run(self) -> None:
        results = run_experiment(self.config)
        experiment_dir = self.results.save(results, self.config.__dict__)
        self._visualize(results, experiment_dir)
        self._analyze(results)

        print(f"Experiment results saved in: {experiment_dir}")

    def _visualize(
        self, results: list[dict[str, Any]], output_dir: Path
    ) -> None:
        scales = []
        all_weights = []
        weight_dict = {}
        for result in results:
            scale = result["scale"]
            weights = np.array(result["weights"]).squeeze(0)
            scales.append(scale)
            all_weights.append(weights)
            output_base = output_dir / f"scale_{scale}"
            weight_dict[scale] = weights
            plot_attention_hm(
                weights, scale, str(output_base) + "_heatmap.png"
            )

        plot_attention_dist(
            weight_dict, str(output_dir / "attention_distributions.png")
        )
        plot_attention_entropy(
            all_weights, scales, str(output_dir / "attention_entropy.png")
        )
        plot_attention_focus_and_sparsity(
            all_weights, scales, 0.01, str(output_dir / "attention_focus.png")
        )
        plot_attention_pca(
            all_weights, scales, str(output_dir / "attention_pca.png")
        )

    def _analyze(self, results: list[dict[str, Any]]) -> None:
        for result in results:
            scale = result["scale"]
            weights = result["weights"]

            print(f"\nAnalysis for scale {scale}:")
            print(f"Mean attention weight: {np.mean(weights):.4f}")
            print(f"Max attention weight: {np.max(weights):.4f}")
            print(f"Min attention weight: {np.min(weights):.4f}")
            print(f"Standard deviation of weights: {np.std(weights):.4f}")

    def load_experiment(self, experiment_name: str) -> None:
        data: dict[str, Any] = self.results.load(experiment_name)
        print("Loaded experiment config:")
        print(json.dumps(data["config"], indent=2))
        self._visualize(
            data["results"], self.results.base_dir / experiment_name
        )
        self._analyze(data["results"])
