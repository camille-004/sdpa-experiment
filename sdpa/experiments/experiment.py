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
from sdpa.utils.math_utils import (
    calc_entropy,
    calc_focus,
    calc_pca,
    calc_sparsity,
    round_floats,
)


class Experiment:
    def __init__(self, config_path: str) -> None:
        self.config = load_config(config_path)
        self.results = ResultsManager()

    def run(self) -> None:
        results = run_experiment(self.config)
        analysis_results = self._analyze(results)
        experiment_dir = self.results.save(
            results, self.config.__dict__, analysis_results
        )
        self._visualize(results, experiment_dir)

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

    def _analyze(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        analysis_results = []

        for result in results:
            scale = result["scale"]
            weights = np.array(result["weights"])

            pca_result = calc_pca([weights])

            analysis = {
                "scale": scale,
                "mean_attention_weight": round_floats(float(np.mean(weights))),
                "max_attention_weight": round_floats(float(np.max(weights))),
                "min_attention_weight": round_floats(float(np.min(weights))),
                "std_attention_weight": round_floats(float(np.std(weights))),
                "entropy": round_floats(calc_entropy(weights)),
                "focus": round_floats(calc_focus(weights)),
                "sparsity": round_floats(
                    calc_sparsity(weights, threshold=0.01)
                ),
                "pca_first_component_std": round_floats(
                    np.std(pca_result[1][:, 0])
                ),
                "pca_second_component_std": round_floats(
                    np.std(pca_result[1][:, 1])
                ),
                "pca_first_component_variance_ratio": round_floats(
                    pca_result[0].explained_variance_ratio_[0]
                ),
                "pca_second_component_variance_ratio": round_floats(
                    pca_result[0].explained_variance_ratio_[1]
                ),
            }
            analysis_results.append(analysis)

        return analysis_results

    def load_experiment(self, experiment_name: str) -> None:
        data: dict[str, Any] = self.results.load(experiment_name)
        print("Loaded experiment config:")
        print(json.dumps(data["config"], indent=2))
        self._visualize(
            data["results"], self.results.base_dir / experiment_name
        )
        self._analyze(data["results"])
