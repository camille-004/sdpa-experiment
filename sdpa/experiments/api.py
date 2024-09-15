import json
from pathlib import Path
from typing import Any

import numpy as np

from sdpa.experiments.config import load_config
from sdpa.experiments.results_manager import ResultsManager
from sdpa.experiments.runner import run_experiment
from sdpa.visualization.heatmap import plot_attention_hm


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
        for result in results:
            output_path = (
                output_dir / f"attention_heatmap_scale_{result['scale']}.png"
            )
            plot_attention_hm(
                result["weights"], result["scale"], str(output_path)
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
