import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np

from sdpa.experiments.config import ExperimentConfig
from sdpa.experiments.experiment import Experiment


class TestAPI(unittest.TestCase):
    @patch("sdpa.experiments.experiment.load_config")
    @patch("sdpa.experiments.experiment.ResultsManager")
    def setUp(
        self, MockResultsManager: MagicMock, mock_load_config: MagicMock
    ) -> None:
        self.mock_config = ExperimentConfig(
            batch_size=1,
            seq_length=5,
            d_model=512,
            num_heads=8,
            scaling_factors=[0.1, 1.0, 10.0],
        )
        mock_load_config.return_value = self.mock_config
        self.mock_results_manager = MockResultsManager.return_value
        self.experiment = Experiment("dummy_config.yaml")
        self.experiment.results = self.mock_results_manager

    @patch("sdpa.experiments.experiment.run_experiment")
    @patch("sdpa.experiments.experiment.plot_attention_hm")
    @patch("sdpa.experiments.experiment.plot_attention_dist")
    @patch("sdpa.experiments.experiment.plot_attention_entropy")
    @patch("sdpa.experiments.experiment.plot_attention_focus_and_sparsity")
    @patch("sdpa.experiments.experiment.plot_attention_pca")
    def test_run(
        self,
        mock_pca: MagicMock,
        mock_focus: MagicMock,
        mock_entropy: MagicMock,
        mock_dist: MagicMock,
        mock_hm: MagicMock,
        mock_run_experiment: MagicMock,
    ) -> None:
        mock_results = [
            {"scale": 0.1, "weights": np.random.rand(1, 5, 5)},
            {"scale": 1.0, "weights": np.random.rand(1, 5, 5)},
            {"scale": 10.0, "weights": np.random.rand(1, 5, 5)},
        ]
        mock_run_experiment.return_value = mock_results

        save_mock = MagicMock(return_value=Path("dummy_path"))
        self.mock_results_manager.save = save_mock

        self.experiment.run()

        save_mock.assert_called_once()
        self.assertEqual(mock_hm.call_count, 3)
        mock_dist.assert_called_once()
        mock_entropy.assert_called_once()
        mock_focus.assert_called_once()
        mock_pca.assert_called_once()

    def test_analyze_results(self) -> None:
        results = [
            {"scale": 1.0, "weights": np.array([[[0.1, 0.9], [0.4, 0.6]]])}
        ]

        analysis_results = self.experiment._analyze(results)

        self.assertEqual(len(analysis_results), 1)
        self.assertIn("scale", analysis_results[0])
        self.assertIn("mean_attention_weight", analysis_results[0])
        self.assertIn("max_attention_weight", analysis_results[0])
        self.assertIn("min_attention_weight", analysis_results[0])
        self.assertIn("std_attention_weight", analysis_results[0])
        self.assertIn("entropy", analysis_results[0])
        self.assertIn("focus", analysis_results[0])
        self.assertIn("sparsity", analysis_results[0])
        self.assertIn("pca_first_component_std", analysis_results[0])
        self.assertIn("pca_second_component_std", analysis_results[0])
        self.assertIn(
            "pca_first_component_variance_ratio", analysis_results[0]
        )
        self.assertIn(
            "pca_second_component_variance_ratio", analysis_results[0]
        )

    @patch("sdpa.experiments.experiment.plot_attention_hm")
    @patch("sdpa.experiments.experiment.plot_attention_dist")
    @patch("sdpa.experiments.experiment.plot_attention_entropy")
    @patch("sdpa.experiments.experiment.plot_attention_focus_and_sparsity")
    @patch("sdpa.experiments.experiment.plot_attention_pca")
    def test_load_experiment(
        self,
        mock_pca: MagicMock,
        mock_focus: MagicMock,
        mock_entropy: MagicMock,
        mock_dist: MagicMock,
        mock_hm: MagicMock,
    ) -> None:
        mock_data: dict[str, Any] = {
            "config": {
                "batch_size": 1,
                "seq_length": 5,
                "description": "Test experiment",
            },
            "results": [{"scale": 1.0, "weights": np.random.rand(1, 5, 5)}],
            "analysis": [
                {
                    "scale": 1.0,
                    "mean_attention_weight": 0.5,
                    "max_attention_weight": 0.9,
                    "min_attention_weight": 0.1,
                    "std_attention_weight": 0.2,
                    "entropy": 0.69,
                    "focus": 0.7,
                    "sparsity": 0.1,
                    "pca_first_component_std": 0.3,
                    "pca_second_component_std": 0.2,
                    "pca_first_component_variance_ratio": 0.6,
                    "pca_second_component_variance_ratio": 0.3,
                }
            ],
        }

        load_mock = MagicMock(return_value=mock_data)
        self.mock_results_manager.load = load_mock

        with patch("builtins.print") as mock_print:
            self.experiment.load_experiment("dummy_experiment")

        load_mock.assert_called_once_with("dummy_experiment")
        mock_hm.assert_called_once()
        mock_dist.assert_called_once()
        mock_entropy.assert_called_once()
        mock_focus.assert_called_once()
        mock_pca.assert_called_once()
        mock_print.assert_called()


if __name__ == "__main__":
    unittest.main()
