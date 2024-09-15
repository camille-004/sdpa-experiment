import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from sdpa.experiments.api import API
from sdpa.experiments.config import ExperimentConfig


class TestAPI(unittest.TestCase):
    @patch("sdpa.experiments.api.load_config")
    @patch("sdpa.experiments.api.ResultsManager")
    def setUp(self, mock_results, mock_load_config) -> None:
        self.mock_config = ExperimentConfig(
            batch_size=1,
            seq_length=5,
            d_model=512,
            d_k=64,
            d_v=64,
            num_heads=8,
            scaling_factors=[0.1, 1.0, 10.0],
        )
        mock_load_config.return_value = self.mock_config
        self.api = API("dummy_config.yaml")

    @patch("sdpa.experiments.api.run_experiment")
    @patch("sdpa.experiments.api.plot_attention_hm")
    def test_run(self, mock_plot, mock_run_experiment) -> None:
        mock_results = [
            {"scale": 0.1, "weights": np.random.rand(5, 5)},
            {"scale": 1.0, "weights": np.random.rand(5, 5)},
            {"scale": 10.0, "weights": np.random.rand(5, 5)},
        ]
        mock_run_experiment.return_value = mock_results
        self.api.results.save.return_value = Path("dummy_path")

        self.api.run()

        self.api.results.save.assert_called_once()
        self.assertEqual(mock_plot.call_count, 3)

    def test_analyze_results(self) -> None:
        results = [
            {"scale": 1.0, "weights": np.array([[0.1, 0.9], [0.4, 0.6]])}
        ]

        with patch("builtins.print") as mock_print:
            self.api._analyze(results)

        mock_print.assert_called()

    @patch("sdpa.experiments.api.plot_attention_hm")
    def test_load_experiment(self, mock_plot) -> None:
        mock_data = {
            "config": {"batch_size": 1, "seq_length": 5},
            "results": [{"scale": 1.0, "weights": np.random.rand(5, 5)}],
        }
        self.api.results.load.return_value = mock_data

        with patch("builtins.print") as mock_print:
            self.api.load_experiment("dummy_experiment")

        self.api.results.load.assert_called_once_with("dummy_experiment")
        mock_plot.assert_called_once()
        mock_print.assert_called()


if __name__ == "__main__":
    unittest.main()
