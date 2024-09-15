import shutil
import unittest
from pathlib import Path

from sdpa.experiments.results_manager import ResultsManager


class TestResultsManager(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = Path("test_experiments")
        self.results = ResultsManager(str(self.test_dir))

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)

    def test_save_results(self) -> None:
        results = [{"scale": 1.0, "weights": [[[0.5, 0.5]]]}]
        config = {
            "batch_size": 1,
            "seq_length": 2,
            "d_model": 8,
            "num_heads": 2,
            "scaling_factors": [0.1, 1.0],
            "description": "Test experiment",
        }
        analysis = [
            {
                "scale": 1.0,
                "mean_attention_weight": 0.5,
                "max_attention_weight": 0.5,
                "min_attention_weight": 0.5,
                "std_attention_weight": 0.0,
                "entropy": 0.69314718,
                "focus": 0.5,
                "sparsity": 0.0,
                "pca_first_component_std": 0.0,
                "pca_second_component_std": 0.0,
                "pca_first_component_variance_ratio": 0.0,
                "pca_second_component_variance_ratio": 0.0,
            }
        ]

        experiment_dir = self.results.save(results, config, analysis)

        self.assertTrue(experiment_dir.exists())
        self.assertTrue((experiment_dir / "results.json").exists())
        self.assertTrue((experiment_dir / "config.json").exists())
        self.assertTrue((experiment_dir / "analysis.json").exists())
        self.assertIn("Test_experiment", experiment_dir.name)

    def test_load_results(self) -> None:
        results = [{"scale": 1.0, "weights": [[[0.5, 0.5]]]}]
        config = {
            "batch_size": 1,
            "seq_length": 2,
            "d_model": 8,
            "num_heads": 2,
            "scaling_factors": [0.1, 1.0],
        }
        analysis = [
            {
                "scale": 1.0,
                "mean_attention_weight": 0.5,
                "max_attention_weight": 0.5,
                "min_attention_weight": 0.5,
                "std_attention_weight": 0.0,
                "entropy": 0.69314718,
                "focus": 0.5,
                "sparsity": 0.0,
                "pca_first_component_std": 0.0,
                "pca_second_component_std": 0.0,
                "pca_first_component_variance_ratio": 0.0,
                "pca_second_component_variance_ratio": 0.0,
            }
        ]
        experiment_dir = self.results.save(results, config, analysis)

        data = self.results.load(experiment_dir.name)

        self.assertEqual(data["results"], results)
        self.assertEqual(data["config"], config)
        self.assertEqual(data["analysis"], analysis)

    def test_list_experiments(self) -> None:
        (self.test_dir / "experiment_1").mkdir()
        (self.test_dir / "experiment_2").mkdir()

        experiments = self.results.list_experiments()

        self.assertEqual(set(experiments), {"experiment_1", "experiment_2"})


if __name__ == "__main__":
    unittest.main()
