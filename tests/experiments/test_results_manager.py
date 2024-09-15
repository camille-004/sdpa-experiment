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
        results = [{"scale": 1.0, "weights": [[0.5, 0.5]]}]
        config = {"batch_size": 1, "seq_length": 2}

        experiment_dir = self.results.save(results, config)

        self.assertTrue(experiment_dir.exists())
        self.assertTrue((experiment_dir / "results.json").exists())
        self.assertTrue((experiment_dir / "config.json").exists())

    def test_load_results(self) -> None:
        results = [{"scale": 1.0, "weights": [[0.5, 0.5]]}]
        config = {"batch_size": 1, "seq_length": 2}
        experiment_dir = self.results.save(results, config)

        data = self.results.load(experiment_dir.name)

        self.assertEqual(data["results"], results)
        self.assertEqual(data["config"], config)

    def test_list_experiments(self) -> None:
        (self.test_dir / "experiment_1").mkdir()
        (self.test_dir / "experiment_2").mkdir()

        experiments = self.results.list_experiments()

        self.assertEqual(set(experiments), {"experiment_1", "experiment_2"})


if __name__ == "__main__":
    unittest.main()
