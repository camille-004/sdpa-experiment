import json
from datetime import datetime
from pathlib import Path
from typing import Any


class ResultsManager:
    def __init__(self, base_dir: str = "results") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self, results: list[dict[str, Any]], config: dict[str, Any]
    ) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"sdpa_{timestamp}"
        experiment_dir = self.base_dir / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)

        results_file = experiment_dir / "results.json"
        results_file.write_text(json.dumps(results, indent=2))

        config_file = experiment_dir / "config.json"
        config_file.write_text(json.dumps(config, indent=2))

        return experiment_dir

    def load(self, experiment_name: str) -> dict[str, Any]:
        experiment_dir = self.base_dir / experiment_name

        results_file = experiment_dir / "results.json"
        results = json.loads(results_file.read_text())

        config_file = experiment_dir / "config.json"
        config = json.loads(config_file.read_text())

        return {"results": results, "config": config}

    def list_experiments(self) -> list[str]:
        return [d.name for d in self.base_dir.iterdir() if d.is_dir()]
