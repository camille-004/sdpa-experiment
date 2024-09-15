import json
from datetime import datetime
from pathlib import Path
from typing import Any


class ResultsManager:
    def __init__(self, base_dir: str = "results") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        results: list[dict[str, Any]],
        config: dict[str, Any],
        analysis: list[dict[str, Any]],
    ) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        description = config.get("description", "").replace(" ", "_")[:30]
        batch_size = config.get("batch_size", "")
        seq_length = config.get("seq_length", "")

        if config["d_model"] is not None and config["num_heads"] is not None:
            model_info = f"d{config['d_model']}_h{config['num_heads']}"
        elif config["d_k"] is not None:
            model_info = f"dk{config['d_k']}"
        else:
            model_info = ""
        experiment_name = f"{timestamp}_b{batch_size}_s{seq_length}_{model_info}_{description}"  # noqa
        experiment_name = experiment_name.rstrip("_")
        experiment_dir = self.base_dir / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)

        results_file = experiment_dir / "results.json"
        results_file.write_text(json.dumps(results, indent=2))

        config_file = experiment_dir / "config.json"
        config_file.write_text(json.dumps(config, indent=2))

        analysis_file = experiment_dir / "analysis.json"
        analysis_file.write_text(json.dumps(analysis, indent=2))

        return experiment_dir

    def load(self, experiment_name: str) -> dict[str, Any]:
        experiment_dir = self.base_dir / experiment_name

        results_file = experiment_dir / "results.json"
        results = json.loads(results_file.read_text())

        config_file = experiment_dir / "config.json"
        config = json.loads(config_file.read_text())

        analysis_file = experiment_dir / "analysis.json"
        analysis = json.loads(analysis_file.read_text())

        return {"results": results, "config": config, "analysis": analysis}

    def list_experiments(self) -> list[str]:
        return [d.name for d in self.base_dir.iterdir() if d.is_dir()]
