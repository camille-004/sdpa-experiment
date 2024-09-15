from dataclasses import dataclass

import yaml


@dataclass
class ExperimentConfig:
    batch_size: int
    seq_length: int
    scaling_factors: list[float]
    d_k: int | None = None
    d_model: int | None = None
    num_heads: int | None = None


def load_config(config_path: str) -> ExperimentConfig:
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return ExperimentConfig(**config_dict)
