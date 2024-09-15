from dataclasses import dataclass

import yaml


@dataclass
class ExperimentConfig:
    batch_size: int
    seq_length: int
    d_model: int
    d_k: int
    d_v: int
    num_heads: int
    scaling_factors: list[float]


def load_config(config_path: str) -> ExperimentConfig:
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return ExperimentConfig(**config_dict)
