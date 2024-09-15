import os
from pathlib import Path

import yaml
from dotenv import load_dotenv


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_env_var(var_name: str, default=None):
    load_dotenv()
    return os.getenv(var_name, default)
