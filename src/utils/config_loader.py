import yaml
from typing import Any


def load_config(config_path) -> dict[str, Any]:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
