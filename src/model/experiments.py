"""Functions to define different experiments"""

from typing import Any

from src.configs import ml_config, names

from src.model.model import Model


def get_experiment_config(num_experiment: int) -> dict:
    return ml_config.EXPERIMENTS_CONFIGS[num_experiment]


def init_model_from_config(num_experiment: int, config: dict) -> Any:
    if config[names.MODEL] == names.LIGHTGBM:
        return Model(name=f"{ names.LIGHTGBM }_{ num_experiment }")
    else:
        return None
