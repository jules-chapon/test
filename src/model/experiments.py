"""Functions to define different experiments"""

from src.configs import ml_config, names

from src.model.lgbm_model import LGBMModel, _LGBMModel


def init_model_from_config(num_experiment: int) -> _LGBMModel | None:
    config = ml_config.EXPERIMENTS_CONFIGS[num_experiment]
    if config[names.MODEL_TYPE] == names.LIGHTGBM:
        return LGBMModel(num_experiment=num_experiment)
    else:
        return None
