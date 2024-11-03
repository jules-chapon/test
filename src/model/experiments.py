"""Functions to define different experiments"""

from src.configs import ml_config, names

from src.model.lgbm_model import LGBMModel, _LGBMModel


def init_model_from_config(id_experiment: int) -> _LGBMModel | None:
    """
    Initialize a model for a given experiment.

    Args:
        id_experiment (int): ID of the experiment.

    Returns:
        _LGBMModel | None: Model with the parameters of the given experiment.
    """
    config = ml_config.EXPERIMENTS_CONFIGS[id_experiment]
    if config[names.MODEL_TYPE] == names.LIGHTGBM:
        return LGBMModel(id_experiment=id_experiment)
    else:
        return None
