"""Parameters for ML models"""

from src.configs import constants, names


###############################################################
#                                                             #
#                           METRICS                           #
#                                                             #
###############################################################

TARGET = "target"

METRICS_TO_SAVE = [names.SCORES, names.TRAINING_TIME]


###############################################################
#                                                             #
#                     EXPERIMENTS CONFIGS                     #
#                                                             #
###############################################################

EXPERIMENTS_CONFIGS = {
    0: {
        names.MODEL_TYPE: names.LIGHTGBM,
        "train_ratio": 0.8,
        "feature_selection": None,
        "target": TARGET,
        "features": None,
        "cols_id": "id",
        "training_params": {
            "objective": "binary",
            "metric": "auc",
            "random_seed": constants.RANDOM_SEED,
            "verbose": -1,
            "num_estimators": 10,
            "early_stopping_rounds": 2,
            "learning_rate": 0.1,
        },
    },
    1: {},
    2: {},
    # Add more experiments as needed
}
