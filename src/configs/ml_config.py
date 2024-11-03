"""Parameters for ML models"""

from src.configs import constants, names


###############################################################
#                                                             #
#                           METRICS                           #
#                                                             #
###############################################################

TARGET = "target"

METRICS_TO_SAVE = [names.SCORES, names.TIMES]


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
            "metrics": ["binary_logloss", "auc"],
            "random_seed": constants.RANDOM_SEED,
            "verbose": -1,
            "n_estimators": 10,
            "early_stopping_rounds": 9,
            "learning_rate": 0.1,
            "max_depth": 15,
            "max_leaves": 31,
            "min_data_per_leaf": 20,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "feature_fraction": 0.8,
            "lambda_l1": 0.1,
        },
        "cross_validation": 3,
        "fine_tuning": False,
    },
    1: {},
    2: {},
    # Add more experiments as needed
}
