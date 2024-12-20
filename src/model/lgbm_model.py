"""Class for model"""

import lightgbm as lgb
import os
import optuna
import typing
import numpy as np
import pandas as pd
from collections import defaultdict

from src.model.model import Model

from src.configs import ml_config, names


from src.libs.visualization import plot_training_curves


_LGBMModel = typing.TypeVar("_LGBMModel", bound="Model")


class LGBMModel(Model):
    """Class for LGBM models"""

    def __init__(self: _LGBMModel, id_experiment: int | None) -> None:
        """
        Initialize the LGBM model.

        Args:
            self (_LGBMModel): Class instance.
            id_experiment (int | None): ID of the experiment.
        """
        super().__init__(id_experiment)

    def train(
        self: _LGBMModel,
        df_train: pd.DataFrame,
    ) -> None:
        """
        Train the model.

        Args:
            self (_LGBMModel): Class instance.
            df_train (pd.DataFrame): Train set (whole learning set).
        """
        evals_result = {}
        train_data = lgb.Dataset(
            df_train[self.config[names.FEATURES]],
            label=df_train[self.config[names.TARGET]],
        )
        model = lgb.train(
            params=self.config[names.TRAINING_PARAMS],
            train_set=train_data,
            valid_sets=train_data,
            valid_names="train",
            callbacks=[
                lgb.record_evaluation(evals_result),
            ],
        )
        train_scores = evals_result["train"]
        output_folder = os.path.join(self.model_path, "testing")
        os.makedirs(output_folder, exist_ok=True)
        plot_training_curves(
            dict_metrics=train_scores,
            output_path=os.path.join(output_folder, "training_curves.png"),
        )
        scores = defaultdict(dict)
        for metric in self.config[names.TRAINING_PARAMS]["metrics"]:
            scores["train"][metric] = train_scores[metric]
        self.testing_scores["train"] = {
            "scores": scores["train"],
            "best_scores": model.best_score,
        }
        self.model = model

    def learn(
        self: _LGBMModel,
        df_train: pd.DataFrame,
        df_valid: pd.DataFrame,
    ) -> None:
        """
        Learning function.
        Get the loss of the model on both train and valid sets.
        Used to find the best hyperparameters of the model.

        Args:
            self (_LGBMModel): Class instance.
            df_train (pd.DataFrame): Train set.
            df_valid (pd.DataFrame): Valid set.
        """
        evals_result = {}
        train_data = lgb.Dataset(
            df_train[self.config[names.FEATURES]],
            label=df_train[self.config[names.TARGET]],
        )
        valid_data = lgb.Dataset(
            df_valid[self.config[names.FEATURES]],
            label=df_valid[self.config[names.TARGET]],
        )
        model = lgb.train(
            params=self.config[names.TRAINING_PARAMS],
            train_set=train_data,
            valid_sets=[train_data, valid_data],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.record_evaluation(evals_result),
            ],
        )
        scores = {
            "train": evals_result["train"],
            "valid": evals_result["valid"],
        }
        for dataset in ["train", "valid"]:
            for metric in self.config[names.TRAINING_PARAMS]["metrics"]:
                self.learning_scores[dataset][metric]["scores"].append(
                    scores[dataset][metric]
                )
                self.learning_scores[dataset][metric]["best_scores"].append(
                    model.best_score[dataset][metric]
                )

    def fine_tune_objective(
        self: _LGBMModel,
        df_train: pd.DataFrame,
        df_valid: pd.DataFrame | None,
        trial: optuna.Trial,
    ) -> None:
        """
        Define the objective function for the fine-tuning of the hyperparameters.

        Args:
            self (_LGBMModel): Class instance.
            df_train (pd.DataFrame): Train set.
            df_valid (pd.DataFrame): Valid set.
        """
        params = self.config[names.TRAINING_PARAMS].copy()
        params["learning_rate"] = trial.suggest_float(
            "learning_rate", 0.01, 0.3, log=True
        )
        params["max_depth"] = trial.suggest_int("max_depth", 3, 15)
        params["num_leaves"] = trial.suggest_int("num_leaves", 20, 150)
        params["min_data_in_leaf"] = trial.suggest_int("min_data_in_leaf", 10, 100)
        params["bagging_fraction"] = trial.suggest_float("bagging_fraction", 0.3, 1.0)
        params["bagging_freq"] = trial.suggest_int("bagging_freq", 1, 7)
        params["feature_fraction"] = trial.suggest_float("feature_fraction", 0.3, 1.0)
        params["lambda_l1"] = trial.suggest_float("lambda_l1", 0.01, 1.0)
        if df_valid is None:
            self.cross_validate(df_train=df_train)
            return self.learning_scores["cross_validation"][
                (self.config[names.TRAINING_PARAMS]["metrics"])[0]
            ]["average_score"]
        else:
            self.learn(df_train=df_train, df_valid=df_valid)
            return self.learning_scores["valid"][
                (self.config[names.TRAINING_PARAMS]["metrics"])[0]
            ]["best_scores"][0]

    def fine_tune(
        self: _LGBMModel,
        df_train: pd.DataFrame,
        df_valid: pd.DataFrame,
    ) -> None:
        """
        Fine-tune hypermarameters of the model.

        Args:
            self (_LGBMModel) : Class instance.
            df_train (pd.DataFrame): Training set.
            df_valid (pd.DataFrame): Validation set.
        """
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: self.fine_tuning_objective(df_train, df_valid, trial),
            n_trials=ml_config.NB_OPTUNA_TRIALS,
        )
        self.config[names.TRAINING_PARAMS].update(study.best_params)

    def predict(self: _LGBMModel, df_to_predict: pd.DataFrame) -> np.ndarray:
        """
        Predict outputs for a given dataset.

        Args:
            self (_LGBMModel): Class instance.
            df_to_predict (pd.DataFrame): DataFrame to predict.

        Returns:
            np.ndarray: Predictions.
        """
        df_to_predict = df_to_predict[self.config[names.FEATURES]]
        predictions_proba = self.model.predict(df_to_predict)
        predictions_label = np.round(predictions_proba)
        return predictions_label
