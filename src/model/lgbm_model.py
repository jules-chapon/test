"""Class for model"""

import json
import lightgbm as lgb
import logging
import os
import pickle as pkl
import typing
import numpy as np
import pandas as pd
import time

from sklearn.metrics import accuracy_score

from src.configs import constants, ml_config, names

from src.model.model import Model


_LGBMModel = typing.TypeVar("_LGBMModel", bound="Model")


class LGBMModel(Model):
    """Class for LGBM models"""

    def __init__(self: _LGBMModel, num_experiment: int | None) -> None:
        super().__init__(num_experiment)

    def fine_tuning(
        self: _LGBMModel, df_train: pd.DataFrame, df_valid: pd.DataFrame
    ) -> None:
        return None

    def training(
        self: _LGBMModel, df_train: pd.DataFrame, df_valid: pd.DataFrame
    ) -> None:
        start_time = time.time()
        train_data = lgb.Dataset(
            df_train[self.config["features"]], label=df_train[self.config["target"]]
        )
        valid_data = lgb.Dataset(
            df_valid[self.config["features"]], label=df_valid[self.config["target"]]
        )
        lgbm_model = lgb.train(
            params=self.config["training_params"],
            train_set=train_data,
            valid_sets=[train_data, valid_data],
        )
        self.model = lgbm_model
        self.training_time = time.time() - start_time
        return None

    def predicting(self: _LGBMModel, df_to_predict: pd.DataFrame) -> np.ndarray:
        """
        Predict outputs for a given dataset.

        Args:
            self (_LGBMModel): Class instance.
            df_to_predict (pd.DataFrame): DataFrame to predict.

        Returns:
            np.ndarray: Predictions.
        """
        df_to_predict = df_to_predict[self.config["features"]]
        predictions_proba = self.model.predict(df_to_predict)
        predictions_label = np.round(predictions_proba)
        return predictions_label

    def scoring(self: _LGBMModel, df_to_evaluate: pd.DataFrame) -> dict:
        """
        Get scores for a given dataset.

        Args:
            self (_LGBMModel): Class instance.
            df_to_evaluate (pd.DataFrame): DataFrame to evaluate.

        Returns:
            dict: Dictionary with metric names as keys and values as values.
        """
        label_pred = self.predicting(df_to_evaluate)
        label_true = df_to_evaluate[self.config["target"]]
        accuracy = accuracy_score(label_true, label_pred)
        return {"accuracy": accuracy}

    def saving_config(self: _LGBMModel, output_dir: str | None) -> None:
        """
        Save a given metric.

        Args:
            self (_LGBMModel): Class instance.
            output_dir (str | None): Output directory.
        """
        if output_dir is not None:
            with open(os.path.join(output_dir, "config.json"), "w") as f:
                json.dump(self.config, f, indent=4)

    def saving_metric(
        self: _LGBMModel, output_dir: str | None, metric_name: str
    ) -> None:
        """
        Save a given metric.

        Args:
            self (_LGBMModel): Class instance.
            metric_name (str): Name of the metric to save.
        """
        if output_dir is not None:
            with open(os.path.join(output_dir, f"{ metric_name }.json"), "w") as f:
                if metric_name == names.SCORES:
                    json.dump(self.scores, f, indent=4)
                elif metric_name == names.TRAINING_TIME:
                    json.dump(self.training_time, f, indent=4)
                else:
                    logging.info(f"No metric named : { metric_name }")

    def saving_model(self: _LGBMModel) -> None:
        """
        Save model.

        Args:
            self (_LGBMModel): Class instance.
        """
        with open(os.path.join(self.model_path, "model.pkl"), "wb") as file:
            pkl.dump(self, file)

    def loading(cls, name: str) -> _LGBMModel:
        """
        Load saved model.
        """
