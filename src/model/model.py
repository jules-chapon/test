"""Class for model"""

import abc
import logging
import typing
import pandas as pd

from src.configs import ml_config

from src.libs.preprocessing import (
    preprocessing_learning_data,
    preprocessing_testing_data,
)

from src.libs.feature_selection import (
    selecting_features_with_random_columns,
    selecting_features_with_boruta,
)


_Model = typing.TypeVar("_Model", bound="Model")


class Model(abc.ABC):
    """Abstract base class for ML models"""

    def __init__(self, name: str, num_experiment: int | None) -> None:
        self.name = name
        self.num_experiment = num_experiment

    def get_model_config(self: _Model) -> None:
        if self.num_experiment is not None:
            self.config = ml_config.EXPERIMENTS_CONFIGS[self.num_experiment]
        else:
            self.config = {}

    def learning_preprocessing(
        self: _Model, df_learning: pd.DataFrame
    ) -> tuple[pd.DataFrame]:
        df_train, df_valid = preprocessing_learning_data(df_learning)
        return df_train, df_valid

    def testing_preprocessing(self: _Model, df_testing: pd.DataFrame) -> pd.DataFrame:
        df_test = preprocessing_testing_data(df_testing)
        return df_test

    def dropping_id_columns(self: _Model, df: pd.DataFrame) -> None:
        return df.drop(columns=self.config["cols_id"])

    def selecting_features(self: _Model, df_train: pd.DataFrame) -> None:
        if "random_columns" in self.config["feature_selection"]:
            self.config["features"] = selecting_features_with_random_columns(
                df=df_train,
                features=self.config["features"],
                target=self.config["target"],
            )
        if "boruta" in self.config["feature_selection"]:
            self.config["features"] = selecting_features_with_boruta(
                df=df_train,
                features=self.config["features"],
                target=self.config["target"],
            )
        return None

    def dropping_useless_columns(
        self: _Model,
        df_train: pd.DataFrame,
        df_valid: pd.DataFrame,
        df_test: pd.DataFrame,
    ) -> tuple[pd.DataFrame]:
        df_train_without_ids = self.dropping_id_columns(df=df_train)
        self.selecting_features(df_train=df_train_without_ids)
        cols_to_keep = self.config["features"]
        if isinstance(cols_to_keep, str):
            cols_to_keep = [cols_to_keep]
        return df_train[cols_to_keep], df_valid[cols_to_keep], df_test[cols_to_keep]

    @abc.abstractmethod
    def fine_tuning(
        self: _Model, df_train: pd.DataFrame, df_valid: pd.DataFrame
    ) -> None:
        return None

    @abc.abstractmethod
    def training(self: _Model, df_train: pd.DataFrame, df_valid: pd.DataFrame) -> None:
        return None

    @abc.abstractmethod
    def predicting(self: _Model, df_to_predict: pd.DataFrame) -> pd.Series:
        """
        Predict outputs for a given dataset.

        Args:
            self (_Model): Class instance.
            df_to_predict (pd.DataFrame): DataFrame to predict.

        Returns:
            pd.Series: Predictions.
        """

    @abc.abstractmethod
    def scoring(self: _Model, df_to_evaluate: pd.DataFrame) -> dict:
        """
        Get scores for a given dataset.

        Args:
            self (_Model): Class instance.
            df_to_evaluate (pd.DataFrame): DataFrame to evaluate.

        Returns:
            dict: Dictionary with metric names as keys and values as values.
        """

    def getting_scores(
        self: _Model,
        df_train: pd.DataFrame,
        df_valid: pd.DataFrame,
        df_test: pd.DataFrame,
    ) -> None:
        dict_scores = {}
        dict_scores["train"] = self.scoring(df_to_evaluate=df_train)
        dict_scores["valid"] = self.scoring(df_to_evaluate=df_valid)
        dict_scores["test"] = self.scoring(df_to_evaluate=df_test)
        self.config["scores"] = dict_scores

    @abc.abstractmethod
    def saving_metric(self: _Model, metric_name: str) -> None:
        """
        Save a given metric.

        Args:
            self (_Model): Class instance.
            metric_name (str): Name of the metric to save.
        """

    @abc.abstractmethod
    def saving_model(self: _Model) -> None:
        """
        Save model.

        Args:
            self (_Model): Class instance.
        """

    @classmethod
    @abc.abstractmethod
    def loading(cls, name: str) -> _Model:
        """
        Load saved model.
        """

    def full_pipeline(
        self: _Model, df_learning: pd.DataFrame, df_testing: pd.DataFrame
    ) -> None:
        self.get_model_config()
        df_train, df_valid = self.learning_preprocessing(df_learning)
        df_test = self.testing_preprocessing(df_testing)
        df_train_clean, df_valid_clean, df_test_clean = self.dropping_useless_columns(
            df_train, df_valid, df_test
        )
        self.fine_tuning(df_train_clean, df_valid_clean)
        self.training(df_train_clean, df_valid_clean)
        self.getting_scores(df_train_clean, df_valid_clean, df_test_clean)
        for metric_name in [ml_config.METRICS_TO_SAVE]:
            self.saving_metric(metric_name)
        self.saving_model()
        logging.info(f"Model { self.name } trained and saved.")
