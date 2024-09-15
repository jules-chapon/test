"""Class for model"""

import abc
import os
import logging
import typing
import pandas as pd

from src.configs import ml_config, constants, names

from src.libs.preprocessing import (
    load_data,
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

    def __init__(self, num_experiment: int | None) -> None:
        self.num_experiment = num_experiment

    def get_model_config(self: _Model) -> None:
        if self.num_experiment is not None:
            self.config = ml_config.EXPERIMENTS_CONFIGS[self.num_experiment]
        else:
            logging.info("No corresponding experiment")

    def define_model_path(self: _Model) -> None:
        self.model_path = os.path.join(
            constants.OUTPUT_FOLDER,
            f"{ self.num_experiment }_{ self.config[names.MODEL_TYPE] }",
        )
        os.makedirs(self.model_path, exist_ok=True)

    def learning_preprocessing(
        self: _Model, df_learning: pd.DataFrame
    ) -> tuple[pd.DataFrame]:
        df_train, df_valid = preprocessing_learning_data(
            df_learning, self.config["train_ratio"]
        )
        return df_train, df_valid

    def testing_preprocessing(self: _Model, df_testing: pd.DataFrame) -> pd.DataFrame:
        df_test = preprocessing_testing_data(df_testing)
        return df_test

    def dropping_id_columns(self: _Model, df: pd.DataFrame) -> None:
        return df.drop(columns=self.config["cols_id"])

    def selecting_features(self: _Model, df_train: pd.DataFrame) -> None:
        if self.config["features"] is None:
            self.config["features"] = df_train.columns.tolist()
        elif "random_columns" in self.config["feature_selection"]:
            self.config["features"] = selecting_features_with_random_columns(
                df=df_train,
                features=self.config["features"],
                target=self.config["target"],
            )
        elif "boruta" in self.config["feature_selection"]:
            self.config["features"] = selecting_features_with_boruta(
                df=df_train,
                features=self.config["features"],
                target=self.config["target"],
            )
        return None

    def getting_columns_to_keep(
        self: _Model,
        df_train: pd.DataFrame,
    ) -> tuple[pd.DataFrame]:
        df_train_without_ids = self.dropping_id_columns(df=df_train)
        self.selecting_features(df_train=df_train_without_ids)
        if isinstance(self.config["features"], str):
            self.config["features"] = [self.config["features"]]
        self.config["features"] = [
            col for col in self.config["features"] if col not in self.config["target"]
        ]
        return None

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
        self.scores = dict_scores

    @abc.abstractmethod
    def saving_config(
        self: _Model,
        output_dir: str | None,
    ) -> None:
        """
        Save the config of the experiment.

        Args:
            self (_Model): Class instance.
            output_dir (str | None): Directory to save the config.
        """

    @abc.abstractmethod
    def saving_metric(self: _Model, output_dir: str | None, metric_name: str) -> None:
        """
        Save a given metric.

        Args:
            self (_Model): Class instance.
            output_dir (str | None): Directory to save the metric.
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
        self.define_model_path()
        df_train, df_valid = self.learning_preprocessing(df_learning)
        df_test = self.testing_preprocessing(df_testing)
        self.getting_columns_to_keep(df_train)
        # self.fine_tuning(df_train, df_valid)
        self.training(df_train, df_valid)
        self.getting_scores(df_train, df_valid, df_test)
        for metric_name in ml_config.METRICS_TO_SAVE:
            self.saving_metric(output_dir=self.model_path, metric_name=metric_name)
        self.saving_config(self.model_path)
        self.saving_model()
        logging.info(f"Model { self.num_experiment } trained and saved.")
