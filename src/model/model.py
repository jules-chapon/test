"""Class for model"""

import abc
import os
import logging
import json
import typing
import time
import pickle as pkl
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from collections import defaultdict

from src.configs import ml_config, constants, names

from src.libs.useful_functions import (
    double_nested_defaultdict,
    triple_nested_defaultdict,
)

from src.libs.preprocessing import (
    preprocessing_learning_data,
    preprocessing_testing_data,
)

from src.libs.feature_selection import (
    selecting_features_with_correlations,
    selecting_features_with_random_columns,
    selecting_features_with_boruta,
)

from src.libs.evaluation import (
    compute_evaluation_metrics,
    plot_confusion_matrix,
    plot_all_scatter_feature_given_label,
    plot_describe_results,
)


_Model = typing.TypeVar("_Model", bound="Model")


class Model(abc.ABC):
    """Abstract base class for ML models"""

    def __init__(self, id_experiment: int | None) -> None:
        self.id_experiment = id_experiment
        self.learning_scores = triple_nested_defaultdict()
        self.testing_scores = triple_nested_defaultdict()
        self.times = double_nested_defaultdict()

    def get_model_config(self: _Model) -> None:
        if self.id_experiment in ml_config.EXPERIMENTS_CONFIGS:
            self.config = ml_config.EXPERIMENTS_CONFIGS[self.id_experiment]
        else:
            logging.info("No corresponding experiment")
        if isinstance(self.config["training_params"]["metrics"], str):
            self.config["training_params"]["metrics"] = [
                self.config["training_params"]["metrics"]
            ]
        if isinstance(self.config["features"], str):
            self.config["features"] = [self.config["features"]]

    def define_model_path(self: _Model) -> None:
        self.model_path = os.path.join(
            constants.OUTPUT_FOLDER,
            f"{ self.id_experiment }_{ self.config[names.MODEL_TYPE] }",
        )
        os.makedirs(self.model_path, exist_ok=True)

    def initialize_scores(self: _Model) -> None:
        for dataset in ["train", "valid"]:
            for metric in self.config["training_params"]["metrics"]:
                for score in ["scores", "best_scores"]:
                    self.learning_scores[dataset][metric][score] = []
                    if dataset == "train":
                        self.testing_scores[dataset][metric][score] = []
        self.testing_scores["train"]["metrics"] = {}
        self.testing_scores["test"]["metrics"] = {}
        return None

    def preprocess_learning_set(
        self: _Model, df_learning: pd.DataFrame
    ) -> tuple[pd.DataFrame]:
        if self.config["train_ratio"] == 1:
            df_train = preprocessing_learning_data(
                df_learning=df_learning, train_ratio=self.config["train_ratio"]
            )
            df_valid = None
        else:
            df_train, df_valid = preprocessing_learning_data(
                df_learning=df_learning, train_ratio=self.config["train_ratio"]
            )
        return df_train, df_valid

    def preprocess_training_set(
        self: _Model, df_learning: pd.DataFrame
    ) -> tuple[pd.DataFrame]:
        df_train = preprocessing_learning_data(df_learning=df_learning, train_ratio=1)
        return df_train

    def preprocess_testing_set(self: _Model, df_testing: pd.DataFrame) -> pd.DataFrame:
        df_test = preprocessing_testing_data(df_testing)
        return df_test

    def drop_id_columns(self: _Model, df: pd.DataFrame) -> None:
        return df.drop(columns=self.config["cols_id"])

    def select_features(self: _Model, df_train: pd.DataFrame) -> None:
        if self.config["features"] is None:
            self.config["features"] = df_train.columns.tolist()
        if self.config["feature_selection"] is not None:
            if "correlation" in self.config["feature_selection"]:
                self.config["features"] = selecting_features_with_correlations(
                    df=df_train,
                    features=self.config["features"],
                    target=self.config["target"],
                )
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

    def get_columns_to_keep(
        self: _Model,
        df_train: pd.DataFrame,
    ) -> tuple[pd.DataFrame]:
        df_train_without_ids = self.drop_id_columns(df=df_train)
        self.select_features(df_train=df_train_without_ids)
        self.config["features"] = [
            col for col in self.config["features"] if col not in self.config["target"]
        ]
        return None

    @abc.abstractmethod
    def learn(
        self: _Model,
        df_train: pd.DataFrame,
        df_valid: pd.DataFrame,
    ) -> None:
        return None

    def cross_validate(self: _Model, df_train: pd.DataFrame):
        kf = KFold(
            n_splits=self.config["cross_validation"],
            shuffle=True,
            random_state=constants.RANDOM_SEED,
        )
        scores = defaultdict(list)
        for train_index, valid_index in kf.split(df_train):
            df_train_cv = df_train.iloc[train_index]
            df_valid_cv = df_train.iloc[valid_index]
            self.learn(df_train=df_train_cv, df_valid=df_valid_cv)
        for metric in self.config["training_params"]["metrics"]:
            scores[metric] = self.learning_scores["valid"][metric]["best_scores"]
        for metric, scores_list in scores.items():
            self.learning_scores["cross_validation"][metric]["average_score"] = np.mean(
                scores_list
            )
        return None

    @abc.abstractmethod
    def fine_tune_objective(
        self: _Model, df_train: pd.DataFrame, df_valid: pd.DataFrame
    ) -> None:
        return None

    @abc.abstractmethod
    def fine_tune(self: _Model, df_train: pd.DataFrame, df_valid: pd.DataFrame) -> None:
        return None

    @abc.abstractmethod
    def train(
        self: _Model,
        df_train: pd.DataFrame,
    ) -> None:
        return None

    @abc.abstractmethod
    def predict(self: _Model, df_to_predict: pd.DataFrame) -> pd.Series:
        """
        Predict outputs for a given dataset.

        Args:
            self (_Model): Class instance.
            df_to_predict (pd.DataFrame): DataFrame to predict.

        Returns:
            pd.Series: Predictions.
        """

    def score(self: _Model, df_to_evaluate: pd.DataFrame) -> dict:
        """
        Get scores for a given dataset.

        Args:
            self (_Model): Class instance.
            df_to_evaluate (pd.DataFrame): DataFrame to evaluate.

        Returns:
            dict: Dictionary with metric names as keys and values as values.
        """
        label_pred = self.predict(df_to_evaluate)
        label_true = df_to_evaluate[self.config["target"]]
        metrics = compute_evaluation_metrics(label_true, label_pred)
        return metrics

    def get_scores(
        self: _Model,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
    ) -> None:
        self.testing_scores["train"]["metrics"] = self.score(df_to_evaluate=df_train)
        self.testing_scores["test"]["metrics"] = self.score(df_to_evaluate=df_test)

    def confusion_matrix(self: _Model, df_test: pd.DataFrame) -> None:
        output_folder = os.path.join(self.model_path, "testing")
        os.makedirs(output_folder, exist_ok=True)
        y_true = df_test[names.TARGET]
        y_pred = self.predict(df_test)
        plot_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            output_path=os.path.join(output_folder, "confusion_matrix.png"),
        )

    def save_plots_errors(self: _Model, df_test: pd.DataFrame) -> None:
        output_folder = os.path.join(self.model_path, "testing", "plots")
        os.makedirs(output_folder, exist_ok=True)
        df_test[names.PREDICTION] = self.predict(df_test)
        cols = [col for col in self.config["features"]] + [
            names.TARGET,
            names.PREDICTION,
        ]
        df_errors = df_test[cols]
        plot_all_scatter_feature_given_label(
            df=df_errors, cols=self.config["features"], output_folder=output_folder
        )

    def saves_stats_errors(self: _Model, df_test: pd.DataFrame) -> None:
        output_folder = os.path.join(self.model_path, "testing", "stats")
        os.makedirs(output_folder, exist_ok=True)
        df_test[names.PREDICTION] = self.predict(df_test)
        cols = [col for col in self.config["features"]] + [
            names.TARGET,
            names.PREDICTION,
        ]
        df_errors = df_test[cols]
        plot_describe_results(df=df_errors, output_folder=output_folder)

    def evaluate_errors(self: _Model, df_test: pd.DataFrame) -> None:
        self.confusion_matrix(df_test=df_test)
        self.save_plots_errors(df_test=df_test)
        self.saves_stats_errors(df_test=df_test)

    def save_config(self: _Model) -> None:
        """
        Save a given metric.

        Args:
            self (_Model): Class instance.
            output_dir (str | None): Output directory.
        """
        with open(os.path.join(self.model_path, "config.json"), "w") as f:
            json.dump(self.config, f, indent=4)

    def save_metrics(self: _Model, phase: str) -> None:
        """
        Save a given metric.

        Args:
            self (_Model): Class instance.
        """
        os.makedirs(os.path.join(self.model_path, phase), exist_ok=True)
        with open(
            os.path.join(self.model_path, phase, "learning_scores.json"), "w"
        ) as f:
            json.dump(self.learning_scores, f, indent=4)
        with open(
            os.path.join(self.model_path, phase, "testing_scores.json"), "w"
        ) as f:
            json.dump(self.testing_scores, f, indent=4)
        with open(os.path.join(self.model_path, phase, "times.json"), "w") as f:
            json.dump(self.times, f, indent=4)

    def save_model(self: _Model) -> None:
        """
        Save model.

        Args:
            self (_Model): Class instance.
        """
        with open(os.path.join(self.model_path, "model.pkl"), "wb") as file:
            pkl.dump(self, file)

    @classmethod
    def load(cls, id_experiment: int, model_type: str) -> _Model:
        """
        Load saved model.
        """
        model_path = os.path.join(
            constants.OUTPUT_FOLDER,
            f"{ id_experiment }_{ model_type }",
        )
        with open(os.path.join(model_path, "model.pkl"), "rb") as file:
            model = pkl.load(file)
        return model

    def learning_pipeline(self: _Model, df_learning: pd.DataFrame) -> None:
        self.get_model_config()
        self.define_model_path()
        self.initialize_scores()
        df_train, df_valid = self.preprocess_learning_set(df_learning=df_learning)
        self.get_columns_to_keep(df_train=df_train)
        start_time = time.time()
        if self.config["fine_tuning"]:
            self.fine_tune(df_train=df_train, df_valid=df_valid)
        if self.config["cross_validation"] is not None:
            self.cross_validate(df_train=df_train)
        else:
            self.learn(df_train=df_train, df_valid=df_valid)
        self.times["learning_time"] = time.time() - start_time
        self.save_config()
        self.save_metrics(phase="learning")
        self.save_model()

    def training_pipeline(
        self: _Model,
        df_learning: pd.DataFrame,
        df_testing: pd.DataFrame,
        is_full_pipeline: bool = False,
    ) -> None:
        if not is_full_pipeline:
            self.get_model_config()
            self.define_model_path()
            self.initialize_scores()
        df_train = self.preprocess_training_set(df_learning=df_learning)
        df_test = self.preprocess_testing_set(df_testing=df_testing)
        if not is_full_pipeline:
            self.get_columns_to_keep(df_train=df_train)
        start_time = time.time()
        self.train(df_train=df_train)
        self.times["training_time"] = time.time() - start_time
        self.get_scores(df_train=df_train, df_test=df_test)
        self.evaluate_errors(df_test=df_test)
        self.save_config()
        self.save_metrics(phase="testing")
        self.save_model()

    def full_pipeline(
        self: _Model, df_learning: pd.DataFrame, df_testing: pd.DataFrame
    ) -> None:
        self.learning_pipeline(df_learning=df_learning)
        self.training_pipeline(
            df_learning=df_learning, df_testing=df_testing, is_full_pipeline=True
        )
