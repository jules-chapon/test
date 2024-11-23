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
        """
        Initialize class instance.

        Args:
            id_experiment (int | None): ID of the experiment.
        """
        self.id_experiment = id_experiment
        self.learning_scores = triple_nested_defaultdict()
        self.testing_scores = triple_nested_defaultdict()
        self.times = double_nested_defaultdict()

    def get_model_config(self: _Model) -> None:
        """
        Get the config of the experiment.

        Args:
            self (_Model): Class instance.
        """
        if self.id_experiment in ml_config.EXPERIMENTS_CONFIGS:
            self.config = ml_config.EXPERIMENTS_CONFIGS[self.id_experiment]
        else:
            logging.info("No corresponding experiment")
        if isinstance(self.config[names.TRAINING_PARAMS]["metrics"], str):
            self.config[names.TRAINING_PARAMS]["metrics"] = [
                self.config[names.TRAINING_PARAMS]["metrics"]
            ]
        if isinstance(self.config[names.FEATURES], str):
            self.config[names.FEATURES] = [self.config[names.FEATURES]]

    def define_model_path(self: _Model) -> None:
        """
        Define the folder to save the model.

        Args:
            self (_Model): Class instance.
        """
        self.model_path = os.path.join(
            constants.OUTPUT_FOLDER,
            f"{ self.id_experiment }_{ self.config[names.MODEL_TYPE] }",
        )
        os.makedirs(self.model_path, exist_ok=True)

    def initialize_scores(self: _Model) -> None:
        """
        Initialize all the scores of the model.

        Args:
            self (_Model): Class instance.
        """
        for dataset in ["train", "valid"]:
            for metric in self.config[names.TRAINING_PARAMS]["metrics"]:
                for score in ["scores", "best_scores"]:
                    self.learning_scores[dataset][metric][score] = []
                    if dataset == "train":
                        self.testing_scores[dataset][metric][score] = []
        self.testing_scores["train"]["metrics"] = {}
        self.testing_scores["test"]["metrics"] = {}

    def preprocess_learning_set(
        self: _Model, df_learning: pd.DataFrame
    ) -> tuple[pd.DataFrame]:
        """
        Preprocess the learning set.

        Args:
            self (_Model): Class instance.
            df_learning (pd.DataFrame): Learning set.

        Returns:
            tuple[pd.DataFrame]: Train and valid sets (for learning).
        """
        # If no split bewteen train and valid (cross-validation)
        if self.config[names.TRAIN_RATIO] == 1:
            df_train = preprocessing_learning_data(
                df_learning=df_learning, train_ratio=self.config[names.TRAIN_RATIO]
            )
            df_valid = None
        # If split between train and valid
        else:
            df_train, df_valid = preprocessing_learning_data(
                df_learning=df_learning, train_ratio=self.config[names.TRAIN_RATIO]
            )
        return df_train, df_valid

    def preprocess_training_set(
        self: _Model, df_learning: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Preprocess the training set.
        The training is made on the whole learning set.

        Args:
            self (_Model): Class instance.
            df_learning (pd.DataFrame): Learning set.

        Returns:
            pd.DataFrame: Training set
        """
        df_train = preprocessing_learning_data(df_learning=df_learning, train_ratio=1)
        return df_train

    def preprocess_testing_set(self: _Model, df_testing: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the testing set.

        Args:
            self (_Model): Class instance.
            df_testing (pd.DataFrame): Testing set.

        Returns:
            pd.DataFrame: Prepreocessed testing set.
        """
        df_test = preprocessing_testing_data(df_testing)
        return df_test

    def drop_id_columns(self: _Model, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop ID columns (useless for training and prediction).

        Args:
            self (_Model): Class instance.
            df (pd.DataFrame): DataFrame (training set).

        Returns:
            pd.DataFrame: DataFrame without ID cols.
        """
        return df.drop(columns=self.config[names.COLS_ID])

    def select_features(self: _Model, df_train: pd.DataFrame) -> None:
        if self.config[names.FEATURES] is None:
            self.config[names.FEATURES] = df_train.columns.tolist()
        if self.config[names.FEATURE_SELECTION] is not None:
            if "correlation" in self.config[names.FEATURE_SELECTION]:
                self.config[names.FEATURES] = selecting_features_with_correlations(
                    df=df_train,
                    features=self.config[names.FEATURES],
                    target=self.config[names.TARGET],
                )
            if "random_columns" in self.config[names.FEATURE_SELECTION]:
                self.config[names.FEATURES] = selecting_features_with_random_columns(
                    df=df_train,
                    features=self.config[names.FEATURES],
                    target=self.config[names.TARGET],
                )
            if "boruta" in self.config[names.FEATURE_SELECTION]:
                self.config[names.FEATURES] = selecting_features_with_boruta(
                    df=df_train,
                    features=self.config[names.FEATURES],
                    target=self.config[names.TARGET],
                )
        return None

    def get_columns_to_keep(
        self: _Model,
        df_train: pd.DataFrame,
    ) -> None:
        """
        Get the list of features to keep.

        Args:
            self (_Model): Class instance.
            df_train (pd.DataFrame): Training set.
        """
        df_train_without_ids = self.drop_id_columns(df=df_train)
        self.select_features(df_train=df_train_without_ids)
        self.config[names.FEATURES] = [
            col
            for col in self.config[names.FEATURES]
            if col not in self.config[names.TARGET]
        ]

    @abc.abstractmethod
    def learn(
        self: _Model,
        df_train: pd.DataFrame,
        df_valid: pd.DataFrame,
    ) -> None:
        """
        Learning function.
        Get the loss of the model on both train and valid sets.
        Used to find the best hyperparameters of the model.

        Args:
            self (_Model): Class instance.
            df_train (pd.DataFrame): Train set.
            df_valid (pd.DataFrame): Valid set.
        """

    def cross_validate(self: _Model, df_train: pd.DataFrame):
        """
        Learning with cross-validation.
        Get the average loss of the model over all iterations.
        Used to find the best hyperparameters of the model.

        Args:
            self (_Model): Class instance.
            df_train (pd.DataFrame): Train set.
        """
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
        for metric in self.config[names.TRAINING_PARAMS]["metrics"]:
            scores[metric] = self.learning_scores["valid"][metric]["best_scores"]
        for metric, scores_list in scores.items():
            self.learning_scores["cross_validation"][metric]["average_score"] = np.mean(
                scores_list
            )

    @abc.abstractmethod
    def fine_tune_objective(
        self: _Model, df_train: pd.DataFrame, df_valid: pd.DataFrame
    ) -> None:
        """
        Define the objective function for the fine-tuning of the hyperparameters.

        Args:
            self (_Model): Class instance.
            df_train (pd.DataFrame): Train set.
            df_valid (pd.DataFrame): Valid set.
        """

    @abc.abstractmethod
    def fine_tune(self: _Model, df_train: pd.DataFrame, df_valid: pd.DataFrame) -> None:
        """
        Fine-tune the hyperparameters of the model.

        Args:
            self (_Model): Class instance.
            df_train (pd.DataFrame): Train set.
            df_valid (pd.DataFrame): Valid set.
        """

    @abc.abstractmethod
    def train(
        self: _Model,
        df_train: pd.DataFrame,
    ) -> None:
        """
        Train the model.

        Args:
            self (_Model): Class instance.
            df_train (pd.DataFrame): Train set (whole learning set).
        """

    @abc.abstractmethod
    def predict(self: _Model, df_to_predict: pd.DataFrame) -> np.ndarray:
        """
        Predict outputs for a given dataset.

        Args:
            self (_Model): Class instance.
            df_to_predict (pd.DataFrame): DataFrame to predict.

        Returns:
            np.ndarray: Predictions.
        """

    def score(self: _Model, df_to_evaluate: pd.DataFrame) -> dict:
        """
        Compute metrics for a given dataset.

        Args:
            self (_Model): Class instance.
            df_to_evaluate (pd.DataFrame): DataFrame to evaluate.

        Returns:
            dict: Dictionary with metric names as keys and values as values.
        """
        label_pred = self.predict(df_to_evaluate)
        label_true = df_to_evaluate[self.config[names.TARGET]]
        metrics = compute_evaluation_metrics(label_true, label_pred)
        return metrics

    def get_scores(
        self: _Model,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
    ) -> None:
        """
        Get metrics on train and test sets.

        Args:
            self (_Model): Class instance.
            df_train (pd.DataFrame): Train set.
            df_test (pd.DataFrame): Test set.
        """
        self.testing_scores["train"]["metrics"] = self.score(df_to_evaluate=df_train)
        self.testing_scores["test"]["metrics"] = self.score(df_to_evaluate=df_test)

    def confusion_matrix(self: _Model, df_test: pd.DataFrame) -> None:
        """
        Save the confusion matrix on the test set.

        Args:
            self (_Model): Class instance.
            df_test (pd.DataFrame): Test set.
        """
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
        """
        Save the scatter plots to analyze errors on the test set.

        Args:
            self (_Model): Class instance.
            df_test (pd.DataFrame): Test set.
        """
        output_folder = os.path.join(self.model_path, "testing", "plots")
        os.makedirs(output_folder, exist_ok=True)
        df_test[names.PREDICTION] = self.predict(df_test)
        cols = [col for col in self.config[names.FEATURES]] + [
            names.TARGET,
            names.PREDICTION,
        ]
        df_errors = df_test[cols]
        plot_all_scatter_feature_given_label(
            df=df_errors, cols=self.config[names.FEATURES], output_folder=output_folder
        )

    def saves_stats_errors(self: _Model, df_test: pd.DataFrame) -> None:
        """
        Save the tables that describe the errors on the test set.

        Args:
            self (_Model): Class instance.
            df_test (pd.DataFrame): Test set.
        """
        output_folder = os.path.join(self.model_path, "testing", "stats")
        os.makedirs(output_folder, exist_ok=True)
        df_test[names.PREDICTION] = self.predict(df_test)
        cols = [col for col in self.config[names.FEATURES]] + [
            names.TARGET,
            names.PREDICTION,
        ]
        df_errors = df_test[cols]
        plot_describe_results(df=df_errors, output_folder=output_folder)

    def evaluate_errors(self: _Model, df_test: pd.DataFrame) -> None:
        """
        Save all metrics on errors.

        Args:
            self (_Model): Class instance.
            df_test (pd.DataFrame): Test set.
        """
        self.confusion_matrix(df_test=df_test)
        self.save_plots_errors(df_test=df_test)
        self.saves_stats_errors(df_test=df_test)

    def save_config(self: _Model) -> None:
        """
        Save the config of the model.

        Args:
            self (_Model): Class instance.
            output_dir (str | None): Output directory.
        """
        with open(os.path.join(self.model_path, "config.json"), "w") as f:
            json.dump(self.config, f, indent=4)

    def save_metrics(self: _Model, phase: str) -> None:
        """
        Save the metrics.

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
        Save the model.

        Args:
            self (_Model): Class instance.
        """
        with open(os.path.join(self.model_path, "model.pkl"), "wb") as file:
            pkl.dump(self, file)

    @classmethod
    def load(cls, id_experiment: int, model_type: str) -> _Model:
        """
        Load a saved model.

        Args:
            id_experiment (int): ID of the experiment.
            model_type (str): Model type.

        Returns:
            _Model: Loaded modeL.
        """
        model_path = os.path.join(
            constants.OUTPUT_FOLDER,
            f"{ id_experiment }_{ model_type }",
        )
        with open(os.path.join(model_path, "model.pkl"), "rb") as file:
            model = pkl.load(file)
        return model

    def learning_pipeline(self: _Model, df_learning: pd.DataFrame) -> None:
        """
        Do the whole learning pipeline.

        Args:
            self (_Model): Class instance.
            df_learning (pd.DataFrame): Learning set.
        """
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

    def testing_pipeline(
        self: _Model,
        df_learning: pd.DataFrame,
        df_testing: pd.DataFrame,
        is_full_pipeline: bool = False,
    ) -> None:
        """
        Do the whole training pipeline.

        Args:
            self (_Model): Class instance.
            df_learning (pd.DataFrame): Learning set.
            df_testing (pd.DataFrame): Testing set.
            is_full_pipeline (bool, optional): Whether it is included in the whole pipeline or not.
                Defaults to False.
        """
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
        """
        Do the full pipeline (learning, training and testing).

        Args:
            self (_Model): Class instance.
            df_learning (pd.DataFrame): Learning set.
            df_testing (pd.DataFrame): Testing set.
        """
        self.learning_pipeline(df_learning=df_learning)
        self.testing_pipeline(
            df_learning=df_learning, df_testing=df_testing, is_full_pipeline=True
        )
