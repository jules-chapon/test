"""Feature selection functions"""

from typing import List
import numpy as np
import pandas as pd
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from logger import logging

from src.configs import constants


def selecting_features_with_boruta(
    df: pd.DataFrame, features: List[str] | None, target: str | None
) -> List[str]:
    """
    Select features using Boruta.

    Args:
        df (pd.DataFrame): Dataframe with features and target.
        features (List[str]): Features to select from.
        target (str): Target column.

    Returns:
        List[str]: Selected features by Boruta.
    """
    # Random forest classifier
    rf_classifier = RandomForestClassifier(
        n_estimators=constants.NB_RF_CLASSIFIERS_BORUTA,
        random_state=constants.RANDOM_SEED,
    )
    # Boruta feature selection method
    boruta_selector = BorutaPy(
        rf_classifier,
        n_estimators=constants.NB_BORUTA_ESTIMATORS,
        random_state=constants.RANDOM_SEED,
    )
    boruta_selector.fit(df[features].values, df[target].values)
    selected_features = df[features].columns[boruta_selector.support_].tolist()
    logging.info(f"Selected features by Boruta : { selected_features }")
    return selected_features


def selecting_features_with_random_columns(
    df: pd.DataFrame, features: List[str] | None, target: str | None
) -> List[str]:
    """
    Select features that have less importance than random ones.

    Args:
        df (pd.DataFrame): Dataframe with features and target.
        features (List[str]): List of features.
        target (str): Target column.

    Returns:
        List[str]: Selected features with random columns.
    """
    # Create 5 random columns
    for i in range(constants.NB_RANDOM_COLUMNS):
        df[f"random_{i}"] = np.random.uniform(
            low=constants.LOW_VALUE_FEATURE,
            high=constants.HIGH_VALUE_FEATURE,
            size=df.shape[0],
        )
    # Combine original features with random features
    all_features = features + [f"random_{i}" for i in range(5)]
    # Initialize and fit the RandomForestClassifier
    rf_classifier = RandomForestClassifier(
        n_estimators=constants.NB_RF_CLASSIFIERS_RANDOM_COLUMNS,
        random_state=constants.RANDOM_SEED,
    )
    rf_classifier.fit(df[all_features].values, df[target].values)
    # Get feature importances
    feature_importances = rf_classifier.feature_importances_
    feature_importances_df = pd.DataFrame(
        {"feature": all_features, "importance": feature_importances}
    )
    # Select features with importance greater than the maximum random importance
    max_random_importance = feature_importances_df[
        feature_importances_df["feature"].str.contains("random")
    ]["importance"].max()
    selected_features = feature_importances_df[
        feature_importances_df["importance"] > max_random_importance
    ]["feature"].tolist()
    selected_features = [
        feature for feature in selected_features if not feature.startswith("random")
    ]
    logging.info(f"Selected features with random columns : { selected_features }")
    return selected_features
