"""Functions for preprocessing"""

import numpy as np
import pandas as pd


def load_data(is_train: bool) -> pd.DataFrame:
    if is_train:
        filename = "train"
    else:
        filename = "test"
    df = pd.read_csv(f"data/input/{filename}.csv")
    return df


def preprocessing_learning_data(
    df_learning: pd.DataFrame, train_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Add preprocessing steps
    # And return df_train, df_valid
    df_train = df_learning.sample(frac=train_ratio)
    df_valid = df_learning.drop(df_train.index)
    return df_train, df_valid


def preprocessing_testing_data(df_testing: pd.DataFrame, **kwargs) -> pd.DataFrame:
    # Add preprocessing steps
    # And return df_test
    return df_testing
