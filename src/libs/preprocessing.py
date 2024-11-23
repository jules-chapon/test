"""Functions for preprocessing"""

import pandas as pd
from datasets import load_dataset

from src.configs.constants import HF_DATASET_FOLDER, HF_DATASET_FILES


def load_data_from_hf(is_train: bool) -> pd.DataFrame:
    """
    Load a dataset from Hugging Face.

    Args:
        is_train (bool): Whether the dataset is for training or testing.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    if is_train:
        filename = "train"
    else:
        filename = "test"
    data = load_dataset(HF_DATASET_FOLDER, data_files=HF_DATASET_FILES)
    df = data[filename].to_pandas()
    return df


def load_data_from_local(is_train: bool) -> pd.DataFrame:
    """
    Load a dataset from own computer.

    Args:
        is_train (bool): Whether the dataset is for training or testing.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    if is_train:
        filename = "train"
    else:
        filename = "test"
    df = pd.read_csv(f"data/input/{filename}.csv")
    return df


def load_data(is_train: bool, is_local: bool) -> pd.DataFrame:
    """
    Load a dataset.

    Args:
        is_train (bool): Whether the dataset is for training or testing.
        is_local (bool): Whether the dataset is to load from local or HF.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    if is_local:
        return load_data_from_local(is_train)
    else:
        return load_data_from_hf(is_train)


def preprocessing_learning_data(
    df_learning: pd.DataFrame, train_ratio: float
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess the learning data.
    Split it between training and validation if needed.

    Args:
        df_learning (pd.DataFrame): Learning dataset.
        train_ratio (float, optional) :
            Ratio of training data (the remaining will be used for validation).
            If train_ratio == 1, there won't be any validation set.

    Returns:
        pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]: f train_ratio == 1, training dataset.
            Otherwise, a tuple containing training and validation datasets.
    """
    # Add preprocessing steps
    # And return df_train or df_train, df_valid
    if train_ratio == 1:
        return df_learning
    else:
        df_train = df_learning.sample(frac=train_ratio)
        df_valid = df_learning.drop(df_train.index)
        return df_train, df_valid


def preprocessing_testing_data(df_testing: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the testing data.

    Args:
        df_testing (pd.DataFrame): Testing dataset.

    Returns:
        pd.DataFrame: Preprocessed testing dataset.
    """
    # Add preprocessing steps
    # And return df_test
    return df_testing
