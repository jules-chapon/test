"""Functions for preprocessing"""

import numpy as np
import pandas as pd


def load_data() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "id": range(1000),
            "feature_1": np.random.rand(1000),
            "feature_2": np.random.rand(1000),
        }
    )
    return df


def preprocessing_learning_data(
    df_learning: pd.DataFrame, **kwargs
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Add preprocessing steps
    # And return df_train, df_valid
    return None


def preprocessing_testing_data(df_testing: pd.DataFrame, **kwargs) -> pd.DataFrame:
    # Add preprocessing steps
    # And return df_test
    return None
