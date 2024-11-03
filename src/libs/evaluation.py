"""Evaluation functions"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

from src.configs import constants, ml_config, names

from src.libs.visualization import plot_scatter_feature, plot_df_description_as_image


def compute_evaluation_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return {"accuracy": accuracy, "recall": recall}


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, output_path: str | None = None
) -> None:
    if output_path is not None:
        plt.ioff()
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=np.unique(y_true),
        yticklabels=np.unique(y_true),
    )
    plt.ylabel("True labels")
    plt.xlabel("Predicted labels")
    plt.title("Confusion matrix")
    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()


def plot_scatter_feature_given_label(
    df: pd.DataFrame,
    label: int | str,
    feature_name_1: str,
    feature_name_2: str,
    output_path: str | None = None,
) -> None:
    df_label = df[df[names.TARGET] == label]
    plot_scatter_feature(
        df=df_label,
        feature_name_1=feature_name_1,
        feature_name_2=feature_name_2,
        target=names.PREDICTION,
        output_path=output_path,
    )


def plot_all_scatter_feature_given_label(
    df: pd.DataFrame, cols: list, output_folder: str
) -> None:
    for label in set(df[names.TARGET]):
        for feature_name_1 in cols:
            for feature_name_2 in cols:
                plot_scatter_feature_given_label(
                    df=df,
                    label=label,
                    feature_name_1=feature_name_1,
                    feature_name_2=feature_name_2,
                    output_path=os.path.join(
                        output_folder,
                        f"true_{label}_{feature_name_1}_{feature_name_2}.png",
                    ),
                )


def plot_describe_results(df: pd.DataFrame, output_folder: str | None = None) -> None:
    for target in set(df[names.TARGET]):
        for prediction in set(df[names.PREDICTION]):
            df_summary = df[
                (df[names.TARGET] == target) & (df[names.PREDICTION] == prediction)
            ].describe()
            output_path = os.path.join(
                output_folder, f"true_{target}_pred_{prediction}.png"
            )
            plot_df_description_as_image(df_summary=df_summary, output_path=output_path)
