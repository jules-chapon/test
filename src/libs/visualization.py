"""Visualization functions"""

import ipywidgets as widgets
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from ipywidgets import fixed

from src.configs import constants, ml_config, names


def plot_scatter_feature(
    df: pd.DataFrame,
    feature_name_1: str,
    feature_name_2: str,
    target: str = names.TARGET,
    output_path: str | None = None,
) -> None:
    if output_path is not None:
        plt.ioff()
    unique_targets = df[target].unique()
    palette = sns.color_palette("bright", len(unique_targets))
    target_colors = {target: color for target, color in zip(unique_targets, palette)}
    plt.figure(figsize=(10, 6))
    for target_value, color in target_colors.items():
        subset = df[df[target] == target_value]
        plt.scatter(
            subset[feature_name_1],
            subset[feature_name_2],
            label=str(target_value),
            color=color,
            alpha=0.9,
        )
    plt.xlabel(feature_name_1)
    plt.ylabel(feature_name_2)
    plt.legend(title=target)
    plt.title(f"Scatter plot between {feature_name_1} and {feature_name_2}")
    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()


def plot_slider_scatter_feature(
    df: pd.DataFrame, subset_features: list | None = None
) -> None:
    if subset_features is not None:
        list_features = subset_features
    else:
        list_features = df.columns
    feature_selection_1 = widgets.Dropdown(
        options=list_features,
        description="Feature 1",
        disabled=False,
    )
    feature_selection_2 = widgets.Dropdown(
        options=list_features,
        description="Feature 2",
        disabled=False,
    )
    widgets.interact(
        plot_scatter_feature,
        df=fixed(df),
        feature_name_1=feature_selection_1,
        feature_name_2=feature_selection_2,
    )


def plot_boxplot(
    df: pd.DataFrame,
    feature_name: str,
    target: str = names.TARGET,
    output_path: str | None = None,
) -> None:
    if output_path is not None:
        plt.ioff()
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df,
        x=target,
        y=feature_name,
        palette="viridis",
        hue=target,
        legend=False,
    )
    plt.title(f"Distribution of { feature_name } given { target }")
    plt.xlabel(target)
    plt.ylabel(feature_name)
    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()


def plot_slider_boxplot(df: pd.DataFrame, subset_features: list | None = None) -> None:
    if subset_features is not None:
        list_features = subset_features
    else:
        list_features = df.columns
    feature_selection = widgets.Dropdown(
        options=list_features,
        description="Feature",
        disabled=False,
    )
    widgets.interact(
        plot_boxplot,
        df=fixed(df),
        feature_name=feature_selection,
    )


def plot_correlation_with_target(
    df: pd.DataFrame,
    top_k: int,
    target: str = names.TARGET,
    output_path: str | None = None,
) -> None:
    if output_path is not None:
        plt.ioff()
    correlations = df.corr()[target].drop(target).abs().sort_values(ascending=False)
    top_k_features = correlations.head(top_k)
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=top_k_features.index,
        y=top_k_features.values,
        palette="coolwarm",
        hue=top_k_features.index,
        legend=False,
    )
    plt.title(f"Top {top_k} features most correlated with {target}")
    plt.xlabel("Features")
    plt.ylabel(f"Correlation with {target} (in absolute value)")
    plt.xticks(rotation=45)
    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()


def plot_correlation_matrix(
    df: pd.DataFrame,
    subset_features: list | None = None,
    target: str = names.TARGET,
    output_path: str | None = None,
) -> None:
    if output_path is not None:
        plt.ioff()
    if subset_features is not None:
        list_features = subset_features
    else:
        list_features = [col for col in df.columns if col != target]
    correlation_matrix = df[list_features].corr().abs()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=0, vmax=1)
    plt.title("Correlation matrix (in absolute values) between features")
    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()


def plot_training_curves(
    dict_metrics: dict[str, list[float]], output_path: str | None = None
) -> None:
    if output_path is not None:
        plt.ioff()
    plt.figure(figsize=(10, 6))
    for metric_name, metric_values in dict_metrics.items():
        plt.plot(metric_values, label=metric_name)
    plt.xlabel("Epoch")
    plt.ylabel("Metric value")
    plt.legend()
    plt.title("Training curves")
    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()


def plot_df_description_as_image(
    df_summary: pd.DataFrame, output_path: str | None = None
) -> None:
    if output_path is not None:
        plt.ioff()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(
        cellText=df_summary.values,
        colLabels=df_summary.columns,
        rowLabels=df_summary.index,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    if output_path is not None:
        plt.savefig(output_path)
