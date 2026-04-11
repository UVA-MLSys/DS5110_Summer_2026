from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_actual_vs_predicted(
    pred_df: pd.DataFrame,
    output_path: str,
    max_points: int = 1000,
) -> None:
    """Save Actual vs Predicted scatter plot."""
    if pred_df.empty:
        raise ValueError("Prediction dataframe is empty.")

    plot_df = pred_df.copy()
    if len(plot_df) > max_points:
        plot_df = plot_df.sample(max_points, random_state=42)

    plt.figure(figsize=(7, 7))
    plt.scatter(plot_df["actual"], plot_df["predicted"], alpha=0.45, s=12)
    min_v = min(plot_df["actual"].min(), plot_df["predicted"].min())
    max_v = max(plot_df["actual"].max(), plot_df["predicted"].max())
    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted (Test Set)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
