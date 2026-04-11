from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd
from tqdm.auto import tqdm


def _extract_point_forecast(pred_df: pd.DataFrame) -> Tuple[float, Optional[pd.Timestamp], str]:
    """Extract one-step point forecast and metadata from Chronos output."""
    if pred_df.empty:
        raise ValueError("Chronos returned an empty prediction dataframe.")

    row = pred_df.iloc[-1]

    # Common Chronos outputs include quantile columns; prefer median forecast.
    if 0.5 in pred_df.columns:
        pred_col = 0.5
        y_hat = float(row[pred_col])
    elif "0.5" in pred_df.columns:
        pred_col = "0.5"
        y_hat = float(row[pred_col])
    else:
        candidate_cols = ["median", "mean", "prediction", "pred"]
        pred_col = None
        for col in candidate_cols:
            if col in pred_df.columns:
                pred_col = col
                break

        if pred_col is None:
            # Fallback: first numeric non-id/time column. Explicitly avoid "target".
            skip_cols = {"id", "item_id", "timestamp", "ds", "target"}
            numeric_cols = [
                c
                for c in pred_df.columns
                if c not in skip_cols and pd.api.types.is_numeric_dtype(pred_df[c])
            ]
            if not numeric_cols:
                raise ValueError(
                    f"Could not infer prediction column from Chronos output columns: {list(pred_df.columns)}"
                )
            pred_col = numeric_cols[0]

        y_hat = float(row[pred_col])

    pred_ts = None
    if "timestamp" in pred_df.columns:
        pred_ts = pd.to_datetime(row["timestamp"], errors="coerce")
    elif "ds" in pred_df.columns:
        pred_ts = pd.to_datetime(row["ds"], errors="coerce")

    return y_hat, pred_ts, str(pred_col)


def rolling_one_step_predictions(
    pipeline,
    context_df: pd.DataFrame,
    test_df: pd.DataFrame,
    lookback_window: int = 30,
    quantile_levels: List[float] | None = None,
    debug_first_n: int = 0,
    allow_test_cold_start: bool = True,
) -> pd.DataFrame:
    """
    Run rolling one-step predictions per basin with observed-history updates.

    Returns a dataframe with columns:
    - id
    - timestamp
    - actual
    - predicted
    """
    if quantile_levels is None:
        quantile_levels = [0.5]

    context_df = context_df.sort_values(["id", "timestamp"]).reset_index(drop=True)
    test_df = test_df.sort_values(["id", "timestamp"]).reset_index(drop=True)

    hist_by_id = {k: g.copy() for k, g in context_df.groupby("id", sort=False)}
    out_rows = []
    debug_logs = []
    seeded_from_test_basins = 0
    skipped_no_history_basins = 0

    # Iterate per basin and roll forward using actual past observations.
    for basin_id, g_test in tqdm(test_df.groupby("id", sort=False), desc="Rolling inference"):
        history = hist_by_id.get(basin_id, pd.DataFrame(columns=["id", "timestamp", "target"]))
        history = history.sort_values("timestamp").reset_index(drop=True)
        g_test = g_test.sort_values("timestamp").reset_index(drop=True)

        # For location-holdout setups there may be no context rows for test IDs.
        # Seed history from the first lookback_window test points, then evaluate the remainder.
        if history.empty and allow_test_cold_start:
            if len(g_test) <= lookback_window:
                skipped_no_history_basins += 1
                continue
            history = g_test.iloc[:lookback_window][["id", "timestamp", "target"]].copy()
            eval_rows = g_test.iloc[lookback_window:]
            seeded_from_test_basins += 1
        elif history.empty:
            skipped_no_history_basins += 1
            continue
        else:
            eval_rows = g_test

        for step_idx, (_, row) in enumerate(eval_rows.iterrows()):
            context_slice = history.tail(lookback_window)
            pred_df = pipeline.predict_df(
                context_slice,
                prediction_length=1,
                quantile_levels=quantile_levels,
                id_column="id",
                timestamp_column="timestamp",
                target="target",
            )
            y_hat, pred_ts, pred_col = _extract_point_forecast(pred_df)
            actual_ts = pd.to_datetime(row["timestamp"], errors="coerce")
            ts_aligned = bool(
                pred_ts is not None
                and pd.notna(pred_ts)
                and pd.notna(actual_ts)
                and pred_ts == actual_ts
            )

            out_rows.append(
                {
                    "id": basin_id,
                    "timestamp": row["timestamp"],
                    "actual": float(row["target"]),
                    "predicted": y_hat,
                    "pred_timestamp": pred_ts,
                    "prediction_column": pred_col,
                    "timestamp_aligned": ts_aligned,
                }
            )

            if len(debug_logs) < debug_first_n:
                debug_logs.append(
                    {
                        "id": basin_id,
                        "step_idx": int(step_idx),
                        "history_last_timestamp": context_slice["timestamp"].max(),
                        "actual_timestamp": actual_ts,
                        "pred_timestamp": pred_ts,
                        "prediction_column": pred_col,
                        "pred_df_columns": [str(c) for c in pred_df.columns],
                        "predicted_value": y_hat,
                        "actual_value": float(row["target"]),
                    }
                )

            # Append actual observation so next one-step forecast uses true history.
            history = pd.concat(
                [
                    history,
                    pd.DataFrame(
                        [{"id": basin_id, "timestamp": row["timestamp"], "target": row["target"]}]
                    ),
                ],
                ignore_index=True,
            )

    out_df = pd.DataFrame(out_rows)
    if debug_first_n > 0 and debug_logs:
        print("Rolling inference debug samples:")
        for item in debug_logs:
            print(item)
    print(
        "Rolling inference summary:",
        {
            "predictions": len(out_df),
            "seeded_from_test_basins": seeded_from_test_basins,
            "skipped_no_history_basins": skipped_no_history_basins,
        },
    )
    return out_df
