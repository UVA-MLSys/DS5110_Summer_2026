from __future__ import annotations

import numpy as np


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        raise ValueError("y_true is empty; cannot compute RMSE.")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
