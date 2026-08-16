"""Metrics.

We train on log(value) because market values span 25k to 200m, but a scout
thinks in euros and in ranking order - so we report all three languages.
"""

import numpy as np
import pandas as pd


def evaluate(y_true_log, y_pred_log, label="model"):
    """Return one row of metrics for a set of predictions."""
    y_true_log = np.asarray(y_true_log, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)
    error = y_pred_log - y_true_log

    true_eur = np.expm1(y_true_log)
    pred_eur = np.expm1(y_pred_log)
    abs_pct_error = np.abs(pred_eur - true_eur) / np.maximum(true_eur, 1.0)

    ss_res = float((error ** 2).sum())
    ss_tot = float(((y_true_log - y_true_log.mean()) ** 2).sum())

    spearman = float("nan")
    if y_true_log.size >= 2:
        spearman = float(
            pd.Series(y_pred_log).corr(pd.Series(y_true_log), method="spearman")
        )

    return {
        "model": label,
        # error in log space - what the model actually optimises
        "MAE_log": float(np.abs(error).mean()),
        "RMSE_log": float(np.sqrt((error ** 2).mean())),
        "R2_log": 1.0 - ss_res / ss_tot if ss_tot else float("nan"),
        # The error a human understands, ALREADY IN PERCENT: 38.0 means
        # "typically off by 38%". This used to be returned as the fraction 0.38
        # under the name MedAPE_eur, which is how "0.38%" ends up on a slide.
        "MedAPE_pct": float(np.median(abs_pct_error) * 100),
        # the metric that actually matters: do we get the ORDER right?
        "Spearman": spearman,
        "n": int(y_true_log.size),
    }
