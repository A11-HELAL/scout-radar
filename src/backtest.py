"""Backtest - the heart of the project."""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from src.features import ALL_FEATURES


def add_residuals(train, model):
    """A large negative residual means the player is undervalued."""
    out = train.copy()
    out["pred"] = model.predict(out[ALL_FEATURES])
    out["residual"] = out["y_log"] - out["pred"]
    out["decile"] = pd.qcut(out["residual"], 10, labels=False)
    return out


def run_backtest(train_scored, test):
    """Link the 2023 recommendation to the actual 2024 outcome."""
    fwd = test[["player_id", "market_value_in_eur"]].rename(
        columns={"market_value_in_eur": "value_t1"}
    )
    bt = train_scored.merge(fwd, on="player_id", how="inner")

    bt["growth_log"] = np.log1p(bt["value_t1"]) - np.log1p(bt["market_value_in_eur"])
    bt["growth_pct"] = bt["value_t1"] / bt["market_value_in_eur"] - 1

    by_decile = bt.groupby("decile")["growth_log"].agg(["median", "mean", "count"])

    under = bt.loc[bt["decile"] == 0, "growth_log"]
    over = bt.loc[bt["decile"] == 9, "growth_log"]
    stat, pval = mannwhitneyu(under, over, alternative="greater")

    top20 = bt.nsmallest(20, "residual")
    precision20 = (top20["growth_pct"] > 0.50).mean()
    baseline = (bt["growth_pct"] > 0.50).mean()

    bt["age_band"] = pd.cut(bt["age"], [15, 21, 24, 27, 30, 45])
    by_age = bt.pivot_table(
        index="age_band",
        columns="decile",
        values="growth_log",
        aggfunc="median",
        observed=False,
    )

    summary = {
        "p_value": round(pval, 5),
        "precision_at_20": round(precision20, 3),
        "baseline_rate": round(baseline, 3),
        "lift": round(precision20 / baseline, 2) if baseline > 0 else None,
    }
    return bt, by_decile, by_age, summary
