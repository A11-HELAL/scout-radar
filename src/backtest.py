"""The backtest - the part that decides whether this project is real.

The question is not "can the model predict market value". It is: of the players
the model called cheap in June 2023, did the market agree by June 2024?
"""

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from src import config


def add_residuals(df, model, features=None):
    """residual = what the market pays  -  what the model says he is worth.

    A very negative residual means the market is asleep, so it is a candidate.
    Note this is a residual, not truth: residual = real mispricing + model
    error, and we can never fully separate the two.
    """
    features = list(features or config.FEATURES)
    out = df.copy()
    out["pred_log"] = model.predict(out[features])
    out["residual"] = out[config.TARGET] - out["pred_log"]
    out["pred_value_eur"] = np.expm1(out["pred_log"])
    # duplicates="drop" so a small filtered slice cannot crash qcut
    out["decile"] = pd.qcut(out["residual"], 10, labels=False, duplicates="drop")
    return out


def attach_future_value(scored, valuations, horizon_days=None):
    """Look the real future value up in the RAW valuations table.

    Why not just join the 2024 feature table? Because that table only contains
    players who still had 900+ minutes and a fresh valuation a year later - in
    other words the players who did not get injured, relegated or dropped. That
    is survivorship bias, and it silently flatters the result. Here everyone is
    kept and the ones we genuinely cannot follow are flagged as censored.
    """
    horizon_days = horizon_days or config.HORIZON_DAYS
    out = scored.copy()
    snapshot = pd.Timestamp(out["snapshot"].iloc[0])
    target_date = snapshot + pd.Timedelta(days=horizon_days)

    future = valuations[valuations["date"] <= target_date].sort_values("date")
    future = future.groupby("player_id").tail(1)[
        ["player_id", "date", "market_value_in_eur"]
    ].rename(columns={"date": "future_date", "market_value_in_eur": "future_value_eur"})

    out = out.merge(future, on="player_id", how="left")

    missing = out["future_value_eur"].isna()
    # a valuation that stopped being refreshed is not a real "one year later"
    stale = (target_date - out["future_date"]).dt.days > config.MAX_VALUATION_AGE_DAYS
    # and it must genuinely be AFTER the snapshot, not the same old number
    too_early = out["future_date"] <= snapshot

    out["censored"] = missing | stale.fillna(True) | too_early.fillna(True)
    out.loc[out["censored"], "future_value_eur"] = np.nan

    out["growth_log"] = (
        np.log1p(out["future_value_eur"]) - np.log1p(out["market_value_in_eur"])
    )
    out["growth_pct"] = out["future_value_eur"] / out["market_value_in_eur"] - 1
    return out


def make_shortlist(scored, size=None, max_age=None, min_value=None):
    """The one definition of "our recommendation".

    Both the backtest and the exported JSON call this function, so the number
    we quote in the defence is measured on the exact list the app shows.

    Ranking on the raw residual alone hands you 34-year-olds worth 200k,
    because on a log scale a cheap player is easy to be very wrong about. The
    age cap and the value floor are what turn a residual into a shortlist.
    """
    size = size or config.SHORTLIST_SIZE
    max_age = max_age or config.SHORTLIST_MAX_AGE
    min_value = min_value or config.SHORTLIST_MIN_VALUE
    return eligible_pool(scored, max_age, min_value).nsmallest(size, "residual")


def eligible_pool(scored, max_age=None, min_value=None):
    """Every player who COULD have made the shortlist - the fair comparison."""
    max_age = max_age or config.SHORTLIST_MAX_AGE
    min_value = min_value or config.SHORTLIST_MIN_VALUE
    return scored[
        (scored["age"] <= max_age) & (scored["market_value_in_eur"] >= min_value)
    ]


def bootstrap_diff_ci(a, b, n_boot=2000, seed=None):
    """95% interval for median(a) - median(b).

    Twenty players is a tiny sample and a tiny sample can look impressive by
    luck. This shows how wide the honest range really is.
    """
    rng = np.random.default_rng(seed or config.RANDOM_STATE)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if a.size < 2 or b.size < 2:
        return (float("nan"), float("nan"))
    diffs = [
        np.median(rng.choice(a, a.size)) - np.median(rng.choice(b, b.size))
        for _ in range(n_boot)
    ]
    low, high = np.percentile(diffs, [2.5, 97.5])
    return round(float(low), 3), round(float(high), 3)


def run_backtest(scored_with_future):
    """Returns (backtested_rows, table_by_decile, shortlist, summary)."""
    followed = scored_with_future[~scored_with_future["censored"]].copy()
    coverage = len(followed) / max(len(scored_with_future), 1)

    by_decile = followed.groupby("decile").agg(
        n=("growth_log", "size"),
        median_growth_log=("growth_log", "median"),
        mean_growth_log=("growth_log", "mean"),
        median_value_eur=("market_value_in_eur", "median"),
    )
    by_decile["hit_rate"] = followed.groupby("decile")["growth_pct"].apply(
        lambda s: (s > config.GROWTH_THRESHOLD).mean()
    )

    cheapest, dearest = by_decile.index.min(), by_decile.index.max()
    undervalued = followed.loc[followed["decile"] == cheapest, "growth_log"].dropna()
    overvalued = followed.loc[followed["decile"] == dearest, "growth_log"].dropna()
    _, p_value = mannwhitneyu(undervalued, overvalued, alternative="greater")

    # precision@k and its baseline are measured on the SAME eligible pool.
    # Comparing a 26-and-under shortlist against the whole population would
    # inflate the lift with nothing but the age filter.
    shortlist = make_shortlist(followed)
    pool = eligible_pool(followed)
    precision = (shortlist["growth_pct"] > config.GROWTH_THRESHOLD).mean()
    baseline = (pool["growth_pct"] > config.GROWTH_THRESHOLD).mean()

    summary = {
        "n_recommended": int(len(scored_with_future)),
        "n_followed": int(len(followed)),
        "coverage": round(coverage, 3),
        "median_growth_undervalued": round(float(undervalued.median()), 3),
        "median_growth_overvalued": round(float(overvalued.median()), 3),
        "diff_95pct_ci": bootstrap_diff_ci(undervalued, overvalued),
        "p_value": round(float(p_value), 5),
        "eligible_pool_size": int(len(pool)),
        "precision_at_k": round(float(precision), 3),
        "baseline_rate": round(float(baseline), 3),
        "lift": round(float(precision / baseline), 2) if baseline > 0 else None,
    }
    return followed, by_decile, shortlist, summary
