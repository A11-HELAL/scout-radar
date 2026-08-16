"""The backtest - the part that decides whether this project is real.

The question is not "can the model predict market value". It is: of the players
the model called cheap in June 2023, did the market agree by June 2024?
"""

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from src import config
from src.features import safe_merge


def _round(value, digits=3):
    """round(), except NaN becomes None.

    json.dumps writes NaN as the bare token NaN, which is not valid JSON and
    makes the website's JSON.parse throw. None becomes null, which is fine.
    """
    value = float(value)
    return round(value, digits) if np.isfinite(value) else None


def _p_text(p_value, floor=1e-5):
    """A p-value you can safely put on a slide.

    round(1e-17, 5) is 0.0, and "p = 0.0" is wrong - no test ever returns
    exactly zero. Below the floor we say "smaller than" instead.
    """
    if not np.isfinite(p_value):
        return "not enough data"
    if p_value < floor:
        return f"p < {floor:g}"
    return f"p = {p_value:.5f}"


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

    And "one year later" is a WINDOW, not "the last valuation before the target
    date". Taking the last one lets a price refreshed ten days after the
    snapshot count as a one-year outcome, which turns the whole backtest into a
    measurement of almost nothing. We keep only valuations inside
    [snapshot + HORIZON_MIN_DAYS, snapshot + HORIZON_MAX_DAYS] and, when a
    player has several in there, the one closest to the target date.
    """
    horizon_days = horizon_days or config.HORIZON_DAYS
    out = scored.copy()
    snapshot = pd.Timestamp(out["snapshot"].iloc[0])
    target_date = snapshot + pd.Timedelta(days=horizon_days)
    earliest = snapshot + pd.Timedelta(days=config.HORIZON_MIN_DAYS)
    latest = snapshot + pd.Timedelta(days=config.HORIZON_MAX_DAYS)

    window = valuations[
        (valuations["date"] >= earliest)
        & (valuations["date"] <= latest)
        & (valuations["market_value_in_eur"] > 0)
    ].copy()
    window["days_off_target"] = (window["date"] - target_date).abs().dt.days
    future = (
        window.sort_values(["player_id", "days_off_target"])
        .groupby("player_id")
        .head(1)[["player_id", "date", "market_value_in_eur", "days_off_target"]]
        .rename(
            columns={
                "date": "future_date",
                "market_value_in_eur": "future_value_eur",
            }
        )
    )

    out = safe_merge(out, future, on="player_id", label="scored + future value")

    # censored = we could not observe this player one year later. He stays in
    # the table so coverage is honest, he just has no growth number.
    out["censored"] = out["future_value_eur"].isna()

    out["growth_log"] = (
        np.log1p(out["future_value_eur"]) - np.log1p(out["market_value_in_eur"])
    )
    out["growth_pct"] = out["future_value_eur"] / out["market_value_in_eur"] - 1
    return out


def eligible_pool(scored, max_age=None, min_value=None):
    """Every player who COULD have made the shortlist - the fair comparison."""
    max_age = max_age or config.SHORTLIST_MAX_AGE
    min_value = min_value or config.SHORTLIST_MIN_VALUE
    return scored[
        (scored["age"] <= max_age) & (scored["market_value_in_eur"] >= min_value)
    ]


def make_shortlist(scored, size=None, max_age=None, min_value=None):
    """The one definition of "our recommendation".

    The backtest and the JSON export both call this function, so the number we
    quote in the defence describes the same rule the app ships. Results are
    sorted by residual, so the first 20 rows of a 100-row call are exactly the
    20 rows a size=20 call would return.
    """
    size = size or config.SHORTLIST_SIZE
    max_age = max_age or config.SHORTLIST_MAX_AGE
    min_value = min_value or config.SHORTLIST_MIN_VALUE
    return eligible_pool(scored, max_age, min_value).nsmallest(size, "residual")


def bootstrap_diff_ci(a, b, n_boot=2000, seed=None):
    """95% interval for median(a) - median(b).

    Twenty players is a tiny sample and a tiny sample can look impressive by
    luck. We resample each group with replacement n_boot times, recompute the
    difference every time, and report the 2.5th and 97.5th percentiles. If both
    ends have the same sign, the direction of the result is not a coin flip.
    """
    rng = np.random.default_rng(seed or config.RANDOM_STATE)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if a.size < 2 or b.size < 2:
        return (None, None)
    diffs = [
        np.median(rng.choice(a, a.size)) - np.median(rng.choice(b, b.size))
        for _ in range(n_boot)
    ]
    low, high = np.percentile(diffs, [2.5, 97.5])
    return _round(low), _round(high)


def run_backtest(scored_with_future):
    """Returns (followed, by_decile, measured_shortlist, summary)."""
    followed = scored_with_future[~scored_with_future["censored"]].copy()
    coverage = len(followed) / max(len(scored_with_future), 1)
    if followed.empty:
        raise ValueError(
            "not one player could be followed a year later. Check "
            "HORIZON_MIN_DAYS / HORIZON_MAX_DAYS in config.py against the dates "
            "that actually exist in player_valuations.csv."
        )

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
    if len(undervalued) and len(overvalued):
        _, p_value = mannwhitneyu(undervalued, overvalued, alternative="greater")
    else:
        p_value = float("nan")

    # Two lists, on purpose, and both are reported:
    #   shipped  - built from every scored player. On the day we publish the
    #              list nobody knows who will still be valued a year later, so
    #              filtering on that would mean using the future to pick.
    #   measured - built from the followed players only, because those are the
    #              only rows that have a growth number at all.
    # Treating these two as one list is the exact mistake this project keeps
    # making, so we report the overlap instead of hiding the difference.
    shipped = make_shortlist(scored_with_future)
    measured = make_shortlist(followed)
    overlap = len(set(shipped["player_id"]) & set(measured["player_id"]))

    # precision@k and its baseline are measured on the SAME eligible pool.
    # Comparing an under-26 shortlist against the whole population would
    # manufacture a lift out of the age filter alone.
    pool = eligible_pool(followed)
    precision = (
        (measured["growth_pct"] > config.GROWTH_THRESHOLD).mean()
        if len(measured) else float("nan")
    )
    baseline = (
        (pool["growth_pct"] > config.GROWTH_THRESHOLD).mean()
        if len(pool) else float("nan")
    )
    lift = (
        precision / baseline
        if np.isfinite(baseline) and baseline > 0 else float("nan")
    )

    summary = {
        "n_recommended": int(len(scored_with_future)),
        "n_followed": int(len(followed)),
        "coverage": _round(coverage),
        "median_growth_undervalued": _round(undervalued.median()),
        "median_growth_overvalued": _round(overvalued.median()),
        "diff_95pct_ci": bootstrap_diff_ci(undervalued, overvalued),
        "p_value": _round(p_value, 8),
        "p_value_text": _p_text(p_value),
        "eligible_pool_size": int(len(pool)),
        "shortlist_measured": int(len(measured)),
        "shortlist_shipped": int(len(shipped)),
        "shortlist_overlap": int(overlap),
        "precision_at_k": _round(precision),
        "baseline_rate": _round(baseline),
        "lift": _round(lift, 2),
    }
    return followed, by_decile, measured, summary
