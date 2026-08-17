"""The four figures for the report. One idea per figure, no decoration."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402

BLUE = "#3b6fb6"


def _save(fig, name, out_dir=None):
    out_dir = Path(out_dir or config.FIGURES)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_decile_growth(by_decile, out_dir=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(by_decile.index.astype(str), by_decile["median_growth_log"], color=BLUE)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("residual decile   (0 = the model says cheapest)")
    ax.set_ylabel("median growth in log value")
    ax.set_title("One year later: did the market catch up?")
    return _save(fig, "decile_growth.png", out_dir)


def plot_pred_vs_actual(scored, out_dir=None):
    fig, ax = plt.subplots(figsize=(5, 5))
    actual = scored["market_value_in_eur"]
    predicted = scored["pred_value_eur"]
    ax.scatter(actual, predicted, s=8, alpha=0.35, color=BLUE)

    low = max(min(actual.min(), predicted.min()), 1_000)
    high = max(actual.max(), predicted.max())
    ax.plot([low, high], [low, high], color="black", linewidth=1, linestyle="--")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("market value (EUR)")
    ax.set_ylabel("model value (EUR)")
    ax.set_title("Below the line = the market pays more than the model")
    return _save(fig, "pred_vs_actual.png", out_dir)


def plot_importance(importance, out_dir=None, top=15):
    rows = importance.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(rows["feature"], rows["importance"], xerr=rows["std"], color=BLUE)
    ax.set_xlabel("drop in score when the column is shuffled")
    ax.set_title("Permutation importance (measured on the test snapshot)")
    return _save(fig, "feature_importance.png", out_dir)


def plot_age_curve(scored, out_dir=None):
    bins = pd.cut(scored["age"], bins=[15, 19, 21, 23, 25, 27, 29, 31, 34, 45])
    curve = scored.groupby(bins, observed=True)["market_value_in_eur"].median()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([str(i) for i in curve.index], curve.values, marker="o", color=BLUE)
    ax.set_yscale("log")
    ax.set_xlabel("age band")
    ax.set_ylabel("median market value (EUR, log scale)")
    ax.set_title("Value peaks in the mid twenties")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return _save(fig, "age_curve.png", out_dir)


def save_all(followed, by_decile, importance, out_dir=None):
    return [
        plot_decile_growth(by_decile, out_dir),
        plot_pred_vs_actual(followed, out_dir),
        plot_importance(importance, out_dir),
        plot_age_curve(followed, out_dir),
    ]
