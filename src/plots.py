"""The four figures that go in the report and the slides.

matplotlib only - no seaborn, no styling tricks, so it renders the same
everywhere and nobody has to debug a chart during the defence.
"""

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")  # save to file instead of trying to open a window
import matplotlib.pyplot as plt  # noqa: E402

from src import config  # noqa: E402


def _save(fig, out_dir, name):
    out_dir = Path(out_dir or config.FIGURES)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_decile_growth(by_decile, out_dir=None):
    """THE money chart: cheapest decile on the left, dearest on the right."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(by_decile.index.astype(str), by_decile["median_growth_log"], color="#2b6cb0")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("residual decile  (0 = model says cheapest)")
    ax.set_ylabel("median log growth over 1 year")
    ax.set_title("Do the players we called cheap actually go up?")
    return _save(fig, out_dir, "01_decile_growth.png")


def plot_pred_vs_actual(scored, out_dir=None):
    """How good is the pricing model itself, in millions of euros."""
    fig, ax = plt.subplots(figsize=(5, 5))
    actual = scored["market_value_in_eur"] / 1e6
    predicted = scored["pred_value_eur"] / 1e6
    ax.scatter(actual, predicted, s=8, alpha=0.35, color="#2b6cb0")
    top = max(actual.max(), predicted.max())
    ax.plot([0, top], [0, top], color="red", linewidth=1, label="perfect prediction")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("actual market value (m EUR, log scale)")
    ax.set_ylabel("model value (m EUR, log scale)")
    ax.set_title("Predicted vs actual value")
    ax.legend()
    return _save(fig, out_dir, "02_pred_vs_actual.png")


def plot_importance(importance, out_dir=None):
    """Which columns the model leans on - the 'why' behind the shortlist."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ordered = importance.sort_values()
    ax.barh(ordered.index, ordered.values, color="#2f855a")
    ax.set_xlabel("importance")
    ax.set_title("What drives the model's price")
    return _save(fig, out_dir, "03_feature_importance.png")


def plot_age_curve(scored, out_dir=None):
    """Sanity check: value must peak around 25 or something is wrong."""
    bands = [15, 19, 21, 23, 25, 27, 29, 31, 34, 45]
    grouped = scored.groupby(pd.cut(scored["age"], bands), observed=False)[
        "market_value_in_eur"
    ].median() / 1e6

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([str(i) for i in grouped.index], grouped.values, marker="o", color="#805ad5")
    ax.set_xlabel("age band")
    ax.set_ylabel("median market value (m EUR)")
    ax.set_title("Sanity check: value should peak in the mid-twenties")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return _save(fig, out_dir, "04_age_curve.png")


def save_all(scored, by_decile, importance, out_dir=None):
    """Make every figure in one call and return the paths."""
    paths = [
        plot_decile_growth(by_decile, out_dir),
        plot_pred_vs_actual(scored, out_dir),
        plot_age_curve(scored, out_dir),
    ]
    if len(importance):
        paths.append(plot_importance(importance, out_dir))
    return paths
