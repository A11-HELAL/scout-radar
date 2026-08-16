"""Tests for the whole pipeline.

Run them with:      python -m pytest -q

They use tests/make_fake_data.py, so they finish in seconds and do not need the
1 GB Kaggle download. Most tests do not need scikit-learn either: TinyModel
stands in for the real pipeline, because what we are testing here is OUR logic,
not theirs.

The fake data has a mispricing signal planted in it on purpose, so
test_signal_is_recovered is a known-answer test: if the pipeline cannot find a
signal we put there ourselves, the pipeline is broken.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.make_fake_data import make_fake_data  # noqa: E402


class TinyModel:
    """Least squares on the numeric columns - a stand-in for the real pipeline."""

    def fit(self, X, y):
        design = self._design(X)
        self.beta, *_ = np.linalg.lstsq(design, np.asarray(y, dtype=float), rcond=None)
        return self

    def predict(self, X):
        return self._design(X) @ self.beta

    @staticmethod
    def _design(X):
        numbers = X.select_dtypes(include=[np.number]).astype(float)
        numbers = numbers.fillna(numbers.median())
        return np.column_stack([np.ones(len(numbers)), numbers.to_numpy()])


@pytest.fixture(scope="session")
def pipeline(tmp_path_factory):
    """Build fake data once, then run the real pipeline over it."""
    raw_dir = tmp_path_factory.mktemp("raw")
    make_fake_data(raw_dir)
    os.environ["SCOUT_RAW"] = str(raw_dir)

    from src import config
    from src.backtest import add_residuals, attach_future_value, run_backtest
    from src.features import build_dataset, load_raw

    config.RAW = str(raw_dir)
    raw = load_raw(str(raw_dir))
    train = build_dataset(config.SNAPSHOT_TRAIN, raw)
    test = build_dataset(config.SNAPSHOT_TEST, raw)

    model = TinyModel().fit(train[config.NUMERIC_FEATURES], train[config.TARGET])
    scored = add_residuals(train, model, features=config.NUMERIC_FEATURES)
    scored = attach_future_value(scored, raw[1])
    followed, by_decile, shortlist, summary = run_backtest(scored)

    return {
        "config": config, "raw": raw, "train": train, "test": test,
        "scored": scored, "followed": followed, "by_decile": by_decile,
        "shortlist": shortlist, "summary": summary, "raw_dir": str(raw_dir),
    }


# ---------------------------------------------------------------- data shape

def test_one_row_per_player(pipeline):
    for name in ("train", "test"):
        assert pipeline[name]["player_id"].is_unique, f"{name} has duplicate players"


def test_every_feature_exists(pipeline):
    config = pipeline["config"]
    missing = [f for f in config.FEATURES if f not in pipeline["train"].columns]
    assert not missing, f"features missing from the table: {missing}"


def test_filters_are_applied(pipeline):
    config, train = pipeline["config"], pipeline["train"]
    assert train["minutes"].min() >= config.MIN_MINUTES
    assert train["market_value_in_eur"].min() > 0
    assert train["age"].between(config.MIN_AGE, config.MAX_AGE).all()


def test_height_zero_became_missing(pipeline):
    """0 cm is not a height, it is a missing value."""
    assert (pipeline["train"]["height_in_cm"] == 0).sum() == 0


# ------------------------------------------------------------------- leakage

def test_no_future_appearances_leak_in(pipeline):
    """Recompute one player's minutes by hand and demand the same number."""
    config, train, appearances = pipeline["config"], pipeline["train"], pipeline["raw"][2]
    snapshot = pd.Timestamp(config.SNAPSHOT_TRAIN)
    start = snapshot - pd.Timedelta(days=config.WINDOW_DAYS)

    row = train.iloc[0]
    mine = appearances[
        (appearances["player_id"] == row["player_id"])
        & (appearances["date"] > start)
        & (appearances["date"] <= snapshot)
    ]
    assert row["minutes"] == mine["minutes_played"].sum()


def test_target_valuation_is_not_stale(pipeline):
    config, train = pipeline["config"], pipeline["train"]
    days_old = (pd.Timestamp(config.SNAPSHOT_TRAIN) - train["valuation_date"]).dt.days
    assert days_old.max() <= config.MAX_VALUATION_AGE_DAYS
    assert days_old.min() >= 0


def test_leakage_guard_catches_an_injected_leak(pipeline):
    """Plant the answer as a feature; assert_no_leakage must refuse it."""
    from src.features import assert_no_leakage

    config, train = pipeline["config"], pipeline["train"]
    assert_no_leakage(train)  # the clean table must pass

    leaky = train.copy()
    leaky["squad_size"] = leaky[config.TARGET]  # a clean name hiding the answer
    with pytest.raises(AssertionError, match="leakage"):
        assert_no_leakage(leaky)


def test_leakage_guard_catches_a_banned_name(pipeline):
    from src.features import assert_no_leakage

    with pytest.raises(AssertionError, match="look like the target"):
        assert_no_leakage(pipeline["train"], features=["age", "highest_market_value"])


# ------------------------------------------------------------------ backtest

def test_future_value_comes_after_the_snapshot(pipeline):
    scored = pipeline["scored"]
    followed = scored[~scored["censored"]]
    assert (followed["future_date"] > followed["snapshot"]).all()
    assert followed["future_value_eur"].notna().all()


def test_players_we_cannot_follow_are_censored_not_dropped(pipeline):
    """The censored rows must still be there, so coverage is honest."""
    scored, summary = pipeline["scored"], pipeline["summary"]
    assert len(scored) == summary["n_recommended"]
    assert summary["n_followed"] <= summary["n_recommended"]
    assert 0 < summary["coverage"] <= 1
    assert scored.loc[scored["censored"], "future_value_eur"].isna().all()


def test_growth_matches_the_two_raw_numbers(pipeline):
    followed = pipeline["followed"]
    expected = followed["future_value_eur"] / followed["market_value_in_eur"] - 1
    assert np.allclose(followed["growth_pct"], expected)


def test_shortlist_obeys_the_published_rules(pipeline):
    from src.backtest import eligible_pool

    config, shortlist = pipeline["config"], pipeline["shortlist"]
    assert len(shortlist) <= config.SHORTLIST_SIZE
    assert (shortlist["age"] <= config.SHORTLIST_MAX_AGE).all()
    assert (shortlist["market_value_in_eur"] >= config.SHORTLIST_MIN_VALUE).all()

    # it must really be the most-undervalued end of the eligible pool:
    # nobody left outside may have a smaller residual than anyone inside
    pool = eligible_pool(pipeline["followed"])
    outside = pool[~pool["player_id"].isin(shortlist["player_id"])]
    assert shortlist["residual"].max() <= outside["residual"].min() + 1e-9


def test_baseline_uses_the_same_pool_as_the_shortlist(pipeline):
    """Otherwise the age filter alone would manufacture a lift."""
    from src.backtest import eligible_pool

    followed, summary = pipeline["followed"], pipeline["summary"]
    assert summary["eligible_pool_size"] == len(eligible_pool(followed))
    assert summary["eligible_pool_size"] < len(followed)


def test_signal_is_recovered(pipeline):
    """Known-answer test: undervalued must grow more than overvalued.

    The fake data was generated with value = fair_value * exp(eps) and eps
    decaying, so a negative residual MUST predict growth.
    """
    followed, by_decile = pipeline["followed"], pipeline["by_decile"]
    correlation = followed["residual"].corr(followed["growth_log"], method="spearman")
    assert correlation < -0.15, f"expected a negative link, got {correlation:.3f}"

    cheapest = by_decile["median_growth_log"].loc[by_decile.index.min()]
    dearest = by_decile["median_growth_log"].loc[by_decile.index.max()]
    assert cheapest > dearest


# -------------------------------------------------------------------- export

def test_export_json_is_valid_and_leaks_nothing(pipeline, tmp_path):
    from src.export import export_shortlist

    path, shortlist = export_shortlist(pipeline["scored"], out_dir=tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["players"], "the export is empty"
    assert len(payload["players"]) == len(shortlist)

    card = payload["players"][0]
    for key in ("name", "age", "market_value_eur", "model_value_eur",
                "undervalued_pct", "reason"):
        assert key in card, f"the UI needs {key}"
    # the file must never carry the future outcome into the front-end
    assert "future_value_eur" not in card
    assert "growth_pct" not in card
    json.dumps(payload)  # must stay JSON-serialisable


# ------------------------------------------------- the real sklearn pipeline

def test_real_models_train_and_beat_the_baseline(pipeline):
    """Skipped automatically if scikit-learn is not installed."""
    pytest.importorskip("sklearn")
    from src.train import train_all

    leaderboard, fitted, best = train_all(pipeline["train"], pipeline["test"])
    assert best != "baseline_mean", "no model beat predicting the average"
    assert leaderboard.loc[best, "MAE_log"] < leaderboard.loc["baseline_mean", "MAE_log"]
    assert leaderboard["Spearman"].max() > 0.5
