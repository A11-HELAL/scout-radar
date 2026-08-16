"""Tests that check the pipeline is CORRECT, not merely that it runs.

Everything here runs on generated data (tests/make_fake_data.py) where we know
the truth, so a failure points straight at our code. Read the test names top to
bottom: they are a checklist of every way this project could quietly be wrong.

    python -m pytest -q
"""

import json

import numpy as np
import pandas as pd
import pytest

from src import config
from src.backtest import (
    add_residuals,
    attach_future_value,
    eligible_pool,
    make_shortlist,
    run_backtest,
)
from src.export import export_shortlist
from src.features import assert_no_leakage, build_dataset, load_raw, safe_merge
from src.train import train_all
from tests.make_fake_data import make_fake_data


class TinyModel:
    """A fake fitted model: predicts exactly what we tell it to predict."""

    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def predict(self, X):
        return self.values[: len(X)]


def _reject_constant(value):
    raise AssertionError(f"the file contains the invalid JSON constant {value}")


@pytest.fixture(scope="session")
def pipeline(tmp_path_factory):
    """Build both snapshots once, then let every test assert against them."""
    raw_dir = tmp_path_factory.mktemp("raw")
    make_fake_data(raw_dir)
    raw = load_raw(raw_dir)
    return {
        "raw_dir": raw_dir,
        "raw": raw,
        "train": build_dataset(config.SNAPSHOT_TRAIN, raw=raw),
        "test": build_dataset(config.SNAPSHOT_TEST, raw=raw),
    }


@pytest.fixture(scope="session")
def fitted(pipeline):
    """Train the real models once - they are the slow part of the suite."""
    leaderboard, models, _ = train_all(pipeline["train"], pipeline["test"])
    return {"leaderboard": leaderboard, "models": models}


@pytest.fixture(scope="session")
def backtested(pipeline, fitted):
    """Score June 2023 and look up what really happened by June 2024."""
    _, valuations, _, _, _ = pipeline["raw"]
    scored = add_residuals(pipeline["train"], fitted["models"]["gradient_boosting"])
    scored = attach_future_value(scored, valuations)
    followed, by_decile, measured, summary = run_backtest(scored)
    return {
        "scored": scored,
        "followed": followed,
        "by_decile": by_decile,
        "measured": measured,
        "summary": summary,
    }


# --- the table itself ------------------------------------------------------

def test_both_snapshots_have_rows(pipeline):
    assert len(pipeline["train"]) > 200
    assert len(pipeline["test"]) > 200


def test_one_row_per_player(pipeline):
    for name in ("train", "test"):
        assert pipeline[name]["player_id"].is_unique, f"{name}: a player is duplicated"


def test_filters_are_applied(pipeline):
    train = pipeline["train"]
    assert train["minutes"].min() >= config.MIN_MINUTES
    assert train["age"].between(config.MIN_AGE, config.MAX_AGE).all()
    assert (train["market_value_in_eur"] > 0).all()
    # the filters have to actually remove somebody, otherwise they prove nothing
    assert len(train) < 900


def test_target_valuation_is_not_stale(pipeline):
    train = pipeline["train"]
    days_old = (pd.Timestamp(config.SNAPSHOT_TRAIN) - train["valuation_date"]).dt.days
    assert (days_old >= 0).all(), "a target valuation comes from after the snapshot"
    assert days_old.max() <= config.TARGET_MAX_VALUATION_AGE_DAYS


def test_no_future_appearances_leak_in(pipeline):
    """Recompute one player's minutes by hand and compare."""
    _, _, appearances, _, _ = pipeline["raw"]
    snapshot = pd.Timestamp(config.SNAPSHOT_TRAIN)
    window_start = snapshot - pd.Timedelta(days=config.WINDOW_DAYS)

    row = pipeline["train"].iloc[0]
    mine = appearances[appearances["player_id"] == row["player_id"]]
    inside = mine[(mine["date"] > window_start) & (mine["date"] <= snapshot)]

    assert row["minutes"] == pytest.approx(inside["minutes_played"].sum())
    # and there really is data outside the window that we correctly ignored
    assert inside["minutes_played"].sum() < mine["minutes_played"].sum()


def test_ids_were_normalised(pipeline):
    """make_fake_data writes club_id as a float and competition_id with spaces.

    Without the normalisation in load_raw() these joins match nothing, every
    club column comes back empty, and nothing anywhere raises an error.
    """
    train = pipeline["train"]
    assert train["club_name"].notna().all()
    assert train["league_name"].notna().all()
    assert train["league_tier"].notna().all()
    # pct_minutes_major is only non-zero if the competitions join worked
    assert train["pct_minutes_major"].gt(0).any()
    assert train["pct_minutes_major"].lt(1).any()


def test_major_league_flag_falls_back_to_top5(pipeline):
    """The real dump has no is_major_national_league column, so TOP5 is used."""
    on_disk = pd.read_csv(pipeline["raw_dir"] / "competitions.csv")
    assert "is_major_national_league" not in on_disk.columns

    _, _, _, _, competitions = pipeline["raw"]
    expected = competitions["competition_id"].isin(config.TOP5).astype(int)
    assert (competitions["is_major"] == expected).all()


# --- leakage ---------------------------------------------------------------

def test_leakage_guard_catches_an_obvious_name(pipeline):
    with pytest.raises(AssertionError):
        assert_no_leakage(pipeline["train"], config.FEATURES + ["market_value_in_eur"])


def test_leakage_guard_catches_a_renamed_copy(pipeline, monkeypatch):
    """Somebody copies the target under an innocent name. Still has to fire."""
    train = pipeline["train"].copy()
    train["form_index"] = train[config.TARGET] * 1.001 + 0.002
    monkeypatch.setattr(
        config, "NUMERIC_FEATURES", config.NUMERIC_FEATURES + ["form_index"]
    )
    with pytest.raises(AssertionError, match="correlated"):
        assert_no_leakage(train, config.FEATURES + ["form_index"])


def test_safe_merge_refuses_to_duplicate_rows():
    left = pd.DataFrame({"player_id": [1, 2]})
    right = pd.DataFrame({"player_id": [1, 1], "x": [10, 20]})
    with pytest.raises(ValueError, match="rows grew"):
        safe_merge(left, right, on="player_id", label="on purpose")


# --- the models ------------------------------------------------------------

def test_real_models_beat_the_dummy(fitted):
    board = fitted["leaderboard"].set_index("model")
    dummy = board.loc["baseline_mean", "MAE_log"]
    assert board.loc["gradient_boosting", "MAE_log"] < dummy
    assert board.loc["random_forest", "MAE_log"] < dummy


def test_ranking_is_recovered(fitted):
    """The quality signal we planted must survive the whole pipeline."""
    board = fitted["leaderboard"].set_index("model")
    assert board.loc["gradient_boosting", "Spearman"] > 0.5


def test_medape_is_a_percent(fitted):
    """38.0 means 38%. If this ever drops below 1, it went back to a fraction."""
    best = fitted["leaderboard"].iloc[0]
    assert 1 < best["MedAPE_pct"] < 500


def test_residual_is_market_minus_model(pipeline):
    train = pipeline["train"].head(50)
    scored = add_residuals(train, TinyModel(np.full(len(train), 15.0)))
    assert scored["residual"].to_numpy() == pytest.approx(
        (train[config.TARGET] - 15.0).to_numpy()
    )


# --- the backtest ----------------------------------------------------------

def test_future_value_is_inside_the_horizon_window(backtested):
    """A price refreshed a month after the snapshot is NOT a one-year outcome."""
    followed = backtested["followed"]
    days = (followed["future_date"] - followed["snapshot"]).dt.days
    assert days.min() >= config.HORIZON_MIN_DAYS
    assert days.max() <= config.HORIZON_MAX_DAYS


def test_players_we_cannot_follow_are_censored_not_dropped(backtested, pipeline):
    scored = backtested["scored"]
    assert len(scored) == len(pipeline["train"]), "somebody vanished from the table"
    assert scored["censored"].any(), "the fake data should contain censored players"
    assert 0 < backtested["summary"]["coverage"] < 1


def test_signal_is_recovered(backtested):
    """The players the model called cheap must really grow. This IS the project."""
    summary = backtested["summary"]
    assert summary["median_growth_undervalued"] > summary["median_growth_overvalued"]
    low, _high = summary["diff_95pct_ci"]
    assert low > 0, "the 95% interval straddles zero - no result to report"


def test_cheapest_decile_grows_more_than_the_dearest(backtested):
    by_decile = backtested["by_decile"]["median_growth_log"]
    assert by_decile.iloc[0] > by_decile.iloc[-1]


def test_shortlist_respects_its_own_rules(backtested):
    shortlist = backtested["measured"]
    assert len(shortlist) == config.SHORTLIST_SIZE
    assert (shortlist["age"] <= config.SHORTLIST_MAX_AGE).all()
    assert (shortlist["market_value_in_eur"] >= config.SHORTLIST_MIN_VALUE).all()
    assert shortlist["residual"].is_monotonic_increasing


def test_baseline_uses_the_same_pool_as_the_shortlist(backtested):
    """Otherwise the age filter alone manufactures a lift out of nothing."""
    followed = backtested["followed"]
    pool = eligible_pool(followed)
    assert len(pool) < len(followed)
    assert (pool["age"] <= config.SHORTLIST_MAX_AGE).all()
    assert backtested["summary"]["eligible_pool_size"] == len(pool)


# --- the export ------------------------------------------------------------

def test_export_writes_both_files(backtested, tmp_path):
    exported = export_shortlist(backtested["scored"], out_dir=tmp_path)
    assert exported["web_path"].exists()
    assert exported["full_path"].exists()
    assert len(exported["web"]) == config.EXPORT_SIZE
    assert len(exported["shortlist"]) == config.SHORTLIST_SIZE


def test_website_file_keeps_the_frozen_contract(backtested, tmp_path):
    """The front-end reads a plain ARRAY with these exact field names."""
    exported = export_shortlist(backtested["scored"], out_dir=tmp_path)
    cards = json.loads(exported["web_path"].read_text(encoding="utf-8"))

    assert isinstance(cards, list), "the site expects a top-level array"
    assert len(cards) == config.EXPORT_SIZE
    for field in (
        "rank", "player_id", "name", "age", "position", "club", "league",
        "market_value_eur", "predicted_value_eur", "gap_pct",
        "contract_months_left", "minutes", "reasons",
    ):
        assert field in cards[0], f"the website needs the field {field}"
    assert cards[0]["rank"] == 1
    assert isinstance(cards[0]["reasons"], list)


def test_export_never_writes_nan(backtested, tmp_path):
    """json.dumps writes NaN as a bare token, and JSON.parse rejects it."""
    exported = export_shortlist(backtested["scored"], out_dir=tmp_path)
    for path in (exported["web_path"], exported["full_path"]):
        text = path.read_text(encoding="utf-8")
        assert "NaN" not in text
        json.loads(text, parse_constant=_reject_constant)


def test_measured_twenty_is_the_head_of_the_published_hundred(backtested, tmp_path):
    """The list we quote numbers about is the top of the list we publish."""
    exported = export_shortlist(backtested["scored"], out_dir=tmp_path)
    top = exported["web"]["player_id"].head(config.SHORTLIST_SIZE).tolist()
    assert exported["shortlist"]["player_id"].tolist() == top
    assert make_shortlist(backtested["scored"])["player_id"].tolist() == top


def test_report_file_carries_the_rules_and_the_meta(backtested, tmp_path):
    exported = export_shortlist(
        backtested["scored"], out_dir=tmp_path,
        meta={"best_model": "gradient_boosting"},
    )
    payload = json.loads(exported["full_path"].read_text(encoding="utf-8"))
    assert payload["snapshot"] == config.SNAPSHOT_TRAIN
    assert payload["rules"]["max_age"] == config.SHORTLIST_MAX_AGE
    assert payload["rules"]["min_market_value_eur"] == config.SHORTLIST_MIN_VALUE
    assert payload["model"]["best_model"] == "gradient_boosting"
    assert len(payload["players"]) == config.SHORTLIST_SIZE


# --- the figures -----------------------------------------------------------

def test_figures_are_written(backtested, tmp_path):
    plots = pytest.importorskip("src.plots")  # skipped if matplotlib is missing
    importance = pd.DataFrame({
        "feature": ["age", "minutes"],
        "importance": [0.20, 0.10],
        "std": [0.01, 0.01],
    })
    paths = plots.save_all(
        backtested["followed"], backtested["by_decile"], importance, out_dir=tmp_path
    )
    assert len(paths) == 4
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)
