"""Turn the five raw Transfermarkt tables into one clean row per player.

The only hard rule in this file: a column may only use information that
existed ON OR BEFORE the snapshot date. If you break that rule the model looks
brilliant on paper and useless in real life.
"""

import numpy as np
import pandas as pd

from src import config


def safe_merge(left, right, how="left", label="", **kwargs):
    """A merge that refuses to silently duplicate rows.

    A one-to-many join is the single most common way a player ends up in the
    table three times with three different targets, which quietly corrupts
    every metric downstream.
    """
    before = len(left)
    out = left.merge(right, how=how, **kwargs)
    if len(out) > before:
        raise ValueError(
            f"{label or 'merge'}: rows grew {before:,} -> {len(out):,}. "
            "The right table has duplicate keys - aggregate it first."
        )
    return out


def _int_key(series):
    """Force a numeric id onto ONE dtype, everywhere.

    Real CSV dumps hand you the same club id as 12, 12.0 and "12" in three
    different files. pandas then merges int64 against object, matches nothing,
    and hands you a table full of empty club names without raising anything.
    Int64 (capital I) is the nullable integer type, so a missing id stays
    missing instead of turning into 0.
    """
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def _text_key(series):
    """Same idea for string ids like "GB1", which arrive with stray spaces."""
    return series.astype(str).str.strip()


def load_raw(raw_dir=None):
    """Read the five CSVs and normalise everything that moves between dumps."""
    raw_dir = raw_dir or config.RAW

    players = pd.read_csv(
        f"{raw_dir}/players.csv",
        parse_dates=["date_of_birth", "contract_expiration_date"],
    )
    valuations = pd.read_csv(f"{raw_dir}/player_valuations.csv", parse_dates=["date"])
    appearances = pd.read_csv(f"{raw_dir}/appearances.csv", parse_dates=["date"])
    clubs = pd.read_csv(f"{raw_dir}/clubs.csv")
    competitions = pd.read_csv(f"{raw_dir}/competitions.csv")

    # the club column in appearances is named differently across dataset dumps
    club_col = "player_club_id" if "player_club_id" in appearances.columns else "club_id"
    appearances = appearances.rename(columns={club_col: "club_id"})

    # --- one dtype per id, in every table -----------------------------------
    for table in (players, valuations, appearances):
        table["player_id"] = _int_key(table["player_id"])
    appearances["club_id"] = _int_key(appearances["club_id"])
    clubs["club_id"] = _int_key(clubs["club_id"])
    appearances["competition_id"] = _text_key(appearances["competition_id"])
    competitions["competition_id"] = _text_key(competitions["competition_id"])
    clubs["domestic_competition_id"] = _text_key(clubs["domestic_competition_id"])

    # match statistics must be numbers, not strings
    for column in ("minutes_played", "goals", "assists", "yellow_cards", "red_cards"):
        appearances[column] = pd.to_numeric(
            appearances[column], errors="coerce"
        ).fillna(0.0)

    valuations["market_value_in_eur"] = pd.to_numeric(
        valuations["market_value_in_eur"], errors="coerce"
    )

    # in this dataset height 0 means "unknown", not "0 cm"
    players["height_in_cm"] = pd.to_numeric(
        players["height_in_cm"], errors="coerce"
    ).replace(0, np.nan)

    # The dump we use (davidcariboo/player-scores) does NOT ship an
    # is_major_national_league column - we opened the file and checked. So in
    # practice the else-branch is the one that runs, and "major league" in this
    # project means "one of the TOP5 ids in config.py". The first branch only
    # exists so a future dump that does have the column is respected instead of
    # being overwritten by our guess.
    if "is_major_national_league" in competitions.columns:
        flag = competitions["is_major_national_league"].astype(str).str.strip().str.lower()
        competitions["is_major"] = flag.isin(["true", "1", "yes", "y"]).astype(int)
    else:
        competitions["is_major"] = (
            competitions["competition_id"].isin(config.TOP5).astype(int)
        )

    return players, valuations, appearances, clubs, competitions


def _target(valuations, snapshot):
    """The newest market value at or before the snapshot - and it must be fresh.

    Taking simply "the last valuation ever recorded before T" attaches a 2015
    price tag to a 2023 row for anyone Transfermarkt stopped re-valuing.
    """
    past = valuations[valuations["date"] <= snapshot].sort_values("date")
    y = past.groupby("player_id").tail(1)[["player_id", "date", "market_value_in_eur"]]
    y = y.rename(columns={"date": "valuation_date"})
    days_old = (snapshot - y["valuation_date"]).dt.days
    keep = (days_old <= config.TARGET_MAX_VALUATION_AGE_DAYS) & (
        y["market_value_in_eur"] > 0
    )
    return y[keep].copy()


def _performance(appearances, competitions, window_start, snapshot):
    """Everything the player did on the pitch inside the observation window."""
    window = appearances[
        (appearances["date"] > window_start) & (appearances["date"] <= snapshot)
    ].copy()
    window = window.merge(
        competitions[["competition_id", "is_major"]], on="competition_id", how="left"
    )
    window["is_major"] = window["is_major"].fillna(0)
    window["major_minutes"] = window["minutes_played"] * window["is_major"]

    perf = (
        window.groupby("player_id")
        .agg(
            games=("game_id", "nunique"),
            minutes=("minutes_played", "sum"),
            goals=("goals", "sum"),
            assists=("assists", "sum"),
            yellows=("yellow_cards", "sum"),
            reds=("red_cards", "sum"),
            n_competitions=("competition_id", "nunique"),
            n_clubs=("club_id", "nunique"),
            major_minutes=("major_minutes", "sum"),
        )
        .reset_index()
    )

    # per-90 rates, because 5 goals in 400 minutes is not 5 goals in 3000
    per90 = (perf["minutes"] / 90).replace(0, np.nan)
    perf["goals_p90"] = perf["goals"] / per90
    perf["assists_p90"] = perf["assists"] / per90
    perf["ga_p90"] = perf["goals_p90"] + perf["assists_p90"]
    perf["yellows_p90"] = perf["yellows"] / per90
    perf["reds_p90"] = perf["reds"] / per90
    perf["min_per_game"] = perf["minutes"] / perf["games"].replace(0, np.nan)
    perf["pct_minutes_major"] = perf["major_minutes"] / perf["minutes"].replace(0, np.nan)
    perf["moved_midseason"] = (perf["n_clubs"] > 1).astype(int)

    # The club we describe him with must be the club he actually played for in
    # the window - not whichever club happened to sit on his valuation row.
    main_club = (
        window.groupby(["player_id", "club_id"])["minutes_played"]
        .sum()
        .reset_index()
        .sort_values(["player_id", "minutes_played"], ascending=[True, False])
        .drop_duplicates("player_id")[["player_id", "club_id"]]
    )
    return safe_merge(perf, main_club, on="player_id", label="perf + main club")


def _player_info(players, snapshot):
    """Static facts about the player, plus age measured at the snapshot."""
    keep = [
        "player_id", "name", "date_of_birth", "position", "sub_position",
        "foot", "height_in_cm", "country_of_citizenship",
        "contract_expiration_date",
    ]
    info = players[keep].copy()
    info["age"] = (snapshot - info["date_of_birth"]).dt.days / 365.25
    info["age_sq"] = info["age"] ** 2  # value peaks around 25, so it is a curve
    info["contract_months_left"] = (
        info["contract_expiration_date"] - snapshot
    ).dt.days / 30.44

    # Group rare nationalities once, from the full player table, so the same
    # categories appear at every snapshot.
    top = (
        players["country_of_citizenship"].value_counts().head(config.TOP_N_COUNTRIES).index
    )
    info["country_grp"] = (
        info["country_of_citizenship"].where(info["country_of_citizenship"].isin(top), "Other")
    ).fillna("Unknown")
    return info


def build_dataset(snapshot, raw=None):
    """Build the modelling table for one snapshot date."""
    players, valuations, appearances, clubs, competitions = (
        raw if raw is not None else load_raw()
    )
    snapshot = pd.Timestamp(snapshot)
    window_start = snapshot - pd.Timedelta(days=config.WINDOW_DAYS)

    df = safe_merge(
        _target(valuations, snapshot),
        _performance(appearances, competitions, window_start, snapshot),
        on="player_id", how="inner", label="target + performance",
    )
    df = safe_merge(df, _player_info(players, snapshot), on="player_id", label="+ player")

    club_cols = [
        "club_id", "name", "domestic_competition_id",
        "squad_size", "average_age", "national_team_players",
    ]
    df = safe_merge(
        df, clubs[club_cols].rename(columns={"name": "club_name"}),
        on="club_id", label="+ club",
    )
    df = safe_merge(
        df,
        competitions[["competition_id", "name", "country_name"]].rename(
            columns={"name": "league_name"}
        ),
        left_on="domestic_competition_id", right_on="competition_id", label="+ league",
    )

    # A join that matched almost nothing is a dtype bug, not a data problem, and
    # it is invisible unless we check for it: every club column would just be
    # empty and the model would quietly train on NaN.
    if len(df):
        matched = float(df["club_name"].notna().mean())
        if matched < 0.5:
            raise ValueError(
                f"only {matched:.0%} of rows found their club. The club_id dtypes "
                "in appearances.csv and clubs.csv do not line up - check "
                "_int_key() in load_raw()."
            )

    df["league_tier"] = (
        df["domestic_competition_id"].map(config.LEAGUE_TIERS).fillna(config.DEFAULT_TIER)
    )

    df = df[
        (df["minutes"] >= config.MIN_MINUTES)
        & df["age"].between(config.MIN_AGE, config.MAX_AGE)
    ].copy()

    df[config.TARGET] = np.log1p(df["market_value_in_eur"])
    df["snapshot"] = snapshot
    return df.reset_index(drop=True)


def assert_no_leakage(df, features=None):
    """Fail loudly if any feature is - or nearly is - the answer.

    Returns the correlation of every numeric feature with the target so you can
    eyeball the top of the list yourself.
    """
    features = list(features or config.FEATURES)

    suspicious = [
        f for f in features
        if any(word in f.lower() for word in config.BANNED_SUBSTRINGS)
    ]
    if suspicious:
        raise AssertionError(f"these feature names look like the target: {suspicious}")

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise AssertionError(f"features missing from the table: {missing}")

    numeric = [f for f in features if f in config.NUMERIC_FEATURES]
    corr = (
        df[numeric].apply(pd.to_numeric, errors="coerce")
        .corrwith(df[config.TARGET])
        .abs()
        .sort_values(ascending=False)
    )
    if corr.max() > 0.98:
        raise AssertionError(
            f"'{corr.idxmax()}' is {corr.max():.3f} correlated with the target. "
            "That is leakage, not a feature."
        )
    return corr
