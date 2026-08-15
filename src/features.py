"""Feature table builder - exactly one row per player."""
import numpy as np
import pandas as pd

from src.config import MIN_MINUTES, RAW, TOP5, WINDOW_DAYS


def load_raw(raw_dir=RAW):
    """Read the five tables and derive the missing columns."""
    players = pd.read_csv(
        f"{raw_dir}/players.csv",
        parse_dates=["date_of_birth", "contract_expiration_date"],
    )
    valuations = pd.read_csv(f"{raw_dir}/player_valuations.csv", parse_dates=["date"])
    apps = pd.read_csv(f"{raw_dir}/appearances.csv", parse_dates=["date"])
    clubs = pd.read_csv(f"{raw_dir}/clubs.csv")
    comps = pd.read_csv(f"{raw_dir}/competitions.csv")

    # competitions.csv has no is_major_national_league column - we build it
    comps["is_major_national_league"] = comps["competition_id"].isin(TOP5).astype(int)

    return players, valuations, apps, clubs, comps


def safe_merge(left, right, how="left", **kwargs):
    """Merge that raises if the row count grows (silent duplication guard)."""
    before = len(left)
    out = left.merge(right, how=how, **kwargs)
    if len(out) > before:
        raise ValueError(
            f"Rows grew from {before} to {len(out)} - aggregate with groupby first"
        )
    return out


def build_dataset(snapshot, raw=None, window_days=WINDOW_DAYS, min_minutes=MIN_MINUTES):
    """Build the train/test table at a given snapshot date."""
    players, valuations, apps, clubs, comps = raw if raw else load_raw()

    T = pd.Timestamp(snapshot)
    T0 = T - pd.Timedelta(days=window_days)

    # 1) TARGET - last valuation at or before the snapshot
    v = valuations[valuations["date"] <= T].sort_values("date")
    y = v.groupby("player_id").tail(1)[
        ["player_id", "market_value_in_eur", "current_club_id"]
    ]

    # 2) Performance - only the 12 months before the snapshot -> no leakage
    w = apps[(apps["date"] > T0) & (apps["date"] <= T)]
    perf = (
        w.groupby("player_id")
        .agg(
            games=("game_id", "nunique"),
            minutes=("minutes_played", "sum"),
            goals=("goals", "sum"),
            assists=("assists", "sum"),
            yellows=("yellow_cards", "sum"),
            reds=("red_cards", "sum"),
            n_comps=("competition_id", "nunique"),
        )
        .reset_index()
    )

    p90 = (perf["minutes"] / 90).replace(0, np.nan)
    perf["goals_p90"] = perf["goals"] / p90
    perf["assists_p90"] = perf["assists"] / p90
    perf["ga_p90"] = perf["goals_p90"].fillna(0) + perf["assists_p90"].fillna(0)
    perf["yellows_p90"] = perf["yellows"] / p90
    perf["min_per_game"] = perf["minutes"] / perf["games"].replace(0, np.nan)

    # 3) Demographics - never take market_value_in_eur from players.csv
    p = players[
        [
            "player_id",
            "name",
            "date_of_birth",
            "position",
            "sub_position",
            "foot",
            "height_in_cm",
            "country_of_citizenship",
            "contract_expiration_date",
            "current_club_name",
        ]
    ].copy()
    p["age"] = (T - p["date_of_birth"]).dt.days / 365.25
    p["contract_months_left"] = (p["contract_expiration_date"] - T).dt.days / 30.44

    # 4) Merge
    df = safe_merge(y, perf, on="player_id", how="inner")
    df = safe_merge(df, p, on="player_id", how="left")
    df = safe_merge(
        df,
        clubs[
            [
                "club_id",
                "domestic_competition_id",
                "squad_size",
                "average_age",
                "national_team_players",
            ]
        ],
        left_on="current_club_id",
        right_on="club_id",
        how="left",
    )
    df = safe_merge(
        df,
        comps[["competition_id", "country_name", "is_major_national_league"]],
        left_on="domestic_competition_id",
        right_on="competition_id",
        how="left",
    )

    # 5) Filter and target
    df = df[(df["minutes"] >= min_minutes) & (df["market_value_in_eur"] > 0)]
    df["y_log"] = np.log1p(df["market_value_in_eur"])
    df["snapshot"] = T

    return df.reset_index(drop=True)


# Model inputs - never add anything derived from market value
NUMERIC_FEATURES = [
    "age",
    "games",
    "minutes",
    "min_per_game",
    "goals_p90",
    "assists_p90",
    "ga_p90",
    "yellows_p90",
    "n_comps",
    "height_in_cm",
    "contract_months_left",
    "squad_size",
    "average_age",
    "national_team_players",
    "is_major_national_league",
]
CATEGORICAL_FEATURES = ["position", "sub_position", "foot"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
