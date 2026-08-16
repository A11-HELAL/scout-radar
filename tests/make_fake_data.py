"""Generate tiny Transfermarkt-shaped CSVs that contain a KNOWN mispricing signal.

Why this exists: the real Kaggle dump is ~1 GB. This module builds the same five
tables in a few seconds so the whole pipeline can be tested anywhere, and so we
can check that the backtest recovers a signal we planted on purpose.

Ground truth planted here:
    log(market_value) = log(fair_value) + eps
    eps_next = 0.5 * eps + noise          (mispricing decays)
So a player with a negative eps (undervalued) must grow faster than average.
If the pipeline cannot see that, the pipeline is broken - not the data.
"""

from pathlib import Path

import numpy as np
import pandas as pd

LEAGUE_TIERS = {
    "GB1": 1, "ES1": 1, "L1": 1, "IT1": 1, "FR1": 1,
    "PO1": 2, "NL1": 2, "TR1": 2, "GR1": 3, "DK1": 3,
}
CUPS = ["CL", "EL"]
SEASONS = [2021, 2022, 2023, 2024]
VALUATION_DATES = pd.to_datetime([
    "2021-12-01", "2022-06-01", "2022-12-01", "2023-06-01",
    "2023-12-01", "2024-06-01", "2024-12-01", "2025-06-01",
])
SUB_POSITIONS = {
    "Attack": ["Centre-Forward", "Left Winger", "Right Winger"],
    "Midfield": ["Central Midfield", "Attacking Midfield", "Defensive Midfield"],
    "Defender": ["Centre-Back", "Left-Back", "Right-Back"],
    "Goalkeeper": ["Goalkeeper"],
}
GOAL_RATE = {"Attack": 0.45, "Midfield": 0.18, "Defender": 0.05, "Goalkeeper": 0.0}
ASSIST_RATE = {"Attack": 0.20, "Midfield": 0.22, "Defender": 0.08, "Goalkeeper": 0.0}
TIER_PREMIUM = {1: 0.90, 2: 0.35, 3: 0.0}


def _competitions():
    leagues = list(LEAGUE_TIERS)
    ids = leagues + CUPS
    return pd.DataFrame({
        "competition_id": ids,
        "competition_code": [i.lower() for i in ids],
        "name": [f"League {i}" for i in leagues] + ["Champions League", "Europa League"],
        "type": ["domestic_league"] * len(leagues) + ["international_cup"] * len(CUPS),
        "country_name": [f"Country {i}" for i in leagues] + [None] * len(CUPS),
        "domestic_league_code": leagues + [None] * len(CUPS),
        # real file stores lowercase strings, not python booleans
        "is_major_national_league": [
            "true" if LEAGUE_TIERS[i] == 1 else "false" for i in leagues
        ] + ["false"] * len(CUPS),
    })


def _clubs(rng, n_clubs):
    return pd.DataFrame({
        "club_id": np.arange(1, n_clubs + 1),
        "name": [f"Club {i}" for i in range(1, n_clubs + 1)],
        "domestic_competition_id": rng.choice(list(LEAGUE_TIERS), n_clubs),
        "total_market_value": np.nan,
        "squad_size": rng.integers(18, 33, n_clubs),
        "average_age": np.round(rng.normal(26.0, 1.4, n_clubs), 1),
        "national_team_players": rng.integers(0, 13, n_clubs),
        "last_season": 2024,
    })


def _players(rng, n_players, clubs):
    positions = rng.choice(list(SUB_POSITIONS), n_players, p=[0.30, 0.33, 0.30, 0.07])
    heights = rng.normal(182, 6, n_players).round()
    heights[rng.random(n_players) < 0.05] = 0  # real data encodes missing height as 0
    club_ids = rng.choice(clubs["club_id"], n_players)
    names = dict(zip(clubs["club_id"], clubs["name"]))
    return pd.DataFrame({
        "player_id": np.arange(1, n_players + 1),
        "name": [f"Player {i}" for i in range(1, n_players + 1)],
        "date_of_birth": pd.Timestamp("1988-01-01")
        + pd.to_timedelta(rng.integers(0, 5200, n_players), unit="D"),
        "position": positions,
        "sub_position": [rng.choice(SUB_POSITIONS[p]) for p in positions],
        "foot": rng.choice(["right", "left", "both", None], n_players, p=[.72, .22, .04, .02]),
        "height_in_cm": heights,
        "country_of_citizenship": rng.choice([f"Nation {i}" for i in range(40)], n_players),
        "contract_expiration_date": pd.Timestamp("2025-06-30")
        + pd.to_timedelta(rng.integers(-700, 1300, n_players), unit="D"),
        "current_club_id": club_ids,
        "current_club_name": [names[c] for c in club_ids],
    })


def _appearances(rng, players, clubs, quality):
    league_of = dict(zip(clubs["club_id"], clubs["domestic_competition_id"]))
    club_pool = clubs["club_id"].to_numpy()
    rows = []
    appearance_id = 0
    for idx, player_id in enumerate(players["player_id"]):
        club = rng.choice(club_pool)
        share = 1 / (1 + np.exp(-quality[idx]))
        for season in SEASONS:
            if rng.random() < 0.15:
                club = rng.choice(club_pool)
            clubs_this_season = [club]
            if rng.random() < 0.12:
                clubs_this_season.append(rng.choice(club_pool))
            n_games = int(np.clip(rng.poisson(26) + 4, 6, 38))
            dates = pd.Timestamp(f"{season}-08-05") + pd.to_timedelta(
                np.sort(rng.integers(0, 290, n_games)), unit="D"
            )
            position = players["position"].iat[idx]
            for j, date in enumerate(dates):
                match_club = clubs_this_season[0] if j < n_games * 0.7 else clubs_this_season[-1]
                minutes = int(np.clip(rng.normal(35 + 55 * share, 22), 1, 90))
                comp = league_of[match_club] if rng.random() > 0.12 else rng.choice(CUPS)
                per90 = minutes / 90
                rows.append((
                    appearance_id, int(rng.integers(1, 400_000)), int(player_id),
                    int(match_club), int(club), date, comp,
                    int(rng.random() < 0.12), int(rng.random() < 0.01),
                    int(rng.poisson(GOAL_RATE[position] * (0.5 + share) * per90)),
                    int(rng.poisson(ASSIST_RATE[position] * (0.5 + share) * per90)),
                    minutes,
                ))
                appearance_id += 1
    return pd.DataFrame(rows, columns=[
        "appearance_id", "game_id", "player_id", "player_club_id",
        "player_current_club_id", "date", "competition_id",
        "yellow_cards", "red_cards", "goals", "assists", "minutes_played",
    ])


def _valuations(rng, players, clubs, quality):
    tier_of = {c: LEAGUE_TIERS[l] for c, l in
               zip(clubs["club_id"], clubs["domestic_competition_id"])}
    rows = []
    for idx, player_id in enumerate(players["player_id"]):
        dob = players["date_of_birth"].iat[idx]
        club = players["current_club_id"].iat[idx]
        premium = TIER_PREMIUM[tier_of[club]]
        eps = rng.normal(0, 0.55)
        retires_at = rng.choice(VALUATION_DATES[3:]) if rng.random() < 0.06 else None
        for date in VALUATION_DATES:
            if retires_at is not None and date > retires_at:
                break
            if rng.random() < 0.08:  # market_value simply not refreshed that period
                continue
            age = (date - dob).days / 365.25
            log_fair = 13.9 + 1.15 * quality[idx] - 0.021 * (age - 25.0) ** 2 + premium
            eps = 0.5 * eps + rng.normal(0, 0.30)
            rows.append((int(player_id), date,
                         float(np.expm1(log_fair + eps).round(-4) + 25_000), int(club)))
    return pd.DataFrame(rows, columns=[
        "player_id", "date", "market_value_in_eur", "current_club_id",
    ])


def make_fake_data(out_dir, n_players=900, n_clubs=60, seed=7):
    """Write the five CSVs into out_dir and return them as a dict."""
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    competitions = _competitions()
    clubs = _clubs(rng, n_clubs)
    players = _players(rng, n_players, clubs)
    quality = rng.normal(0, 1, n_players)

    appearances = _appearances(rng, players, clubs, quality)
    valuations = _valuations(rng, players, clubs, quality)

    # In the real dataset, current_club_id on a valuation row is the club the
    # player was at ON THAT DATE. Reproduce that instead of a fixed club.
    history = appearances.sort_values("date")[["player_id", "date", "player_club_id"]]
    valuations = pd.merge_asof(
        valuations.sort_values("date"), history,
        on="date", by="player_id", direction="backward",
    )
    valuations["current_club_id"] = (
        valuations["player_club_id"].fillna(valuations["current_club_id"]).astype(int)
    )
    valuations = valuations.drop(columns=["player_club_id"])

    last = valuations.sort_values("date").drop_duplicates("player_id", keep="last")
    value_of = dict(zip(last["player_id"], last["market_value_in_eur"]))
    peak = valuations.groupby("player_id")["market_value_in_eur"].max()
    # leakage bait: the real players.csv carries these two columns too
    players["market_value_in_eur"] = players["player_id"].map(value_of)
    players["highest_market_value_in_eur"] = players["player_id"].map(peak)

    tables = {
        "players": players, "player_valuations": valuations,
        "appearances": appearances, "clubs": clubs, "competitions": competitions,
    }
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    return tables


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "data/raw_fake"
    for name, table in make_fake_data(target).items():
        print(f"{name:20s} {table.shape}")
    print(f"written to {target}")
