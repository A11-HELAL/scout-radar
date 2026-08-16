"""Generate Transfermarkt-shaped CSVs with a signal planted inside them.

The real dataset cannot tell us whether our pipeline is correct, because nobody
knows the true fair value of a real footballer. So here we invent it: every fake
player gets a hidden `quality`, a fair value computed from it, and a temporary
gap between that fair value and the price the fake market prints. The gap halves
every six months, so the players we made cheap really do grow afterwards - and
the tests can check that the pipeline finds them.

If a test fails on this data, the bug is in our code, not in the data.

None of this is meant to look like real football. It is meant to be a ruler.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# index 3 is SNAPSHOT_TRAIN and index 5 is one year later - the exact pair the
# backtest measures. If you edit this list, keep that pair intact.
VALUATION_DATES = [
    "2021-12-01", "2022-06-01", "2022-12-01", "2023-06-01",
    "2023-12-01", "2024-06-01", "2024-12-01", "2025-06-01",
]
HORIZON_INDEX = 5

SEASONS = [2021, 2022, 2023, 2024]
LEAGUE_TIERS = {
    "GB1": 1, "ES1": 1, "L1": 1, "IT1": 1, "FR1": 1,
    "PO1": 2, "NL1": 2, "TR1": 2, "BE1": 2,
    "GR1": 3, "DK1": 3, "SC1": 3,
}
CUPS = ["CL", "EL"]
TIER_PREMIUM = {1: 0.90, 2: 0.35, 3: 0.0}
POSITIONS = {
    "Attack": ["Centre-Forward", "Left Winger", "Right Winger"],
    "Midfield": ["Central Midfield", "Attacking Midfield", "Defensive Midfield"],
    "Defender": ["Centre-Back", "Left-Back", "Right-Back"],
    "Goalkeeper": ["Goalkeeper"],
}
GOAL_RATE = {"Attack": 0.45, "Midfield": 0.25, "Defender": 0.10, "Goalkeeper": 0.01}
COUNTRIES = [f"Country {i:02d}" for i in range(20)]


def make_fake_data(out_dir, n_players=900, n_clubs=60, seed=7):
    """Write the five CSVs into out_dir and return {filename: path}."""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- clubs and competitions --------------------------------------------
    league_ids = list(LEAGUE_TIERS)
    clubs = pd.DataFrame({
        "club_id": np.arange(1, n_clubs + 1),
        "name": [f"Club {i:02d}" for i in range(1, n_clubs + 1)],
        "domestic_competition_id": [league_ids[i % len(league_ids)] for i in range(n_clubs)],
        "squad_size": rng.integers(18, 33, n_clubs),
        "average_age": rng.normal(25.5, 1.4, n_clubs).round(1),
        "national_team_players": rng.integers(0, 12, n_clubs),
    })
    league_of_club = dict(zip(clubs["club_id"], clubs["domestic_competition_id"]))
    tier_of_club = {club: LEAGUE_TIERS[league] for club, league in league_of_club.items()}

    # NOTE: no is_major_national_league column here, because the real dump does
    # not have one either. features.load_raw() must fall back to config.TOP5.
    competitions = pd.DataFrame({
        "competition_id": league_ids + CUPS,
        "name": [f"League {c}" for c in league_ids] + ["Champions League", "Europa League"],
        "type": ["domestic_league"] * len(league_ids) + ["international_cup"] * len(CUPS),
        "country_name": [f"Country {i:02d}" for i in range(len(league_ids))] + [None, None],
    })

    # --- the players, and the hidden truth about them -----------------------
    n = n_players
    player_id = np.arange(1001, 1001 + n)
    quality = rng.normal(0, 1, n)                      # never written to any file
    club_id = rng.integers(1, n_clubs + 1, n)
    second_club_id = rng.integers(1, n_clubs + 1, n)   # for the winter movers
    moves = rng.random(n) < 0.10

    position = rng.choice(list(POSITIONS), n, p=[0.30, 0.32, 0.30, 0.08])
    sub_position = np.array([rng.choice(POSITIONS[p]) for p in position])
    date_of_birth = pd.Timestamp("1990-01-01") + pd.to_timedelta(
        rng.integers(0, 17 * 365, n), unit="D"
    )
    # 0 cm means "unknown" in this dataset, not a very short footballer
    height = np.where(rng.random(n) < 0.05, 0, rng.normal(182, 6, n).round())
    contract_end = pd.Series(
        pd.Timestamp("2023-06-01") + pd.to_timedelta(rng.integers(-200, 1500, n), unit="D")
    ).mask(rng.random(n) < 0.07)   # and some contracts are simply missing

    players = pd.DataFrame({
        "player_id": player_id,
        "name": [f"Player {i}" for i in player_id],
        "date_of_birth": date_of_birth,
        "position": position,
        "sub_position": sub_position,
        "foot": rng.choice(["right", "left", "both"], n, p=[0.70, 0.25, 0.05]),
        "height_in_cm": height,
        "country_of_citizenship": rng.choice(COUNTRIES, n),
        "current_club_id": club_id,
        "contract_expiration_date": contract_end,
    })

    # --- the fake market ----------------------------------------------------
    # gap = how far today's printed price is from fair value. It halves every
    # six months, which IS the signal the whole project is looking for: a player
    # who is cheap today drifts back up towards what he is worth.
    gap = rng.normal(0, 0.35, n)
    forgotten = rng.random((len(VALUATION_DATES), n)) < 0.08   # the site skips some
    disappears = rng.random(n) < 0.06        # injured / retired / left the dataset
    premium = np.array([TIER_PREMIUM[tier_of_club[c]] for c in club_id])

    valuation_frames = []
    for index, date in enumerate(VALUATION_DATES):
        date = pd.Timestamp(date)
        age = (date - players["date_of_birth"]).dt.days.to_numpy() / 365.25
        log_fair = 13.9 + 1.15 * quality - 0.021 * (age - 25) ** 2 + premium
        value = np.expm1(log_fair + gap)
        value = (np.clip(value, 25_000, None) / 25_000).round() * 25_000

        keep = ~forgotten[index]
        if index >= HORIZON_INDEX:
            keep = keep & ~disappears        # these are the censored players
        valuation_frames.append(pd.DataFrame({
            "player_id": player_id[keep],
            "date": date,
            "market_value_in_eur": value[keep],
            "current_club_id": club_id[keep],
        }))
        gap = 0.5 * gap + rng.normal(0, 0.30, n)

    valuations = pd.concat(valuation_frames, ignore_index=True)

    # The leakage bait. The real players.csv carries these two columns, and
    # merging that table in wholesale gives a model that scores beautifully and
    # knows nothing. assert_no_leakage() exists to catch exactly this.
    players["market_value_in_eur"] = players["player_id"].map(
        valuations.sort_values("date").groupby("player_id")["market_value_in_eur"].last()
    )
    players["highest_market_value_in_eur"] = players["player_id"].map(
        valuations.groupby("player_id")["market_value_in_eur"].max()
    )

    # --- appearances -------------------------------------------------------
    # Performance has to carry `quality`, otherwise there is nothing for the
    # model to learn and the tests would be measuring noise.
    ga_rate = np.clip(
        np.array([GOAL_RATE[p] for p in position]) + 0.22 * quality, 0.01, 1.6
    )
    frames = []
    next_game_id = 1
    for season in SEASONS:
        kickoff = pd.Timestamp(f"{season}-08-05")
        games = rng.integers(8, 31, n)   # some players stay under MIN_MINUTES
        per_game = np.clip(rng.normal(72 + 6 * quality, 13), 20, 90)

        row_player = np.repeat(np.arange(n), games)
        matchday = np.concatenate([np.arange(g) for g in games])
        rows = len(row_player)

        minutes = np.clip(per_game[row_player] + rng.normal(0, 12, rows), 0, 90).round()
        played_for = club_id[row_player].copy()
        switched = moves[row_player] & (matchday >= 0.7 * np.repeat(games, games))
        played_for[switched] = second_club_id[row_player][switched]

        frames.append(pd.DataFrame({
            # one id per row: `games` is counted as nunique(game_id)
            "game_id": np.arange(next_game_id, next_game_id + rows),
            "player_id": player_id[row_player],
            "player_club_id": played_for,
            "competition_id": [league_of_club[c] for c in played_for],
            "date": kickoff + pd.to_timedelta(matchday * 9, unit="D"),
            "minutes_played": minutes,
            "goals": rng.poisson(0.6 * ga_rate[row_player] * minutes / 90),
            "assists": rng.poisson(0.4 * ga_rate[row_player] * minutes / 90),
            "yellow_cards": rng.poisson(0.12 * minutes / 90),
            "red_cards": (rng.random(rows) < 0.004).astype(int),
        }))
        next_game_id += rows

        # A few European nights for half the tier-1 players, so that
        # n_competitions and pct_minutes_major are not the same for everybody.
        tier1 = np.flatnonzero([tier_of_club[c] == 1 for c in club_id])
        european = rng.choice(tier1, size=max(len(tier1) // 2, 1), replace=False)
        cup_player = np.repeat(european, 6)
        cup_rows = len(cup_player)
        cup_minutes = np.clip(
            per_game[cup_player] + rng.normal(0, 15, cup_rows), 0, 90
        ).round()
        frames.append(pd.DataFrame({
            "game_id": np.arange(next_game_id, next_game_id + cup_rows),
            "player_id": player_id[cup_player],
            "player_club_id": club_id[cup_player],
            "competition_id": rng.choice(CUPS, cup_rows),
            "date": kickoff + pd.to_timedelta(rng.integers(30, 260, cup_rows), unit="D"),
            "minutes_played": cup_minutes,
            "goals": rng.poisson(0.5 * ga_rate[cup_player] * cup_minutes / 90),
            "assists": rng.poisson(0.3 * ga_rate[cup_player] * cup_minutes / 90),
            "yellow_cards": rng.poisson(0.12 * cup_minutes / 90),
            "red_cards": np.zeros(cup_rows, dtype=int),
        }))
        next_game_id += cup_rows

    appearances = pd.concat(frames, ignore_index=True)

    # Real dumps are messy: the same id shows up as text with stray spaces in
    # one file and as a float in another. We reproduce that ON PURPOSE. Delete
    # the normalisation in features.load_raw() and several tests must fail -
    # that is exactly what those tests are for.
    appearances["competition_id"] = " " + appearances["competition_id"].astype(str) + " "
    clubs["club_id"] = clubs["club_id"].astype(float)

    tables = {
        "players.csv": players,
        "player_valuations.csv": valuations,
        "appearances.csv": appearances,
        "clubs.csv": clubs,
        "competitions.csv": competitions,
    }
    for filename, frame in tables.items():
        frame.to_csv(out_dir / filename, index=False)
    return {filename: out_dir / filename for filename in tables}


if __name__ == "__main__":
    written = make_fake_data("data/fake")
    for name, path in written.items():
        print(f"{name:<24} -> {path}")
