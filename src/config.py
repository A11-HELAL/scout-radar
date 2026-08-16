"""Every constant of the experiment lives here.

Rule for the team: if you want to change a snapshot date, a filter or a feature
list, you change it in THIS file only. Nothing else hardcodes these values.
"""

import os

# --- where the data lives ---------------------------------------------------
# Environment variables let the same code run in Colab, on a laptop and inside
# the tests without anybody editing this file.
BASE = os.environ.get("SCOUT_BASE", "/content/drive/MyDrive/scout-radar")
RAW = os.environ.get("SCOUT_RAW", f"{BASE}/data/raw")
PROCESSED = os.environ.get("SCOUT_PROCESSED", f"{BASE}/data/processed")
EXPORTS = os.environ.get("SCOUT_EXPORTS", f"{BASE}/exports")
FIGURES = os.environ.get("SCOUT_FIGURES", f"{BASE}/reports/figures")

# --- experiment design ------------------------------------------------------
SNAPSHOT_TRAIN = "2023-06-01"    # we learn the pricing model here
SNAPSHOT_TEST = "2024-06-01"     # we check one year later here
WINDOW_DAYS = 365                # performance is read from the year before a snapshot
HORIZON_DAYS = 365               # how far ahead the backtest looks
MIN_MINUTES = 900                # ~10 full matches; drops bench players
MAX_VALUATION_AGE_DAYS = 400     # the target must be a reasonably fresh valuation
MIN_AGE, MAX_AGE = 15, 45        # guards against broken dates of birth
RANDOM_STATE = 42

# --- shortlist rules -------------------------------------------------------
# IMPORTANT: the backtest and the exported JSON both call make_shortlist(), so
# the number we report in the defence is measured on the list the app ships.
SHORTLIST_MAX_AGE = 26
SHORTLIST_MIN_VALUE = 1_000_000   # below this, a log residual is mostly noise
SHORTLIST_SIZE = 20
GROWTH_THRESHOLD = 0.50           # "a hit" = market value grew more than 50%

# --- leagues ---------------------------------------------------------------
TOP5 = ["GB1", "ES1", "L1", "IT1", "FR1"]
LEAGUE_TIERS = {
    "GB1": 1, "ES1": 1, "L1": 1, "IT1": 1, "FR1": 1,
    "PO1": 2, "NL1": 2, "TR1": 2, "BE1": 2, "RU1": 2,
    "GR1": 3, "UKR1": 3, "DK1": 3, "SC1": 3, "A1": 3, "SUI1": 3,
}
DEFAULT_TIER = 4
TOP_N_COUNTRIES = 15              # nationalities kept as their own category

# --- model inputs ----------------------------------------------------------
NUMERIC_FEATURES = [
    "age", "age_sq", "height_in_cm",
    "minutes", "games", "min_per_game",
    "goals_p90", "assists_p90", "ga_p90", "yellows_p90", "reds_p90",
    "n_competitions", "n_clubs", "moved_midseason", "pct_minutes_major",
    "league_tier",
    "contract_months_left", "squad_size", "average_age", "national_team_players",
]
CATEGORICAL_FEATURES = ["position", "sub_position", "foot", "country_grp"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "y_log"

# These four are read from present-day snapshots (clubs.csv and the contract
# column), so they describe the player slightly in the FUTURE of a 2023
# snapshot. We keep them but run_all.py measures how much they matter.
ANACHRONISTIC_FEATURES = [
    "contract_months_left", "squad_size", "average_age", "national_team_players",
]

# A feature whose name contains one of these is almost certainly the answer.
BANNED_SUBSTRINGS = ("value", "price", "fee", "worth", "cost", "highest", "transfer")
