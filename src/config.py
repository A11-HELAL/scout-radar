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
MIN_MINUTES = 900                # ~10 full matches; drops bench players
MIN_AGE, MAX_AGE = 15, 45        # guards against broken dates of birth
RANDOM_STATE = 42

# The price we LEARN has to be a reasonably fresh valuation. Without this, a
# 2015 price tag gets glued onto a 2023 row for every player the site stopped
# re-valuing.
TARGET_MAX_VALUATION_AGE_DAYS = 400

# --- the one-year horizon ---------------------------------------------------
# Careful: "one year later" is a WINDOW, not "the last valuation recorded
# before the target date". A price refreshed ten days after the snapshot is
# not a one-year outcome and must never be counted as one. These three
# constants used to be folded into MAX_VALUATION_AGE_DAYS, which quietly made
# one number mean two different things.
HORIZON_DAYS = 365               # the date the backtest aims at
HORIZON_MIN_DAYS = 300           # earliest valuation accepted as "a year later"
HORIZON_MAX_DAYS = 430           # latest valuation accepted

# --- shortlist rules -------------------------------------------------------
# Ranking on the raw residual alone hands you 34-year-olds worth 200k, because
# on a log scale a cheap player is easy to be very wrong about. The age cap and
# the value floor are what turn a residual into a shortlist.
SHORTLIST_MAX_AGE = 26
SHORTLIST_MIN_VALUE = 1_000_000   # below this, a log residual is mostly noise
SHORTLIST_SIZE = 20               # the list the backtest MEASURES
EXPORT_SIZE = 100                 # the list the website SHOWS - filters need rows
GROWTH_THRESHOLD = 0.50           # "a hit" = market value grew more than 50%

# --- export files ----------------------------------------------------------
# WEB_JSON is a frozen contract: a plain JSON array with the field names the
# front-end was built against. Anything new belongs in FULL_JSON instead, so
# the website never breaks because the pipeline changed.
WEB_JSON = "undervalued.json"
FULL_JSON = "undervalued_full.json"

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
# snapshot. We keep them, and the ablation below measures how much they matter.
ANACHRONISTIC_FEATURES = [
    "contract_months_left", "squad_size", "average_age", "national_team_players",
]

# --- ablation --------------------------------------------------------------
# Three groups, one model. Every group costs one extra training run, so keep
# this list short: the question is "does the result depend on this?", not
# "what happens for every possible combination?".
ABLATION_GROUPS = {
    "present-day club info": ANACHRONISTIC_FEATURES,
    "on-pitch performance": [
        "minutes", "games", "min_per_game",
        "goals_p90", "assists_p90", "ga_p90", "yellows_p90", "reds_p90",
    ],
    "age": ["age", "age_sq"],
}

# A feature whose name contains one of these is almost certainly the answer.
BANNED_SUBSTRINGS = ("value", "price", "fee", "worth", "cost", "highest", "transfer")
