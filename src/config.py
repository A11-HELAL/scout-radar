"""Paths and constants. Every module reads from here."""

BASE = "/content/drive/MyDrive/scout-radar"
RAW = f"{BASE}/data/raw"
PROC = f"{BASE}/data/processed"
EXP = f"{BASE}/exports"
FIG = f"{BASE}/figures"

# Snapshots - do not change without team agreement
SNAPSHOT_TRAIN = "2023-06-01"
SNAPSHOT_TEST = "2024-06-01"

WINDOW_DAYS = 365
MIN_MINUTES = 900

# Top 5 european leagues
TOP5 = ["GB1", "ES1", "L1", "IT1", "FR1"]

RANDOM_STATE = 42
