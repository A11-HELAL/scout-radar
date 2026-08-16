"""Write the JSON that the front-end reads.

The website never runs a model. It reads one small file. That is why the demo
cannot break live in front of the examiners.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.backtest import make_shortlist


def _reason(row):
    """One short human sentence per player, so the card is not just a number."""
    bits = []
    if pd.notna(row.get("ga_p90")) and row["ga_p90"] >= 0.50:
        bits.append(f"{row['ga_p90']:.2f} goals+assists per 90")
    if row.get("minutes", 0) >= 2000:
        bits.append("a season-long starter")
    if row.get("age", 99) <= 23:
        bits.append("still under 23")
    if row.get("league_tier", 4) <= 2:
        bits.append("already playing in a strong league")
    return ", ".join(bits) or "the model prices him well above his market value"


def _to_card(row):
    """One player -> one flat JSON object the UI can render directly."""
    return {
        "player_id": int(row["player_id"]),
        "name": row["name"],
        "age": round(float(row["age"]), 1),
        "position": row.get("position"),
        "sub_position": row.get("sub_position"),
        "club": row.get("club_name"),
        "league": row.get("league_name"),
        "country": row.get("country_of_citizenship"),
        "market_value_eur": int(row["market_value_in_eur"]),
        "model_value_eur": int(round(float(row["pred_value_eur"]))),
        # exp(-residual) is how many times the model thinks he is worth more
        "undervalued_pct": round(float(np.expm1(-row["residual"]) * 100), 1),
        "minutes": int(row["minutes"]),
        "games": int(row["games"]),
        "goals_p90": round(float(row["goals_p90"]), 2),
        "assists_p90": round(float(row["assists_p90"]), 2),
        "reason": _reason(row),
    }


def export_shortlist(scored, out_dir=None, filename="undervalued.json", meta=None):
    """Save the shortlist as JSON and return (path, shortlist_dataframe)."""
    out_dir = Path(out_dir or config.EXPORTS)
    out_dir.mkdir(parents=True, exist_ok=True)

    shortlist = make_shortlist(scored)
    payload = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "snapshot": str(pd.Timestamp(scored["snapshot"].iloc[0]).date()),
        "rules": {
            "max_age": config.SHORTLIST_MAX_AGE,
            "min_market_value_eur": config.SHORTLIST_MIN_VALUE,
            "min_minutes": config.MIN_MINUTES,
            "size": config.SHORTLIST_SIZE,
        },
        "model": meta or {},
        "players": [_to_card(row) for _, row in shortlist.iterrows()],
    }

    path = out_dir / filename
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path, shortlist
