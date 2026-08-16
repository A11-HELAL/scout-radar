"""Write the JSON files the website and the report read.

Two files, on purpose:

  undervalued.json       a plain top-level JSON array of EXPORT_SIZE players.
                         This is the file the website reads and its shape is a
                         FROZEN CONTRACT - the front-end was built against
                         these exact field names. Do not rename anything here
                         without telling whoever owns the site.

  undervalued_full.json  an object: the SHORTLIST_SIZE players the backtest
                         actually measured, plus the shortlist rules and the
                         model and backtest numbers, for the report and slides.

Both files are written from the same dataframe in the same call, so they can
never disagree with each other. The website never runs a model - it reads one
small file - which is why the demo cannot break live in front of the examiners.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.backtest import make_shortlist


def _num(value, digits=2):
    """A number the front-end can trust: never NaN, and int when digits is 0.

    json.dumps writes NaN as the bare token NaN. That is not valid JSON and it
    makes JSON.parse throw in the browser, so a missing number becomes null.
    """
    if value is None or pd.isna(value):
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return int(round(value)) if digits == 0 else round(value, digits)


def _text(value):
    """A string or null - never the literal "nan"."""
    if value is None or pd.isna(value):
        return None
    return str(value)


def _at(row, column, default):
    """row.get() that also treats a present-but-missing value as absent."""
    value = row.get(column)
    return default if value is None or pd.isna(value) else value


def _reasons(row):
    """Short human phrases, so a card is not just a number.

    Deliberately hand-written rules rather than SHAP: every phrase here is one
    a team member can explain in the defence without opening a library.
    """
    bits = []
    if _at(row, "ga_p90", 0) >= 0.50:
        bits.append(f"{row['ga_p90']:.2f} goals+assists per 90")
    if _at(row, "minutes", 0) >= 2000:
        bits.append("a season-long starter")
    if _at(row, "age", 99) <= 23:
        bits.append("still under 23")
    if _at(row, "league_tier", 4) <= 2:
        bits.append("already playing in a strong league")
    return bits or ["the model prices him well above his market value"]


def _card(row, rank):
    """One player -> one flat JSON object the UI can render directly."""
    model_value = _num(row["pred_value_eur"], 0)
    # exp(-residual) is how many times more the model thinks he is worth
    gap = _num(np.expm1(-row["residual"]) * 100, 1)
    reasons = _reasons(row)
    return {
        "rank": int(rank),
        "player_id": _num(row["player_id"], 0),
        "name": _text(row.get("name")),
        "age": _num(row.get("age"), 1),
        "position": _text(row.get("position")),
        "sub_position": _text(row.get("sub_position")),
        "club": _text(row.get("club_name")),
        "league": _text(row.get("league_name")),
        "country": _text(row.get("country_of_citizenship")),
        "market_value_eur": _num(row["market_value_in_eur"], 0),
        # the same number under both names: the site was built against
        # predicted_value_eur, the report talks about model_value_eur
        "predicted_value_eur": model_value,
        "model_value_eur": model_value,
        # likewise: gap_pct is the site's name, undervalued_pct is ours
        "gap_pct": gap,
        "undervalued_pct": gap,
        # present-day contract info. Shown to the scout, NOT a model input -
        # see ANACHRONISTIC_FEATURES in config.py.
        "contract_months_left": _num(row.get("contract_months_left"), 1),
        "minutes": _num(row.get("minutes"), 0),
        "games": _num(row.get("games"), 0),
        "goals_p90": _num(row.get("goals_p90"), 2),
        "assists_p90": _num(row.get("assists_p90"), 2),
        "reasons": reasons,
        "reason": ", ".join(reasons),
    }


def _dump(payload):
    """allow_nan=False: fail here rather than ship a file JSON.parse rejects."""
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)


def export_shortlist(scored, out_dir=None, meta=None,
                     web_filename=None, full_filename=None):
    """Write both JSON files.

    Returns a dict:
        web_path   - the file the website reads
        full_path  - the file the report reads
        web        - the EXPORT_SIZE rows shown on the site
        shortlist  - the SHORTLIST_SIZE rows the backtest measures

    make_shortlist sorts by residual, so `shortlist` is literally the head of
    `web`: the top of the published list is the list we quote numbers about.
    """
    out_dir = Path(out_dir or config.EXPORTS)
    out_dir.mkdir(parents=True, exist_ok=True)

    web_list = make_shortlist(scored, size=config.EXPORT_SIZE)
    shortlist = web_list.head(config.SHORTLIST_SIZE)
    cards = [
        _card(row, rank)
        for rank, (_, row) in enumerate(web_list.iterrows(), start=1)
    ]

    web_path = out_dir / (web_filename or config.WEB_JSON)
    web_path.write_text(_dump(cards), encoding="utf-8")

    payload = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "snapshot": str(pd.Timestamp(scored["snapshot"].iloc[0]).date()),
        "rules": {
            "max_age": config.SHORTLIST_MAX_AGE,
            "min_market_value_eur": config.SHORTLIST_MIN_VALUE,
            "min_minutes": config.MIN_MINUTES,
            "measured_size": config.SHORTLIST_SIZE,
            "exported_size": config.EXPORT_SIZE,
        },
        "model": meta or {},
        "players": cards[: config.SHORTLIST_SIZE],
    }
    full_path = out_dir / (full_filename or config.FULL_JSON)
    full_path.write_text(_dump(payload), encoding="utf-8")

    return {
        "web_path": web_path,
        "full_path": full_path,
        "web": web_list,
        "shortlist": shortlist,
    }
