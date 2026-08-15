"""Export the top-100 shortlist for the Lovable web app."""
import numpy as np

from src.config import EXP

COLS = [
    "rank",
    "player_id",
    "name",
    "club",
    "league",
    "position",
    "age",
    "minutes",
    "goals_p90",
    "contract_months_left",
    "market_value_eur",
    "predicted_value_eur",
    "gap_pct",
    "residual",
]


def export_shortlist(scored, path=f"{EXP}/undervalued.json", max_age=26, n=100):
    out = (
        scored.query("age <= @max_age")
        .nsmallest(n, "residual")
        .reset_index(drop=True)
        .assign(
            predicted_value_eur=lambda d: np.expm1(d["pred"]).round(0),
            market_value_eur=lambda d: d["market_value_in_eur"],
            club=lambda d: d["current_club_name"],
            league=lambda d: d["country_name"],
            age=lambda d: d["age"].round(1),
            gap_pct=lambda d: (
                (np.expm1(d["pred"]) / d["market_value_in_eur"] - 1) * 100
            ).round(1),
            residual=lambda d: d["residual"].round(3),
        )
    )
    out["rank"] = out.index + 1
    out[COLS].to_json(path, orient="records", force_ascii=False, indent=2)
    print(f"Exported {len(out)} players -> {path}")
    return out[COLS]
