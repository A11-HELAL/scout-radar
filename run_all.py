"""Run the whole project with one command:  python run_all.py

Every number that ends up in the report or on a slide is printed by this file,
in the order the story is told. Nothing is computed in a notebook and copied by
hand - that is how numbers stop matching between the slides and the code.
"""

import pandas as pd

from src import config
from src.backtest import add_residuals, attach_future_value, run_backtest
from src.export import export_shortlist
from src.features import assert_no_leakage, build_dataset, load_raw
from src.train import ablation, feature_importance, train_all

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)


def banner(text):
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def main(make_figures=True):
    banner("1. loading the raw tables")
    raw = load_raw()
    players, valuations, appearances, clubs, competitions = raw
    print(f"players {len(players):,} | valuations {len(valuations):,} | "
          f"appearances {len(appearances):,} | clubs {len(clubs):,}")

    banner("2. building the two snapshots")
    train = build_dataset(config.SNAPSHOT_TRAIN, raw)
    test = build_dataset(config.SNAPSHOT_TEST, raw)
    print(f"train {config.SNAPSHOT_TRAIN}: {train.shape[0]:,} players")
    print(f"test  {config.SNAPSHOT_TEST}: {test.shape[0]:,} players")
    assert train["player_id"].is_unique, "duplicate players in train"
    assert train["snapshot"].max() <= pd.Timestamp(config.SNAPSHOT_TRAIN)

    banner("3. leakage check")
    correlations = assert_no_leakage(train)
    print("strongest correlations with the target (all must be well under 0.98):")
    print(correlations.head(6).round(3).to_string())

    banner("4. model leaderboard (trained on 2023, scored on 2024)")
    leaderboard, fitted, best = train_all(train, test)
    print(leaderboard.round(4).to_string())
    print(f"\nwinner: {best}")

    banner("5. ablation - do the present-day club columns carry the result?")
    print(ablation(train, test).round(4).to_string(index=False))

    banner("6. backtest")
    scored = add_residuals(train, fitted[best])
    scored = attach_future_value(scored, valuations)
    followed, by_decile, shortlist, summary = run_backtest(scored)
    print(by_decile.round(3).to_string())
    print()
    for key, value in summary.items():
        print(f"{key:28s} {value}")

    banner("7. the shortlist we actually recommend")
    columns = ["name", "age", "club_name", "league_name", "minutes",
               "market_value_in_eur", "pred_value_eur", "residual", "growth_pct"]
    print(shortlist[columns].round(2).to_string(index=False))

    banner("8. exporting for the website")
    meta = {
        "best_model": best,
        "MAE_log": round(float(leaderboard.loc[best, "MAE_log"]), 4),
        "MedAPE_eur": round(float(leaderboard.loc[best, "MedAPE_eur"]), 4),
        "Spearman": round(float(leaderboard.loc[best, "Spearman"]), 4),
        "backtest": summary,
    }
    path, _ = export_shortlist(scored, meta=meta)
    print(f"wrote {path}")

    if make_figures:
        banner("9. figures")
        from src.plots import save_all
        for figure in save_all(scored, by_decile, feature_importance(fitted[best])):
            print(f"wrote {figure}")

    print("\ndone.")
    return {"train": train, "test": test, "leaderboard": leaderboard,
            "scored": scored, "shortlist": shortlist, "summary": summary}


if __name__ == "__main__":
    main()
