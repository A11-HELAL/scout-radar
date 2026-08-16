"""Run the whole story end to end and print every number the report needs.

    python run_all.py

Order matters: build both snapshots -> check for leakage -> train -> ablation ->
score June 2023 -> look up what really happened by June 2024 -> export -> plot.

Nothing in here computes anything itself; it is the narrator. If a number looks
wrong, the bug is in src/, not in this file.
"""

from pathlib import Path

from src import config
from src.backtest import add_residuals, attach_future_value, run_backtest
from src.export import export_shortlist
from src.features import assert_no_leakage, build_dataset, load_raw
from src.plots import save_all
from src.train import ablation, feature_importance, train_all


def _step(title):
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def main():
    _step("1/7  building the two snapshots")
    raw = load_raw()
    players, valuations, appearances, clubs, competitions = raw
    print(f"raw tables: {len(players):,} players, {len(valuations):,} valuations, "
          f"{len(appearances):,} appearances")

    train = build_dataset(config.SNAPSHOT_TRAIN, raw=raw)
    test = build_dataset(config.SNAPSHOT_TEST, raw=raw)
    print(f"train {config.SNAPSHOT_TRAIN}: {train.shape}")
    print(f"test  {config.SNAPSHOT_TEST}: {test.shape}")

    # Saved so the EDA notebook and the report can reload the exact table this
    # run used, without rebuilding it and without re-downloading anything.
    processed = Path(config.PROCESSED)
    processed.mkdir(parents=True, exist_ok=True)
    try:
        train.to_parquet(processed / "train_2023.parquet", index=False)
        test.to_parquet(processed / "test_2024.parquet", index=False)
        print(f"saved  -> {processed}/train_2023.parquet + test_2024.parquet")
    except ImportError:
        print("skipped the parquet snapshots (pip install pyarrow to enable them)")

    _step("2/7  leakage check")
    corr = assert_no_leakage(train)
    print("strongest correlations between a feature and the target:")
    print(corr.head(8).round(3).to_string())

    _step("3/7  leaderboard  (trained on 2023, scored on 2024)")
    leaderboard, fitted, _ = train_all(train, test)
    print(leaderboard.round(3).to_string(index=False))
    best_name = leaderboard.loc[0, "model"]
    best = fitted[best_name]
    print(f"\nwinner: {best_name}")

    _step("4/7  ablation  (same model, one feature group removed at a time)")
    print(ablation(train, test, model_name=best_name).round(3).to_string(index=False))

    _step("5/7  the backtest  (June 2023 -> June 2024)")
    scored = add_residuals(train, best)
    scored = attach_future_value(scored, valuations)
    followed, by_decile, measured, summary = run_backtest(scored)
    print(by_decile.round(3).to_string())
    print()
    for key, value in summary.items():
        print(f"  {key:<26} {value}")

    _step("6/7  export")
    meta = {
        "best_model": best_name,
        "MAE_log": round(float(leaderboard.loc[0, "MAE_log"]), 4),
        "MedAPE_pct": round(float(leaderboard.loc[0, "MedAPE_pct"]), 1),
        "Spearman": round(float(leaderboard.loc[0, "Spearman"]), 3),
        "backtest": summary,
    }
    exported = export_shortlist(scored, meta=meta)
    print(f"website : {exported['web_path']}   ({len(exported['web'])} players)")
    print(f"report  : {exported['full_path']}   ({len(exported['shortlist'])} players)")
    print("\nthe top of the published list:")
    print(
        exported["shortlist"][
            ["name", "age", "club_name", "market_value_in_eur", "pred_value_eur"]
        ].head(10).round(1).to_string(index=False)
    )

    _step("7/7  figures")
    importance = feature_importance(best, test)
    print(importance.head(10).round(4).to_string(index=False))
    print()
    for path in save_all(followed, by_decile, importance):
        print(f"  {path}")

    _step("copy these straight into the results table in README.md")
    print(f"  best model               {best_name}")
    print(f"  MedAPE                   {meta['MedAPE_pct']}%")
    print(f"  Spearman (ranking)       {meta['Spearman']}")
    print(f"  backtest coverage        {summary['coverage']}"
          f"  ({summary['n_followed']} of {summary['n_recommended']} followed)")
    print(f"  cheapest decile growth   {summary['median_growth_undervalued']} (log)")
    print(f"  dearest decile growth    {summary['median_growth_overvalued']} (log)")
    print(f"  difference, 95% interval {summary['diff_95pct_ci']}")
    print(f"  significance             {summary['p_value_text']}")
    print(f"  precision@{config.SHORTLIST_SIZE} vs baseline    "
          f"{summary['precision_at_k']} vs {summary['baseline_rate']} "
          f"(lift {summary['lift']}x, pool of {summary['eligible_pool_size']})")
    print(f"  shipped vs measured list {summary['shortlist_overlap']} of "
          f"{config.SHORTLIST_SIZE} players in common")
    print()


if __name__ == "__main__":
    main()
