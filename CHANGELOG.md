# Changelog

Read this first if you knew the old code. Newest round at the top.

---

## Round 2 - review round (this branch)

We reviewed the whole pipeline line by line, kept everything that was right, and
changed only what was measurably wrong. **Three real bugs were already fixed in
round 1 and are kept:** the shortlist/measured-list mismatch, the inflated
baseline, and the survivorship bias in the one-year lookup.

### Fixed - things that were silently wrong

**1. The joins were matching nothing, and nothing complained.**
Real CSV dumps hand you the same id as `" GB1 "` in one file and `1.0` in
another. `load_raw()` now forces every id column to one type before any merge
(`player_id`, `club_id`, `competition_id`, `domestic_competition_id`), and
`build_dataset()` raises if more than half of `club_name` came back empty.
Without this, `league_tier` and `pct_minutes_major` were quietly blank for
everybody and the model was training on missing values.
*Files: `src/features.py`*

**2. `is_major_national_league` does not exist.**
The old code preferred a column from `competitions.csv` that the Kaggle dump
simply does not have. The fallback to `config.TOP5` was doing all the work all
along. The comment now says so, the fake data no longer invents the column, and
a test asserts the fallback is the path that runs. "Major league" is *our*
definition, and the README says that out loud.
*Files: `src/features.py`, `tests/make_fake_data.py`, `tests/test_pipeline.py`*

**3. "One year later" meant "any time later".**
The outcome lookup accepted any valuation after the snapshot, so one player was
measured over 13 months and another over 3. The window is now explicit -
`HORIZON_MIN_DAYS = 300` to `HORIZON_MAX_DAYS = 430` - and when several
valuations qualify we take the one closest to exactly 365 days. Expect `coverage`
to **drop** (fewer players have a price inside a tight window). That is the cost
of the number meaning what it says.
*Files: `src/config.py`, `src/backtest.py`*

**4. `MedAPE` was a fraction being read as a percent.**
`0.38` was going into slides as "0.38% error". Renamed to `MedAPE_pct` and
multiplied by 100, with a test that fails if it ever drops below 1 again.
*Files: `src/evaluate.py`, `src/train.py`, `run_all.py`*

**5. `p_value: 0.0` is not a p-value.**
It is "smaller than this machine can print". The summary now carries a
`p_value_text` that reads `p < 1e-05`, and that is what goes in the report.
*Files: `src/backtest.py`, `run_all.py`*

**6. `NaN` in the exported JSON.**
`json.dumps` writes a bare `NaN` token, and `JSON.parse` in the browser throws
on it. Export now uses `allow_nan=False` and converts every missing number to
`null`. A test parses both files and rejects any invalid constant.
*Files: `src/export.py`, `tests/test_pipeline.py`*

**7. `MAX_VALUATION_AGE_DAYS` was ambiguous.**
Renamed to `TARGET_MAX_VALUATION_AGE_DAYS`, because it is about how stale the
*label* may be - nothing to do with the outcome horizon. Two different ideas were
sharing one name.
*Files: `src/config.py`, `src/features.py`, `tests/test_pipeline.py`*

### Changed - decisions we took on purpose

**Two export files instead of one.**
`exports/undervalued.json` stays exactly the shape the website already reads: a
top-level array, old field names intact, now 100 ranked players. The new
`exports/undervalued_full.json` carries the 20-player shortlist plus the rules
and the model card, and that is where new structure goes from now on. Renaming a
field in the first file breaks a teammate's page; adding to the second one
breaks nothing.
*Files: `src/export.py`, `src/config.py`*

**The ablation stays, but smaller.**
Best model only, three groups (present-day club info, on-pitch performance,
age) - four fits instead of twenty. It answers the only question the examiner
will ask ("which features actually matter?") without the runtime.
*Files: `src/train.py`, `src/config.py`*

**`pyarrow` is back, `shap` is out.**
`run_all.py` saves `data/processed/train_2023.parquet` and `test_2024.parquet`
again, so the EDA notebook does not rebuild the dataset from scratch. SHAP moved
to `notebooks/shap_demo.py`: optional, standalone, imported by nothing. A broken
`shap` install can no longer take the pipeline down the night before the defence.
*Files: `requirements.txt`, `run_all.py`, `notebooks/shap_demo.py`*

**Figure filenames lost their number prefixes.**
Now `decile_growth.png`, `pred_vs_actual.png`, `feature_importance.png`,
`age_curve.png`. If your slides link the old `01_*.png` names, update them.
*Files: `src/plots.py`*

### Added

- `run_all.py` - seven numbered steps, and a final block of numbers formatted to
  be pasted straight into the report table.
- `src/plots.py` - the four figures, `Agg` backend so it works headless.
- `tests/test_pipeline.py` - 22 tests, one per way this project could be quietly
  wrong. Deliberately dirty ids in the fake data, so the id normalisation is
  actually exercised.
- `README.md` - rewritten as the single entry point: the idea, the quick start,
  what every file does, the four traps, the frozen website contract, and an
  honest list of what we cannot claim.
- `.gitignore` - secrets block first. This repository is public; `kaggle.json`
  must never land in it.

### Removed

- `src/export_json.py` - replaced by `src/export.py`. Nothing imports the old
  name any more.

### Known and accepted

- The residuals that build the shortlist are **in-sample** (the model that
  scores June 2023 was trained on June 2023). `cross_val_predict` would be
  stricter. We chose the simpler version and wrote it in the README instead of
  hiding it.
- The EUR 1M value floor is a business choice. It still needs one written
  sentence of justification in the report.
- One snapshot pair is one measurement. 2021 -> 2022 and 2022 -> 2023 are the
  obvious next runs.

---

## Round 1 - first working pipeline

First end-to-end version: snapshot builder, four models, residual deciles,
one-year backtest, JSON export for the website, first test suite. Fixed the
three bugs listed at the top of this file.
