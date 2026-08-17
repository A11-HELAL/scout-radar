# scout-radar

Find footballers whose market price sits below what their on-pitch numbers say
they are worth, then check what actually happened to that price one year later.

## The idea

We freeze the world at **1 June 2023** and use only what was knowable that day:
the previous 365 days of appearances, age, position, league, contract length. A
model learns what a player with those numbers usually costs, and we compare its
prediction with the price the market actually printed. A player the model prices
far above the market is our candidate. Then we fast-forward to **1 June 2024**
and look up the real price.

```
2023-06-01 snapshot  ->  model predicts value  ->  residual = market - model
                                                        |
                                        most negative residual = "undervalued"
                                                        |
2024-06-01 outcome   <-  did those players really grow?
```

## How to run it (Google Colab)

```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.environ['SCOUT_BASE'] = '/content/drive/MyDrive/scout-radar'
os.environ['SCOUT_RAW']  = '/content/drive/MyDrive/scout-radar/data/raw'

!pip -q install -r requirements.txt
!python run_all.py
```

Raw data: the Kaggle dataset `davidcariboo/player-scores`. Put `players.csv`,
`player_valuations.csv`, `appearances.csv`, `clubs.csv` and `competitions.csv`
into `SCOUT_RAW`. Never commit `kaggle.json` - this repository is public and
`.gitignore` blocks it for you.

`run_all.py` prints seven steps: build both snapshots, leakage check,
leaderboard, ablation, backtest, export, figures. About two minutes.

## The files

| File | What it does |
|---|---|
| `src/config.py` | Every number and path in one place. Change thresholds here, nowhere else. |
| `src/features.py` | Loads the CSVs and builds one row per player as of a date. Refuses to continue if a feature leaks the target. |
| `src/train.py` | Preprocessing pipeline, the four models, the leaderboard, the ablation, permutation importance. |
| `src/evaluate.py` | The metrics: `MAE_log`, `RMSE_log`, `R2_log`, `MedAPE_pct`, `Spearman`. |
| `src/backtest.py` | Residuals, deciles, the one-year lookup, bootstrap interval, precision@20 against the baseline. |
| `src/export.py` | Writes `exports/undervalued.json` for the website and `exports/undervalued_full.json` for the report. |
| `src/plots.py` | The four figures for the report. |
| `run_all.py` | The whole pipeline end to end, printing the numbers the report needs. |

Outputs: `exports/*.json` (committed - the website reads them), four PNGs in
`reports/figures/`, and `data/processed/train_2023.parquet` /
`test_2024.parquet` for the EDA.

## The setup

| | |
|---|---|
| Train / test snapshot | 2023-06-01 / 2024-06-01 |
| Form window | previous 365 days of appearances |
| Eligibility | >= 900 minutes in the window, market value > 0 |
| Target | `log1p(market value)` on the snapshot date |
| Models | mean baseline, ridge, random forest, gradient boosting |
| Signal | `residual = actual_log - predicted_log` (negative = cheap) |
| Shortlist | 20 players, age <= 26, value >= EUR 1M |
| Published list | the same ranking, top 100, for the website |
| Outcome horizon | a valuation 300-430 days after the snapshot |

## Results - the run of 16 August 2026

3,971 eligible players in the 2023 snapshot, 4,044 in 2024. Winner:
`gradient_boosting`.

| Metric | Value |
|---|---|
| `MAE_log` | **0.479** (the mean baseline scores 1.260) |
| `MedAPE_pct` | **37.9%** |
| `R2_log` | 0.838 |
| `Spearman` | **0.914** |

| Backtest | Value |
|---|---|
| Followed one year later (`coverage`) | 3,539 / 3,971 = **0.891** |
| Median growth, cheapest decile | **+0.154 log = +17%** |
| Median growth, dearest decile | **-0.318 log = -27%** |
| Difference, 95% bootstrap CI | **(0.288, 0.580)** - does not contain zero |
| Significance | **p < 1e-05** |
| precision@20 (grew by >= 50%) | **0.65** vs a same-pool baseline of 0.29 |
| Lift | **2.24x** (pool of 1,132 players) |

Ablation, dropping one group and retraining: without present-day club info
`MAE_log` 0.588, without on-pitch performance 0.548, without age 0.573. Top
permutation importances: `national_team_players` 0.297, `pct_minutes_major`
0.245, `age_sq` 0.056, `league_tier` 0.051.

## Limits

- The strongest feature, `national_team_players`, describes the player's club
  **in the dump we downloaded**, not in June 2023. Without that whole group the
  model still beats the baseline by a wide margin (`MAE_log` 0.588 vs 1.260),
  and that is the pair to quote.
- Cheap players drift up and expensive ones drift down without any model's help.
  Ranking the same eligible pool by price alone is the control that separates our
  signal from that effect, and it is not run yet.
- Transfermarkt market values are crowd estimates, not transfer fees.
- The ~11% of players we cannot follow are mostly the ones whose careers went
  wrong, so the measured growth is slightly optimistic.

## Who owns what

| Area | Owner | Files |
|---|---|---|
| Data & features | Helal | `config.py`, `features.py`, `export.py` |
| Measurement | teammate 2 | `backtest.py`, `evaluate.py` |
| EDA & report | teammate 3 | notebooks, figures, the write-up |
| Modelling & website | teammate 4 | `train.py`, `exports/undervalued.json` |

Nobody pushes to `main`. Open a branch, then Helal reviews and merges.
