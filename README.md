# scout-radar

**Find footballers whose market price sits below what their on-pitch numbers say
they are worth - then check what actually happened to that price one year later.**

The second half of that sentence is the whole project. Any model can print a
list of "undervalued" players; almost nothing checks itself afterwards. This
repository does, and it reports the number even when the number is unflattering.

---

## The idea in one paragraph

We freeze the world at **1 June 2023**. Using only what was knowable on that
day - the previous 365 days of appearances, the player's age, position, league,
contract length - we train a model to predict the player's market value. The
model learns what a player with those numbers *usually* costs. We then compare
its prediction to the price the market actually printed. A player the model
prices far above the market is our candidate: cheap relative to his own
performance. Finally we fast-forward to **1 June 2024** and look up the real
price. If the cheap group grew more than the expensive group, the idea works. If
it did not, we say so.

```
2023-06-01 snapshot  ->  model predicts value  ->  residual = market - model
                                                        |
                                        most negative residual = "undervalued"
                                                        |
2024-06-01 outcome   <-  did those players really grow?
```

---

## Quick start

### 1. Run the tests first - they need no data at all

```bash
pip install -r requirements.txt
python -m pytest -q
```

The suite generates its own fake dataset with a **known** answer hidden inside
it (`tests/make_fake_data.py`), then checks that the pipeline finds that answer.
If the tests pass, the machinery is sound. If a test fails, the bug is ours.

### 2. Run the real pipeline

```bash
python run_all.py
```

Seven steps, each printing what it did: build both snapshots -> leakage check ->
leaderboard -> ablation -> backtest -> export -> figures. Roughly a few minutes
on Colab.

### 3. In Google Colab

```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.environ['SCOUT_BASE'] = '/content/drive/MyDrive/scout-radar'
os.environ['SCOUT_RAW']  = '/content/drive/MyDrive/scout-radar/data/raw'

!pip -q install -r requirements.txt
!python -m pytest -q
!python run_all.py
```

Raw data: the Kaggle dataset `davidcariboo/player-scores`. Drop
`players.csv`, `player_valuations.csv`, `appearances.csv`, `clubs.csv` and
`competitions.csv` into `SCOUT_RAW`. **Never commit `kaggle.json`** - this
repository is public and `.gitignore` blocks it for you.

---

## The experiment, in numbers

| | |
|---|---|
| Train snapshot | 2023-06-01 |
| Test snapshot | 2024-06-01 |
| Form window | previous 365 days of appearances |
| Eligibility | >= 900 minutes in the window, market value > 0 |
| Target | `log1p(market value)` on the snapshot date |
| Target freshness | the valuation must be <= 400 days old |
| Models | mean baseline, ridge, random forest, gradient boosting |
| Signal | `residual = actual_log - predicted_log` (negative = cheap) |
| Shortlist | 20 players, age <= 26, value >= EUR 1M |
| Published list | the same ranking, top 100, for the website |
| Outcome horizon | a valuation 300-430 days after the snapshot |

---

## What is in each file

| File | What it is responsible for |
|---|---|
| `src/config.py` | Every number and path in one place. Change thresholds here, nowhere else. |
| `src/features.py` | Load the CSVs, normalise the ids, build one row per player as of a date, and refuse to continue if a feature leaks the target. |
| `src/train.py` | Preprocessing pipeline, the four models, the leaderboard, the ablation, permutation importance. |
| `src/evaluate.py` | The metrics: `MAE_log`, `RMSE_log`, `R2_log`, `MedAPE_pct`, `Spearman`. |
| `src/backtest.py` | Residuals, deciles, the one-year lookup, bootstrap confidence interval, precision@20 vs the honest baseline. |
| `src/export.py` | Writes `exports/undervalued.json` (100 cards for the website) and `exports/undervalued_full.json` (20 players + the rules + the model card). |
| `src/plots.py` | Four figures: decile growth, predicted vs actual, feature importance, age curve. |
| `run_all.py` | The whole thing end to end, printing numbers you can paste into the report. |
| `tests/make_fake_data.py` | Fake Transfermarkt CSVs with a signal planted inside, plus deliberately dirty ids. |
| `tests/test_pipeline.py` | 22 tests. Read the names - they are the list of ways this project could be quietly wrong. |
| `notebooks/shap_demo.py` | Optional. One SHAP waterfall for one player, for the slides. Needs `pip install shap`. Nothing else imports it. |
| `CHANGELOG.md` | What changed in the last review round, and why. Read this first if you saw the old code. |

Outputs: `exports/*.json` (committed - the website reads them), four PNGs in the
figures folder, and `data/processed/train_2023.parquet` / `test_2024.parquet`
for whoever is doing the EDA.

---

## The four traps this repo is built to avoid

**1. Leakage.** `players.csv` carries `market_value_in_eur` and
`highest_market_value_in_eur` - today's answer, sitting next to the 2023
question. Merge that table in wholesale and you get a beautiful score and a
worthless model. `assert_no_leakage()` blocks any feature whose name contains a
banned word, and any feature correlated above 0.98 with the target. It raises;
it does not warn.

**2. Survivorship.** Players who fall out of the dataset are exactly the ones
whose value collapsed. If we silently drop them, the backtest measures the
survivors and looks great. Instead they are marked `censored`, they stay in the
table, and `coverage` is printed so the reader knows how much of the shortlist
we could actually follow.

**3. A flattering baseline.** "15% of our picks doubled" means nothing without
"and 10% of comparable players doubled anyway". The baseline is computed on the
**same eligible pool** the shortlist is drawn from (same age cap, same value
floor), not on the whole dataset.

**4. Measuring a different list than we publish.** The 20 players we quote
numbers about are now provably the first 20 of the 100 we publish - one
function, `make_shortlist()`, builds both, and a test asserts it.

---

## The website contract (frozen - do not break it)

`exports/undervalued.json` is a **top-level JSON array** of 100 objects, ranked,
with these fields:

```json
[
  {
    "rank": 1,
    "player_id": 28003,
    "name": "...",
    "age": 24.3,
    "position": "Attack",
    "club": "...",
    "league": "...",
    "market_value_eur": 12000000,
    "predicted_value_eur": 21500000,
    "gap_pct": 79.2,
    "contract_months_left": 18.0,
    "minutes": 2450,
    "reasons": ["...", "..."]
  }
]
```

Rules: no `NaN` anywhere (missing numbers are `null`), UTF-8, and the field
names above stay as they are. If you need a new field, **add** it - never rename
or remove one, because the front-end is already reading these. The richer report
file `undervalued_full.json` is where new structure belongs.

---

## Results - fill these in from your run

| Metric | Value |
|---|---|
| Best model | |
| `MAE_log` (test snapshot) | |
| `MedAPE_pct` | |
| `Spearman` | |
| Shortlist followed / recommended (`coverage`) | |
| Median 1-year growth, undervalued decile | |
| Median 1-year growth, overvalued decile | |
| Difference, 95% bootstrap CI | |
| p-value | |
| precision@20 vs baseline (lift) | |

`run_all.py` prints this block at the end, already labelled. Copy it straight in.

---

## What we honestly cannot claim

- **Market value is not a transfer fee.** Transfermarkt values are crowd
  estimates. We are modelling a crowd's opinion, not money that changed hands.
- **We are partly predicting the same crowd we want to beat.** "Undervalued"
  here means "cheap relative to what this crowd usually pays for these numbers".
- **The residuals that build the shortlist are in-sample.** The model that
  scores June 2023 was trained on June 2023. Out-of-fold predictions
  (`cross_val_predict`) would be stricter; we chose the simpler version and are
  saying so out loud rather than hiding it.
- **`competitions.csv` has no `is_major_national_league` column.** "Major league"
  is *our* definition - the five leagues listed in `config.TOP5`. It is a
  judgement call, not a fact from the data.
- **Censoring is not random.** The ~13% we cannot follow are disproportionately
  the players whose careers went wrong. Our measured growth is therefore
  slightly optimistic.
- **The EUR 1M value floor is a business choice, not a statistical one.** It
  keeps unfollowable third-tier prospects out of the shortlist. It needs one
  written sentence of justification in the report.
- **One snapshot pair is one measurement, not a track record.** 2023 -> 2024 was
  one particular market. Repeating it for 2021 -> 2022 and 2022 -> 2023 is the
  obvious next step.
- **No injuries, wages, contract clauses, or minutes weighted by opponent
  strength.** All of them matter. None of them are in this dataset.

---

## Who owns what

| Area | Owner | Files |
|---|---|---|
| Data & features | Helal | `config.py`, `features.py`, `export.py` |
| Measurement | teammate 2 | `backtest.py`, `evaluate.py` |
| EDA & report | teammate 3 | notebooks, figures, the write-up |
| Modelling | teammate 4 | `train.py` experiments, the ablation table |
| Website | teammate 5 | reads `exports/undervalued.json` |

If you touch a file you do not own, say so in the pull request. If you change a
number in `config.py`, run `pytest` before you push - several tests read those
numbers directly and will tell you what you broke.

## Definition of done

1. `pytest -q` is green.
2. `run_all.py` finishes and prints the results block.
3. The results table above is filled in with the real numbers.
4. `exports/undervalued.json` is committed and the site renders it.
5. Every person can explain, out loud, one trap from the list above and where in
   the code it is handled.
