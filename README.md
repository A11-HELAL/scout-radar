# Scout Radar

Finding footballers whose market value looks lower than their performance
deserves - and then checking, on real history, whether the market later agreed.

## The one-paragraph version

We train a model to predict a player's market value from his performance only
(minutes, goals, assists, cards, age, league, club context). The model is a
**pricing model**, not a crystal ball. Wherever the real market value sits far
*below* the model's price, the player is a candidate. Then we go back to June
2023, take the 20 players the model called cheapest, and look up what actually
happened to their value by June 2024. That last step is the whole project - a
model with a good MAE and no backtest proves nothing.

```
residual = log(market value)  -  log(model's value)
very negative residual  ->  the market is asleep  ->  shortlist
```

## Quick start

```bash
pip install -r requirements.txt

# 1) tests first - they run on generated data in a few seconds, no download
python -m pytest -q

# 2) the real thing (expects the Kaggle CSVs in data/raw)
python run_all.py
```

In Colab, point the code at Drive instead of editing any file:

```python
import os
os.environ["SCOUT_BASE"] = "/content/drive/MyDrive/scout-radar"
```

Data: [davidcariboo/player-scores](https://www.kaggle.com/datasets/davidcariboo/player-scores)
- needed tables: `players.csv`, `player_valuations.csv`, `appearances.csv`,
  `clubs.csv`, `competitions.csv` -> put them in `data/raw/`.

## How the experiment is set up

| | |
|---|---|
| Train snapshot | **2023-06-01** - features from the 12 months before it |
| Test snapshot | **2024-06-01** - never touched while training |
| Target | `log1p(market_value_in_eur)` at the snapshot |
| Who is included | 900+ minutes in the window, and a valuation refreshed within 400 days |
| Shortlist rules | age <= 26, market value >= EUR 1m, top 20 by residual |
| A "hit" | market value grew more than 50% in the following year |

The train/test split is **by time, not random**. A random split would let a
player's 2024 row teach the model about his own 2023 row.

## What is in each file

| File | Job |
|---|---|
| `src/config.py` | every constant, path and feature list - the only file you edit to change the experiment |
| `src/features.py` | the five raw tables -> one clean row per player, plus `assert_no_leakage` |
| `src/train.py` | preprocessing pipeline, four models, leaderboard, ablation, importances |
| `src/evaluate.py` | MAE / RMSE / R2 in log space, MedAPE in euros, Spearman for ranking |
| `src/backtest.py` | residuals, the future lookup, the shortlist rule, the statistics |
| `src/export.py` | the JSON the website reads |
| `src/plots.py` | the four figures for the report |
| `run_all.py` | runs the whole story end to end and prints every number |
| `tests/make_fake_data.py` | generates Transfermarkt-shaped data with a signal planted in it |
| `tests/test_pipeline.py` | 16 tests over that data |

## The four traps this code is built to avoid

1. **Leakage.** `players.csv` contains `market_value_in_eur` and
   `highest_market_value_in_eur`. Merging the whole table in gives a model that
   scores beautifully and knows nothing. We select columns explicitly, and
   `assert_no_leakage()` refuses to continue if any feature is named like the
   target or correlates above 0.98 with it.
2. **Time travel.** Every performance number is aggregated strictly inside
   `(snapshot - 365 days, snapshot]`, and the club we describe a player with is
   the club he played the most minutes for in that window - not whichever club
   happens to sit on a later row.
3. **Survivorship in the backtest.** The future value is looked up in the raw
   `player_valuations` table, not in the 2024 feature table. Using the feature
   table would quietly keep only the players who were still fit, still playing
   and still being valued a year later. Players we genuinely cannot follow are
   marked `censored` and reported as coverage instead of vanishing.
4. **A metric that measures a different list than the product ships.** The
   backtest and the JSON export both call the same `make_shortlist()`, and
   `precision_at_k` is compared against a baseline computed on the *same*
   eligible pool. Comparing an under-26 shortlist against the whole population
   would manufacture a lift out of the age filter alone.

## Honest limitations

- **The target is an estimate, not a price.** Transfermarkt market values are
  crowd-sourced. We are modelling a community's opinion, not transfer fees.
- **`residual` is not "mispricing".** It is `mispricing + model error`, and we
  cannot separate them. A player can be flagged simply because the model has
  never seen anyone like him.
- **Four features are anachronistic.** `squad_size`, `average_age`,
  `national_team_players` and `contract_months_left` come from present-day
  snapshots, so at a 2023 row they know a little about the future. `run_all.py`
  retrains without them and prints the difference, so we can say out loud how
  much of the result depends on them.
- **One snapshot pair is one experiment.** 2023 -> 2024 is a single draw. The
  bootstrap interval in the summary shows how wide the honest range is.
- **No injuries, no tactical data, no scouting eye.** A model that only sees
  minutes and goals will call an injured player cheap.

## Results

Fill this in from the `run_all.py` output before the defence:

| | value |
|---|---|
| Best model | _ |
| MedAPE (euros) | _ |
| Spearman (ranking) | _ |
| Backtest coverage | _ |
| Median growth, cheapest decile | _ |
| Median growth, dearest decile | _ |
| Difference, 95% interval | _ |
| precision@20 vs baseline | _ |

## Team workflow

Branch per person, pull request into `main`, no direct pushes. Anything
generated (`data/`, `reports/figures/`) stays out of git; `exports/` is
committed because the website reads it.
