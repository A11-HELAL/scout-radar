# Scout Radar

Productivity-based market value model for football players. It flags
undervalued players and proves the advice with a forward-looking backtest.

## 1. Problem

Budget-limited clubs pay for reputation, not output. Who produces like a
10M player while the market prices them at 2M?

## 2. Data

Kaggle: `davidcariboo/player-scores` (Transfermarkt) - 5 tables, ~1.8M rows.
Raw data is **not** in this repo (see `.gitignore`). Download it from Kaggle
into `data/raw/` on the shared Drive.

## 3. How to run

```bash
pip install -r requirements.txt
```

```python
from src.features import build_dataset, load_raw
from src.train import run_all

raw = load_raw()
train = build_dataset("2023-06-01", raw)
test = build_dataset("2024-06-01", raw)
results, models = run_all(train, test)
```

## 4. Method

- Temporal split: train on the 2023-06 snapshot, test on 2024-06
- No leakage: every market-value-derived feature is banned
- Performance window: only the 12 months before each snapshot
- Product: a large negative residual means the player is undervalued

## 5. Results

_Filled in after the backtest._

## 6. Limitations

- The target is Transfermarkt consensus, not true value
- Shallow stats (no xG) - defenders and keepers are modelled worse
- Survivorship bias in the backtest

## Project layout

```
src/config.py       paths and constants
src/features.py     build_dataset(), load_raw(), feature lists
src/train.py        five models + preprocessing pipeline
src/evaluate.py     MAE / RMSE / R2 / MedAPE
src/backtest.py     residual deciles, Mann-Whitney U, Precision@20
src/export_json.py  exports/undervalued.json for the web app
notebooks/          EDA and experiments
reports/figures/    saved charts
```

## Team

A - data & pipeline | B - EDA & report | C - modelling | D - backtest & web app
