# scout-radar

Ranks football players by how likely their market value is to rise over the next year, using only their playing record from the twelve months before the prediction date.

> ## Status: under review — not finished
>
> This started as a proposed graduation project. The team I pitched it to did not take it up, so it was never built as a group project. There are no other contributors.
>
> Most of this code was written with AI assistance. I understand the data preparation, the merge logic and the design of the experiment. I have not yet checked the modelling and measurement code line by line, and I am doing that now.
>
> **The control experiment that the headline result depends on has not been run yet.** Until it is, the lift figure below is provisional. See [What is missing](#what-is-missing).
>
> I would rather publish this with the gap stated than describe it as complete.

---

## The question

A club scouting a player wants to know more than what he is worth today. It wants to know whether he is about to become more expensive.

So the question here is not "what is this player's market value?" — the data already answers that. It is: **given how a player performed last season, is his valuation about to move up?**

## How the experiment is set up

Freeze the world at **1 June 2023**. Look only backwards.

1. Build features from each player's previous 365 days: minutes played, share of minutes in a major competition, national-team presence, age, league tier, club context.
2. Predict `log1p(market value)` at the freeze date.
3. Take `residual = actual_log − predicted_log`. A negative residual means the market prices him below what his record suggests.
4. Walk forward to **1 June 2024** and see what actually happened to his valuation.

Eligibility: at least 900 minutes played and a recorded value above zero. That leaves 3,971 players in 2023 and 4,044 in 2024.

Nothing after the freeze date is allowed into the features. `assert_no_leakage()` in `src/features.py` raises if any feature correlates above 0.98 with the target, and separately rejects any feature whose name contains a banned substring.

## Data preparation

Four guards, because the joins are where this kind of project breaks quietly rather than loudly:

- `safe_merge()` raises if a join increases the row count. A one-to-many fan-out would otherwise inflate the dataset with no error at all.
- `_int_key()` casts every join key to nullable `Int64`. Player IDs arriving as `12`, `12.0` and `"12"` from different files will not match otherwise.
- A check raises if fewer than half the rows matched a club, rather than letting the model train on club features that are mostly missing.
- `_target()` rejects valuations older than a set age, so a stale price is never used as the answer.

## Results

Model selection, on the 2023 snapshot:

| | Value |
| --- | --- |
| Winner | `gradient_boosting` |
| MAE (log) | **0.479** |
| MAE (log), predict-the-mean baseline | 1.260 |
| Median absolute % error | 37.9% |
| R² (log) | 0.838 |
| Spearman correlation | 0.914 |

Forward test, 2023 → 2024:

| | Value |
| --- | --- |
| Coverage | 0.891 — 3,539 of 3,971 players traceable |
| Cheapest decile by residual | **+0.154 log ≈ +17% value** |
| Dearest decile by residual | **−0.318 log ≈ −27% value** |
| Difference, 95% bootstrap CI | (0.288, 0.580) |
| Mann–Whitney p | < 1e-05 |
| Precision@20 | 0.65 |
| Precision@20, pool base rate | 0.29 |
| Lift | **2.24×** — pool of 1,132 |

Ablations — drop a feature group, retrain, watch the error:

| Removed | MAE (log) |
| --- | --- |
| nothing | 0.479 |
| club information | 0.588 |
| on-pitch record | 0.548 |
| age | 0.573 |

Permutation importance is led by `national_team_players` (0.297) and `pct_minutes_major` (0.245). Age (`age_sq`, 0.056) and `league_tier` (0.051) matter far less than I expected going in.

## What is missing

**The price-only control has not been run.** This is the single thing standing between "interesting" and "demonstrated".

Cheap players tend to rise and expensive players tend to fall on their own, with no model involved. That is mean reversion, and it would produce a result that looks a great deal like the one above. Separating the model's signal from that effect means ranking the same eligible pool by **price alone** and measuring precision@20 the same way.

`eligible_pool()` and the pool base rate already exist in `src/backtest.py`. What is missing is roughly fifteen lines: take `nsmallest(20, "market_value_in_eur")` from the same pool, measure precision identically, and print that number next to the 0.65.

Until that runs, **the 2.24× lift is not proven**. It may survive the control. It may not.

## Other limits

- 11% of the 2023 pool could not be traced to a 2024 valuation. They are flagged `censored` rather than silently dropped, but players who disappear from the data are unlikely to be the ones who got more expensive — so the coverage loss probably makes this result look better than it is.
- Market value is an estimate, not a transfer fee. The model is predicting an opinion about a player, not a price anyone paid.
- The horizon is fixed at 300–430 days. Nothing here says anything about shorter windows.
- One freeze date. A single year can simply be a good year for a signal.

## Configuration

Every threshold lives in `src/config.py`: 365-day feature window, eligibility at ≥900 minutes and value >0, target `log1p(market value)`, shortlist of 20 filtered to age ≤26 and value ≥ EUR 1M, outcome horizon 300–430 days.

## Data

Kaggle: `davidcariboo/player-scores`.

## Running it

```bash
python run_all.py
```

| File | Does |
| --- | --- |
| `src/config.py` | Every threshold, in one place |
| `src/features.py` | Merges, cleaning, feature build, leakage check |
| `src/train.py` | Model comparison |
| `src/evaluate.py` | Metrics |
| `src/backtest.py` | Forward test, bootstrap, precision@k |
| `src/export.py` | Shortlist output |
| `src/plots.py` | Figures |
