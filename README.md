# Premier League 2026/27 Prediction Model

A from-scratch, dynamically-updating model for predicting the PL 26/27 table,
built while learning Python/data science along the way.

## Approach

Statistical model, not black-box ML (at least to start):

1. **Elo-style power ratings** for every team, updated after each match based
   on result vs. expectation (with home advantage baked in).
2. **Poisson goal model** converts ratings into expected goals per side, giving
   full score-line probabilities (not just W/D/L).
3. **Monte Carlo season simulation**: simulate all remaining fixtures thousands
   of times using those probabilities to get a distribution over final
   standings (title odds, top-4 odds, relegation odds).
4. **Weekly update loop**: after each real gameweek, feed in results, ratings
   shift, re-simulate the rest of the season.
5. **Team news adjustments** (v2): manual nudges to a team's rating for
   significant injuries/suspensions/manager changes.

## Why this approach

Elo/Poisson is how ClubElo and (originally) FiveThirtyEight's SPI work. It's
transparent (you can always explain *why* the model favors a team), doesn't
need exotic data, and is a strong baseline — any fancier ML model later has to
beat it to be worth the added complexity.

## Project structure

```
pl-predictor/
├── data/
│   ├── raw/          # downloaded CSVs, untouched (season results from football-data.co.uk)
│   └── processed/    # cleaned data we've transformed
├── src/               # pipeline code (data loading, elo, poisson, simulation)
├── notebooks/         # exploratory / learning notebooks
├── ratings/           # current team rating state (JSON), updated weekly
└── tests/
```

## Data source

[football-data.co.uk](https://www.football-data.co.uk/englandm.php) — free
historical match results (scores, stats, odds) for English football back to
the 1990s. `data/raw/E0_<season>.csv` where `E0` = Premier League and season
is e.g. `2627` for 2026/27. Refreshed weekly by re-downloading the current
season's file.

## Build log / phases

- [x] Phase 0 — project setup, venv, historical data pulled (2022/23 – 2026/27 so far)
- [x] Phase 1 — data loading & cleaning (`src/data_load.py`)
- [x] Phase 2 — Elo rating engine (`src/elo.py`), backtested at 68.1% accuracy on decisive matches vs 58.6% home-favorite baseline (`src/backtest.py`)
- [x] Phase 3 — Poisson goal model (`src/poisson_model.py`), full scoreline probabilities per team strength; 51.2% in-sample 3-way accuracy vs 42.4% baseline
- [x] Phase 4 — season Monte Carlo simulator (`src/simulate.py`), 10,000-sim title/top-4/relegation odds; caught and fixed a small-sample overfitting bug via shrinkage
- [x] Phase 5 — weekly update pipeline (`src/update_weekly.py`), backed by a validated full-season schedule (`src/parse_schedule.py` -> `data/raw/schedule_2026_27.csv`); re-downloads results, flags possible postponements, recomputes everything, logs a snapshot history, predicts the next gameweek
- [x] Phase 6 — manual team-news adjustments (`src/team_news.py`, `data/team_news.csv`); date-limited injury/suspension multipliers that apply only to fixtures in their window and auto-expire
- [x] Phase 7 — web dashboard published as an Artifact: https://claude.ai/code/artifact/9ad3c12a-43af-4c19-81fe-c091e8aad30d (`export_dashboard.py` -> `build_dashboard.py` -> republish `dashboard.html`)
- [x] Phase 8 — historical backtest of the full simulator (`src/backtest_season.py`), rewinding to GW5 of a completed season with a leave-one-out promoted-team prior to avoid leakage. On 2024/25: top-4/relegation Brier scores ~5x better than naive baseline; title prediction confidently backed Man City/Arsenal over eventual champion Liverpool -- a real, permanent limit of results-only models (can't foresee injury collapses), not a bug
  - Extended to a 3-season x 7-cutoff grid (GW5-GW35): top-4/relegation Brier scores improve almost monotonically with more data in every season; title Brier score is genuinely non-monotonic (can get worse mid-season before collapsing to ~0 by GW30) since a title race has only one winner among a few contenders, unlike top-4/relegation's multiple "slots" -- see `data/processed/backtest_grid.csv` and `backtest_calibration.png`

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
