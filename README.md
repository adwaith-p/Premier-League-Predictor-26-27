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
- [ ] Phase 4 — season Monte Carlo simulator (`src/simulate.py`)
- [ ] Phase 5 — weekly update script (`src/update_weekly.py`)
- [ ] Phase 6 — manual team-news adjustments

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
