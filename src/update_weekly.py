"""
Phase 5: the weekly update pipeline.

Run this after each real gameweek (or anytime) to:
  1. Re-download the latest results for the current season
  2. Check our full schedule for postponed/rearranged fixtures
  3. Recompute Elo ratings + backtest accuracy
  4. Recompute Poisson attack/defense strengths
  5. Re-run the season simulation (title / top4 / relegation odds) and
     log a snapshot, so we can see how the odds evolve week to week
  6. Predict the next gameweek's specific fixtures
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_load import RAW_DIR, load_all_seasons
from elo import run_ratings_history
from backtest import backtest
from poisson_model import (
    RECENT_SEASONS,
    compute_team_strengths,
    expected_goals,
    match_outcome_probabilities,
    most_likely_scoreline,
)
from simulate import (
    CURRENT_SEASON,
    get_current_standings,
    precompute_fixture_xg,
    remaining_fixtures,
    simulate_seasons,
)

RESULTS_URL = "https://www.football-data.co.uk/mmz4281/2627/E0.csv"
RESULTS_PATH = RAW_DIR / "E0_2627.csv"
SCHEDULE_PATH = RAW_DIR / "schedule_2026_27.csv"
SNAPSHOT_LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "simulation_snapshots.csv"


def refresh_results():
    """Re-download the current season's results file -- same data as Phase 0, just live."""
    response = requests.get(RESULTS_URL, timeout=30)
    response.raise_for_status()
    RESULTS_PATH.write_bytes(response.content)


def check_for_postponements(schedule, season_matches):
    """
    Fixtures our schedule says should have happened by now, but that
    aren't in the results yet -- most likely postponed or rearranged.
    """
    played_pairs = set(zip(season_matches["HomeTeam"], season_matches["AwayTeam"]))
    today = pd.Timestamp.now().normalize()
    is_overdue = schedule["Date"] < today
    is_unplayed = ~schedule.apply(lambda r: (r["HomeTeam"], r["AwayTeam"]) in played_pairs, axis=1)
    return schedule[is_overdue & is_unplayed]


def next_gameweek_fixtures(schedule, season_matches, n=10):
    """The next N still-unplayed fixtures, in date order -- normally one full gameweek."""
    played_pairs = set(zip(season_matches["HomeTeam"], season_matches["AwayTeam"]))
    is_unplayed = ~schedule.apply(lambda r: (r["HomeTeam"], r["AwayTeam"]) in played_pairs, axis=1)
    return schedule[is_unplayed].sort_values("Date").head(n)


def log_snapshot(sim_results):
    """Append today's title/top4/relegation odds to a running history file (one row per team per run)."""
    run_date = datetime.now().strftime("%Y-%m-%d")
    snapshot = sim_results.reset_index().rename(columns={"index": "Team"})
    snapshot.insert(0, "RunDate", run_date)

    SNAPSHOT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if SNAPSHOT_LOG_PATH.exists():
        existing = pd.read_csv(SNAPSHOT_LOG_PATH)
        existing = existing[existing["RunDate"] != run_date]  # overwrite if re-run same day
        combined = pd.concat([existing, snapshot], ignore_index=True)
    else:
        combined = snapshot
    combined.to_csv(SNAPSHOT_LOG_PATH, index=False)


if __name__ == "__main__":
    print("=== Step 1: refreshing results ===")
    refresh_results()

    matches = load_all_seasons()
    season_matches = matches[matches["Season"] == CURRENT_SEASON]
    print(f"{len(season_matches)} matches played so far in {CURRENT_SEASON}")

    print("\n=== Step 2: checking for postponements ===")
    schedule = pd.read_csv(SCHEDULE_PATH, parse_dates=["Date"])
    overdue = check_for_postponements(schedule, season_matches)
    if len(overdue):
        print(f"WARNING: {len(overdue)} fixture(s) scheduled in the past with no result yet "
              f"-- possibly postponed or rearranged:")
        print(overdue.to_string(index=False))
    else:
        print("No postponements detected -- every past-dated fixture has a result.")

    print("\n=== Step 3: Elo ratings + backtest ===")
    _, history = run_ratings_history(matches)
    accuracy, home_win_rate, n_decisive = backtest(history)
    print(f"Elo backtest accuracy: {accuracy:.1%} (baseline {home_win_rate:.1%}) over {n_decisive} decisive matches")

    print("\n=== Step 4: Poisson strengths + season simulation ===")
    recent = matches[matches["Season"].isin(RECENT_SEASONS)]
    attack, defense, avg_home, avg_away = compute_team_strengths(recent)
    teams = sorted(set(season_matches["HomeTeam"]) | set(season_matches["AwayTeam"]))
    standings = get_current_standings(season_matches)
    fixtures = remaining_fixtures(teams, season_matches)
    home_xg, away_xg = precompute_fixture_xg(fixtures, attack, defense, avg_home, avg_away)
    sim_results = simulate_seasons(standings, fixtures, home_xg, away_xg)
    log_snapshot(sim_results)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.1%}" if x <= 1 else f"{x:.1f}")
    print(sim_results.to_string())

    print("\n=== Step 5: next gameweek predictions ===")
    upcoming = next_gameweek_fixtures(schedule, season_matches)
    for f in upcoming.itertuples():
        h_xg, a_xg = expected_goals(f.HomeTeam, f.AwayTeam, attack, defense, avg_home, avg_away)
        p_home, p_draw, p_away = match_outcome_probabilities(h_xg, a_xg)
        (h, a), score_p = most_likely_scoreline(h_xg, a_xg)
        print(f"{f.Date.strftime('%a %d %b')}  {f.HomeTeam:15s} vs {f.AwayTeam:15s}  "
              f"xG {h_xg:.2f}-{a_xg:.2f}  H/D/A {p_home:.0%}/{p_draw:.0%}/{p_away:.0%}  "
              f"most likely {h}-{a} ({score_p:.1%})")
