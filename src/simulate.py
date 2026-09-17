"""
Phase 4: Monte Carlo season simulator.

Idea: simulate the rest of the season thousands of times. Each remaining
fixture gets a random scoreline drawn from its Poisson probabilities (a
70% favorite wins ~70% of simulated playouts, not every one). Add that to
the real current standings and you get one possible final table. Do this
10,000 times and count how often each team finishes 1st / top 4 / bottom 3
-- that fraction IS the title / top-4 / relegation probability.

Simplification: within one simulated season, attack/defense strength is
held FIXED at today's values (not recalculated after each simulated
match). Real updates happen between actual gameweeks (see Phase 5), not
inside hypothetical ones.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_load import RAW_DIR, load_all_seasons
from poisson_model import RECENT_SEASONS, compute_team_strengths
from team_news import expected_goals_for_fixture, load_team_news
from fixture_congestion import load_other_competitions

CURRENT_SEASON = "2026/27"
N_SIMULATIONS = 10_000
SCHEDULE_PATH = RAW_DIR / "schedule_2026_27.csv"


def get_current_standings(season_matches):
    """Real points/goals-for/goals-against from matches actually played."""
    teams = sorted(set(season_matches["HomeTeam"]) | set(season_matches["AwayTeam"]))
    standings = {t: {"Played": 0, "Points": 0, "GF": 0, "GA": 0} for t in teams}

    for m in season_matches.itertuples():
        h, a = m.HomeTeam, m.AwayTeam
        standings[h]["Played"] += 1
        standings[a]["Played"] += 1
        standings[h]["GF"] += m.FTHG
        standings[h]["GA"] += m.FTAG
        standings[a]["GF"] += m.FTAG
        standings[a]["GA"] += m.FTHG
        if m.FTR == "H":
            standings[h]["Points"] += 3
        elif m.FTR == "A":
            standings[a]["Points"] += 3
        else:
            standings[h]["Points"] += 1
            standings[a]["Points"] += 1

    return standings


def remaining_fixtures(teams, season_matches):
    """
    A full PL season is every team playing every other team home and away
    (20 x 19 = 380 fixtures). Subtract the ones already played to get
    what's left. Undated -- use remaining_fixtures_with_dates() when dates
    matter (e.g. for team-news windows).
    """
    full_season = {(h, a) for h in teams for a in teams if h != a}
    played = set(zip(season_matches["HomeTeam"], season_matches["AwayTeam"]))
    return sorted(full_season - played)


def remaining_fixtures_with_dates(schedule, season_matches):
    """Same idea as remaining_fixtures(), but sourced from the real dated schedule."""
    played_pairs = set(zip(season_matches["HomeTeam"], season_matches["AwayTeam"]))
    is_unplayed = ~schedule.apply(lambda r: (r["HomeTeam"], r["AwayTeam"]) in played_pairs, axis=1)
    return schedule[is_unplayed].sort_values("Date").reset_index(drop=True)


def precompute_fixture_xg(fixtures_df, attack, defense, avg_home, avg_away, team_news=None,
                           all_matches=None, other_competitions=None):
    """
    Expected goals for each remaining fixture, computed once (fixed for
    every simulation). Applies team_news adjustments per fixture based on
    its real date, so a time-limited injury only affects fixtures that
    actually fall in that window. Also applies a short-rest fixture-
    congestion penalty when all_matches/other_competitions are given --
    see fixture_congestion.py.
    """
    home_xg, away_xg = [], []
    for f in fixtures_df.itertuples():
        h_xg, a_xg = expected_goals_for_fixture(
            f.HomeTeam, f.AwayTeam, f.Date, attack, defense, avg_home, avg_away, team_news,
            all_matches=all_matches, other_competitions=other_competitions,
        )
        home_xg.append(h_xg)
        away_xg.append(a_xg)
    return np.array(home_xg), np.array(away_xg)


def final_table_order(standings):
    """Sort by Points desc, then Goal Difference desc, then Goals For desc (simplified PL tiebreak)."""
    def sort_key(team):
        s = standings[team]
        gd = s["GF"] - s["GA"]
        return (-s["Points"], -gd, -s["GF"])
    return sorted(standings.keys(), key=sort_key)


def simulate_seasons(standings, fixtures, home_xg, away_xg, n_simulations=N_SIMULATIONS, seed=42):
    teams = list(standings.keys())
    rng = np.random.default_rng(seed)

    title_count = {t: 0 for t in teams}
    top4_count = {t: 0 for t in teams}
    relegated_count = {t: 0 for t in teams}
    points_total = {t: 0 for t in teams}

    for _ in range(n_simulations):
        sim_standings = {t: dict(v) for t, v in standings.items()}

        home_goals = rng.poisson(home_xg)
        away_goals = rng.poisson(away_xg)

        for (home, away), hg, ag in zip(fixtures, home_goals, away_goals):
            sim_standings[home]["GF"] += hg
            sim_standings[home]["GA"] += ag
            sim_standings[away]["GF"] += ag
            sim_standings[away]["GA"] += hg
            if hg > ag:
                sim_standings[home]["Points"] += 3
            elif hg < ag:
                sim_standings[away]["Points"] += 3
            else:
                sim_standings[home]["Points"] += 1
                sim_standings[away]["Points"] += 1

        order = final_table_order(sim_standings)
        title_count[order[0]] += 1
        for t in order[:4]:
            top4_count[t] += 1
        for t in order[-3:]:
            relegated_count[t] += 1
        for t in teams:
            points_total[t] += sim_standings[t]["Points"]

    results = pd.DataFrame({
        "CurrentPoints": {t: standings[t]["Points"] for t in teams},
        "Played": {t: standings[t]["Played"] for t in teams},
        "TitleChance": {t: title_count[t] / n_simulations for t in teams},
        "Top4Chance": {t: top4_count[t] / n_simulations for t in teams},
        "RelegationChance": {t: relegated_count[t] / n_simulations for t in teams},
        "AvgFinalPoints": {t: points_total[t] / n_simulations for t in teams},
    })
    return results.sort_values("TitleChance", ascending=False)


if __name__ == "__main__":
    matches = load_all_seasons()
    season_matches = matches[matches["Season"] == CURRENT_SEASON]
    recent = matches[matches["Season"].isin(RECENT_SEASONS)]

    schedule = pd.read_csv(SCHEDULE_PATH, parse_dates=["Date"])
    standings = get_current_standings(season_matches)
    fixtures_df = remaining_fixtures_with_dates(schedule, season_matches)
    fixture_pairs = list(zip(fixtures_df["HomeTeam"], fixtures_df["AwayTeam"]))

    team_news = load_team_news()
    other_competitions = load_other_competitions()
    attack, defense, avg_home, avg_away = compute_team_strengths(recent)
    home_xg, away_xg = precompute_fixture_xg(
        fixtures_df, attack, defense, avg_home, avg_away, team_news,
        all_matches=matches, other_competitions=other_competitions,
    )

    print(f"{len(season_matches)} matches played, {len(fixtures_df)} remaining. "
          f"Running {N_SIMULATIONS:,} simulations...\n")
    if len(team_news):
        print(f"Applying {len(team_news)} active team-news adjustment(s):")
        print(team_news.to_string(index=False))
        print()

    results = simulate_seasons(standings, fixture_pairs, home_xg, away_xg)

    pd.set_option("display.float_format", lambda x: f"{x:.1%}" if x <= 1 else f"{x:.1f}")
    print(results.to_string())
