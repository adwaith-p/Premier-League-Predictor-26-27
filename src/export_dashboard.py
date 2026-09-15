"""
Export everything the UI needs into one JSON file: current league table +
odds, next gameweek predictions, odds history, and active team news.

Run this after update_weekly.py (or it recomputes from scratch itself --
either way is fine, results are the same, this just avoids re-running the
season simulation if you already just did).
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_load import RAW_DIR, load_all_seasons
from poisson_model import (
    RECENT_SEASONS,
    compute_team_strengths,
    match_outcome_probabilities,
    probability_over_line,
)
from team_news import expected_goals_for_fixture, load_team_news
from simulate import (
    CURRENT_SEASON,
    get_current_standings,
    precompute_fixture_xg,
    remaining_fixtures_with_dates,
    simulate_seasons,
)

SCHEDULE_PATH = RAW_DIR / "schedule_2026_27.csv"
SNAPSHOT_LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "simulation_snapshots.csv"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "dashboard_data.json"


def next_gameweek_fixtures(schedule, season_matches, n=10):
    played_pairs = set(zip(season_matches["HomeTeam"], season_matches["AwayTeam"]))
    is_unplayed = ~schedule.apply(lambda r: (r["HomeTeam"], r["AwayTeam"]) in played_pairs, axis=1)
    return schedule[is_unplayed].sort_values("Date").head(n)


if __name__ == "__main__":
    matches = load_all_seasons()
    season_matches = matches[matches["Season"] == CURRENT_SEASON]
    recent = matches[matches["Season"].isin(RECENT_SEASONS)]
    schedule = pd.read_csv(SCHEDULE_PATH, parse_dates=["Date"])

    attack, defense, avg_home, avg_away = compute_team_strengths(recent)
    standings = get_current_standings(season_matches)
    fixtures_df = remaining_fixtures_with_dates(schedule, season_matches)
    fixture_pairs = list(zip(fixtures_df["HomeTeam"], fixtures_df["AwayTeam"]))
    team_news = load_team_news()

    home_xg, away_xg = precompute_fixture_xg(fixtures_df, attack, defense, avg_home, avg_away, team_news)
    sim_results = simulate_seasons(standings, fixture_pairs, home_xg, away_xg)

    table = []
    for team, row in sim_results.iterrows():
        table.append({
            "team": team,
            "played": int(row["Played"]),
            "points": int(row["CurrentPoints"]),
            "goalsFor": int(standings[team]["GF"]),
            "goalsAgainst": int(standings[team]["GA"]),
            "goalDifference": int(standings[team]["GF"] - standings[team]["GA"]),
            "titleChance": round(float(row["TitleChance"]), 4),
            "top4Chance": round(float(row["Top4Chance"]), 4),
            "relegationChance": round(float(row["RelegationChance"]), 4),
            "avgFinalPoints": round(float(row["AvgFinalPoints"]), 1),
        })

    upcoming = next_gameweek_fixtures(schedule, season_matches)
    next_gameweek = []
    for f in upcoming.itertuples():
        h_xg, a_xg = expected_goals_for_fixture(
            f.HomeTeam, f.AwayTeam, f.Date, attack, defense, avg_home, avg_away, team_news
        )
        p_home, p_draw, p_away = match_outcome_probabilities(h_xg, a_xg)
        p_over = probability_over_line(h_xg, a_xg, line=2.5)
        next_gameweek.append({
            "date": f.Date.strftime("%Y-%m-%d"),
            "homeTeam": f.HomeTeam,
            "awayTeam": f.AwayTeam,
            "homeXg": round(h_xg, 2),
            "awayXg": round(a_xg, 2),
            "pHome": round(p_home, 3),
            "pDraw": round(p_draw, 3),
            "pAway": round(p_away, 3),
            "pOver25": round(p_over, 3),
        })

    history = []
    if SNAPSHOT_LOG_PATH.exists():
        snap = pd.read_csv(SNAPSHOT_LOG_PATH)
        for row in snap.itertuples():
            history.append({
                "runDate": row.RunDate,
                "team": row.Team,
                "titleChance": round(float(row.TitleChance), 4),
                "top4Chance": round(float(row.Top4Chance), 4),
                "relegationChance": round(float(row.RelegationChance), 4),
            })

    news = []
    for row in team_news.itertuples():
        news.append({
            "team": row.Team,
            "startDate": row.StartDate.strftime("%Y-%m-%d"),
            "endDate": row.EndDate.strftime("%Y-%m-%d"),
            "attackMultiplier": row.AttackMultiplier,
            "defenseMultiplier": row.DefenseMultiplier,
            "note": row.Note,
        })

    dashboard_data = {
        "generatedAt": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "season": CURRENT_SEASON,
        "matchesPlayed": len(season_matches),
        "table": table,
        "nextGameweek": next_gameweek,
        "oddsHistory": history,
        "teamNews": news,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(dashboard_data, indent=2))
    print(f"Wrote {OUTPUT_PATH} ({len(table)} teams, {len(next_gameweek)} upcoming fixtures, "
          f"{len(history)} history rows, {len(news)} news entries)")
