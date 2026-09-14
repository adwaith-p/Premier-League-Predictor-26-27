"""
Predict the next real gameweek's fixtures (not a full season simulation --
just match-by-match probabilities and most-likely scorelines).

football-data.co.uk's season file (E0_2627.csv) only contains PLAYED
matches, so it can't tell us what's coming up. Their separate
fixtures.csv covers every division's next batch of scheduled matches with
real dates -- we filter that down to E0 (Premier League) to get the
actual next gameweek.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_load import load_all_seasons, RAW_DIR
from poisson_model import (
    RECENT_SEASONS,
    compute_team_strengths,
    match_outcome_probabilities,
    most_likely_scoreline,
    probability_over_line,
)
from team_news import expected_goals_for_fixture, load_team_news


def load_upcoming_fixtures():
    df = pd.read_csv(RAW_DIR / "fixtures_upcoming.csv")
    df = df[df["Div"] == "E0"][["Date", "HomeTeam", "AwayTeam"]].copy()
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
    return df.sort_values("Date").reset_index(drop=True)


if __name__ == "__main__":
    matches = load_all_seasons()
    recent = matches[matches["Season"].isin(RECENT_SEASONS)]
    attack, defense, avg_home, avg_away = compute_team_strengths(recent)

    fixtures = load_upcoming_fixtures()
    team_news = load_team_news()
    if len(team_news):
        print(f"Applying {len(team_news)} active team-news adjustment(s):")
        print(team_news.to_string(index=False))
        print()

    print(f"Predictions for the next {len(fixtures)} fixtures:\n")
    for f in fixtures.itertuples():
        home_xg, away_xg = expected_goals_for_fixture(
            f.HomeTeam, f.AwayTeam, f.Date, attack, defense, avg_home, avg_away, team_news
        )
        p_home, p_draw, p_away = match_outcome_probabilities(home_xg, away_xg)
        (h, a), score_p = most_likely_scoreline(home_xg, away_xg)
        p_over = probability_over_line(home_xg, away_xg, line=2.5)

        date_str = f.Date.strftime("%a %d %b")
        print(f"{date_str}  {f.HomeTeam:15s} vs {f.AwayTeam:15s}  "
              f"xG {home_xg:.2f}-{away_xg:.2f}  "
              f"H/D/A {p_home:.0%}/{p_draw:.0%}/{p_away:.0%}  "
              f"Over2.5 {p_over:.0%}  "
              f"single most likely score: {h}-{a} ({score_p:.1%})")
