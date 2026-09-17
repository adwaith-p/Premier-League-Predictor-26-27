"""
Phase 6: manual team-news adjustments.

The model only knows what's in past results -- it has no idea a key
striker just got injured or a manager was sacked yesterday. This module
lets you manually log that kind of thing with a date range, so it only
affects the specific fixtures that fall in that window and automatically
stops applying once the window ends.

Add rows to data/team_news.csv:
    Team, StartDate, EndDate, AttackMultiplier, DefenseMultiplier, Note

    Team              exact team name (matches football-data.co.uk spelling,
                       e.g. "Man City", "Nott'm Forest", "Spurs" -> "Tottenham")
    StartDate/EndDate  inclusive date range (YYYY-MM-DD) the adjustment applies to
    AttackMultiplier   e.g. 0.85 = attack reduced 15% (losing a key attacking player)
    DefenseMultiplier  e.g. 1.15 = concedes 15% more (losing a key defender) --
                       remember LOWER defense numbers are better, so a worse
                       defense is a multiplier ABOVE 1.0
    Note               free text, for your own reference

Leave a multiplier at 1.0 if that side of the team's strength isn't affected.
Rough sizing guide: a squad player out ~0.95, a key rotation player ~0.90,
a genuine first-choice star or the first-choice keeper ~0.80-0.85, losing
the manager mid-adjustment period ~0.90 on both. These are judgment calls,
not fitted from data -- that's the point of this being manual.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_load import RAW_DIR
from poisson_model import expected_goals

TEAM_NEWS_PATH = RAW_DIR.parent / "team_news.csv"

TEAM_NEWS_COLUMNS = ["Team", "StartDate", "EndDate", "AttackMultiplier", "DefenseMultiplier", "Note"]


def load_team_news():
    if not TEAM_NEWS_PATH.exists() or TEAM_NEWS_PATH.stat().st_size == 0:
        return pd.DataFrame(columns=TEAM_NEWS_COLUMNS)
    news = pd.read_csv(TEAM_NEWS_PATH)
    if len(news) == 0:
        return news
    news["StartDate"] = pd.to_datetime(news["StartDate"])
    news["EndDate"] = pd.to_datetime(news["EndDate"])
    return news


def adjusted_strength(team, match_date, attack, defense, team_news):
    """
    Base attack/defense for `team`, adjusted by any team_news rows whose
    date range covers `match_date`. If more than one entry applies at
    once (e.g. two separate injuries), multipliers combine multiplicatively.
    """
    team_attack = attack[team]
    team_defense = defense[team]

    if team_news is None or len(team_news) == 0:
        return team_attack, team_defense

    active = team_news[
        (team_news["Team"] == team)
        & (team_news["StartDate"] <= match_date)
        & (match_date <= team_news["EndDate"])
    ]
    for row in active.itertuples():
        team_attack *= row.AttackMultiplier
        team_defense *= row.DefenseMultiplier

    return team_attack, team_defense


def expected_goals_for_fixture(home_team, away_team, match_date, attack, defense, avg_home, avg_away, team_news,
                                all_matches=None, other_competitions=None):
    """
    Same formula as poisson_model.expected_goals, but using team-news- and
    fixture-congestion-adjusted strengths for this date.

    all_matches/other_competitions are optional: pass both to also apply a
    short-rest penalty (see fixture_congestion.py) for teams playing soon
    after a cup/European fixture. Omit either to skip that adjustment
    (e.g. backtest_season.py doesn't use it, since it only knows about
    Premier League matches).
    """
    home_attack, home_defense = adjusted_strength(home_team, match_date, attack, defense, team_news)
    away_attack, away_defense = adjusted_strength(away_team, match_date, attack, defense, team_news)

    if all_matches is not None and other_competitions is not None:
        from fixture_congestion import congestion_multiplier
        home_cong_a, home_cong_d = congestion_multiplier(home_team, match_date, all_matches, other_competitions)
        away_cong_a, away_cong_d = congestion_multiplier(away_team, match_date, all_matches, other_competitions)
        home_attack *= home_cong_a
        home_defense *= home_cong_d
        away_attack *= away_cong_a
        away_defense *= away_cong_d

    adj_attack = {home_team: home_attack, away_team: away_attack}
    adj_defense = {home_team: home_defense, away_team: away_defense}
    return expected_goals(home_team, away_team, adj_attack, adj_defense, avg_home, avg_away)
