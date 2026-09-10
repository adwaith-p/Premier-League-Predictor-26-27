"""
Phase 2: Elo rating engine.

Core idea:
  - every team has a rating (float), starting at BASE_RATING
  - before a match, we compute each team's *expected* score (a probability-
    like value, 0-1) from the rating gap, with the home team getting a
    fixed Elo boost for home advantage (for this calculation only)
  - after the match, we compare expected vs actual result and nudge both
    ratings toward reality, scaled by K_FACTOR

YOUR TASK: implement expected_score(), actual_score(), and
update_after_match() below. Formulas are in the docstrings. Everything
else (looping through match history, building the ratings table) is
already wired up so you can test as soon as those three are done.
"""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_load import load_all_seasons

BASE_RATING = 1500.0
HOME_ADVANTAGE = 100.0   # Elo points added to home team's rating, for the expectation calc only
K_FACTOR = 20.0          # how much a single match can move a rating


def expected_score(rating_a: float, rating_b: float) -> float:
    """
    Return team A's expected score against team B: a value between 0 and 1.
    1500 vs 1500 -> 0.5. Higher rating_a relative to rating_b -> higher.

    Formula:
        E_A = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    """
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def actual_score(result: str, perspective: str) -> float:
    """
    Convert a match result into a 0-1 score from one team's perspective.

    Args:
        result: 'H' (home win), 'D' (draw), or 'A' (away win)
        perspective: 'home' or 'away' -- whose score we want

    Returns:
        1.0 if that team won, 0.5 if draw, 0.0 if that team lost.
    """
    if perspective == "home":
        if result == "H":
            return 1.0
        elif result == "D":
            return 0.5
        else:
            return 0.0
    else:  # perspective == "away"
        if result == "A":
            return 1.0
        elif result == "D":
            return 0.5
        else:
            return 0.0


def update_after_match(home_rating: float, away_rating: float, result: str) -> tuple[float, float]:
    """
    Given pre-match ratings and the result, return the new
    (home_rating, away_rating) after one Elo update.

    Steps:
      1. home_expected = expected_score(home_rating + HOME_ADVANTAGE, away_rating)
      2. away_expected = 1 - home_expected
      3. home_actual = actual_score(result, 'home'); away_actual = actual_score(result, 'away')
      4. new_home_rating = home_rating + K_FACTOR * (home_actual - home_expected)
         new_away_rating = away_rating + K_FACTOR * (away_actual - away_expected)

    Note: home_rating itself is NOT permanently changed by HOME_ADVANTAGE --
    that boost only affects the expectation calc in step 1.
    """
    home_expected = expected_score(home_rating + HOME_ADVANTAGE, away_rating)
    away_expected = 1 - home_expected

    home_actual = actual_score(result, "home")
    away_actual = actual_score(result, "away")

    new_home_rating = home_rating + K_FACTOR * (home_actual - home_expected)
    new_away_rating = away_rating + K_FACTOR * (away_actual - away_expected)

    return new_home_rating, new_away_rating


# --- Everything below this line is already done for you ---

def run_ratings_history(matches: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    """
    Walk through every match in date order, updating ratings as we go.

    Returns:
        final_ratings: dict of team -> current rating after the last match
        history: a DataFrame with one row per match showing pre-match
                 ratings for both teams (useful later for backtesting)
    """
    ratings: dict[str, float] = {}
    rows = []

    for match in matches.itertuples():
        home, away = match.HomeTeam, match.AwayTeam
        ratings.setdefault(home, BASE_RATING)
        ratings.setdefault(away, BASE_RATING)

        pre_home, pre_away = ratings[home], ratings[away]
        new_home, new_away = update_after_match(pre_home, pre_away, match.FTR)

        rows.append({
            "Date": match.Date, "HomeTeam": home, "AwayTeam": away,
            "FTR": match.FTR, "PreHomeElo": pre_home, "PreAwayElo": pre_away,
        })
        ratings[home], ratings[away] = new_home, new_away

    return ratings, pd.DataFrame(rows)


if __name__ == "__main__":
    matches = load_all_seasons()
    final_ratings, history = run_ratings_history(matches)

    print("Current Elo ratings (sorted, highest first):\n")
    for team, rating in sorted(final_ratings.items(), key=lambda kv: -kv[1]):
        print(f"  {team:20s} {rating:7.1f}")
