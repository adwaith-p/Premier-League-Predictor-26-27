"""
Phase 9: ML layer, step 1 -- feature engineering.

Builds one training row per historical match: a handful of features
computed ONLY from information available before that match kicked off
(no leakage), plus the actual goals scored as the prediction target.

This will become a drop-in alternative to poisson_model.expected_goals():
same two targets (home goals, away goals), same downstream Poisson/
simulation machinery -- only the way we estimate expected goals changes,
which keeps the eventual comparison to the statistical baseline honest.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_load import load_all_seasons
from elo import run_ratings_history

FORM_WINDOW = 5  # games of recent form to average over

LEAGUE_AVG_GOALS = 1.4     # fallback for a team's very first tracked match
LEAGUE_AVG_POINTS = 1.35   # fallback "points per game" for a team with no history yet
DEFAULT_REST_DAYS = 7      # fallback for a team's first match in the dataset


def _team_match_log(matches):
    """One row per team-appearance (so each match contributes two rows), for computing rolling form."""
    home = matches[["Date", "Season", "HomeTeam", "FTHG", "FTAG"]].rename(
        columns={"HomeTeam": "Team", "FTHG": "GoalsScored", "FTAG": "GoalsConceded"})
    away = matches[["Date", "Season", "AwayTeam", "FTAG", "FTHG"]].rename(
        columns={"AwayTeam": "Team", "FTAG": "GoalsScored", "FTHG": "GoalsConceded"})
    log = pd.concat([home, away], ignore_index=True)

    log["Points"] = 0
    log.loc[log["GoalsScored"] > log["GoalsConceded"], "Points"] = 3
    log.loc[log["GoalsScored"] == log["GoalsConceded"], "Points"] = 1

    log = log.sort_values(["Team", "Date"]).reset_index(drop=True)
    return log


def _rolling_form(log):
    """
    Rolling averages over the last FORM_WINDOW games, shifted by one so a
    match's features only reflect games strictly BEFORE it -- using the
    match's own result to predict itself would be leakage, not a feature.
    """
    grouped = log.groupby("Team")
    log["FormGoalsScored"] = grouped["GoalsScored"].transform(
        lambda s: s.rolling(FORM_WINDOW, min_periods=1).mean().shift(1))
    log["FormGoalsConceded"] = grouped["GoalsConceded"].transform(
        lambda s: s.rolling(FORM_WINDOW, min_periods=1).mean().shift(1))
    log["FormPoints"] = grouped["Points"].transform(
        lambda s: s.rolling(FORM_WINDOW, min_periods=1).mean().shift(1))
    log["RestDays"] = grouped["Date"].diff().dt.days
    log["GamesPlayedThisSeason"] = log.groupby(["Team", "Season"]).cumcount()

    log["FormGoalsScored"] = log["FormGoalsScored"].fillna(LEAGUE_AVG_GOALS)
    log["FormGoalsConceded"] = log["FormGoalsConceded"].fillna(LEAGUE_AVG_GOALS)
    log["FormPoints"] = log["FormPoints"].fillna(LEAGUE_AVG_POINTS)
    log["RestDays"] = log["RestDays"].fillna(DEFAULT_REST_DAYS)
    return log


def _promoted_lookup(matches):
    """(season, team) -> True if that team has no top-flight data in the season before it."""
    seasons = sorted(matches["Season"].unique())
    lookup = {}
    for i in range(1, len(seasons)):
        prior_teams = set(matches[matches["Season"] == seasons[i - 1]]["HomeTeam"]) | \
            set(matches[matches["Season"] == seasons[i - 1]]["AwayTeam"])
        this_season_teams = set(matches[matches["Season"] == seasons[i]]["HomeTeam"]) | \
            set(matches[matches["Season"] == seasons[i]]["AwayTeam"])
        for team in this_season_teams:
            lookup[(seasons[i], team)] = team not in prior_teams
    return lookup


def build_feature_dataset():
    matches = load_all_seasons()

    _, elo_history = run_ratings_history(matches)
    matches = matches.reset_index(drop=True)
    matches["PreHomeElo"] = elo_history["PreHomeElo"]
    matches["PreAwayElo"] = elo_history["PreAwayElo"]

    log = _rolling_form(_team_match_log(matches))
    feature_cols = ["FormGoalsScored", "FormGoalsConceded", "FormPoints", "RestDays", "GamesPlayedThisSeason"]

    home_features = log.rename(columns={"Team": "HomeTeam", **{c: f"Home{c}" for c in feature_cols}})
    home_features = home_features[["Date", "HomeTeam"] + [f"Home{c}" for c in feature_cols]]
    away_features = log.rename(columns={"Team": "AwayTeam", **{c: f"Away{c}" for c in feature_cols}})
    away_features = away_features[["Date", "AwayTeam"] + [f"Away{c}" for c in feature_cols]]

    dataset = matches.merge(home_features, on=["Date", "HomeTeam"], how="left")
    dataset = dataset.merge(away_features, on=["Date", "AwayTeam"], how="left")

    promoted = _promoted_lookup(matches)
    dataset["HomePromoted"] = dataset.apply(lambda r: promoted.get((r["Season"], r["HomeTeam"]), False), axis=1)
    dataset["AwayPromoted"] = dataset.apply(lambda r: promoted.get((r["Season"], r["AwayTeam"]), False), axis=1)

    return dataset


FEATURE_COLUMNS = [
    "PreHomeElo", "PreAwayElo",
    "HomeFormGoalsScored", "HomeFormGoalsConceded", "HomeFormPoints", "HomeRestDays", "HomeGamesPlayedThisSeason",
    "AwayFormGoalsScored", "AwayFormGoalsConceded", "AwayFormPoints", "AwayRestDays", "AwayGamesPlayedThisSeason",
    "HomePromoted", "AwayPromoted",
]


if __name__ == "__main__":
    dataset = build_feature_dataset()

    print(f"Built {len(dataset)} match rows, {len(FEATURE_COLUMNS)} features.\n")
    print("Missing values per feature column:")
    print(dataset[FEATURE_COLUMNS].isna().sum().to_string())

    print("\nSample (5 most recent matches):")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    cols_to_show = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"] + FEATURE_COLUMNS
    print(dataset[cols_to_show].tail(5).to_string(index=False))

    dataset.to_csv(Path(__file__).resolve().parent.parent / "data" / "processed" / "ml_features.csv", index=False)
    print(f"\nSaved to data/processed/ml_features.csv")
