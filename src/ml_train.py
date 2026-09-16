"""
Phase 9, step 2: train and evaluate the ML goal model.

Time-based split (never random -- shuffling would let the model "see"
matches from the future relative to some training examples): train on
everything before a test season, evaluate on that season untouched.

Uses XGBoost's Poisson objective (count:poisson), not the default squared
-error one -- goals are non-negative counts, exactly what a Poisson
regression is built for. Same statistical idea as Phase 3's model, just
letting gradient boosting find the pattern instead of a fixed formula.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ml_features import FEATURE_COLUMNS, build_feature_dataset
from poisson_model import PROMOTED_ATTACK_PRIOR, PROMOTED_DEFENSE_PRIOR, compute_team_strengths, expected_goals

TEST_SEASON = "2025/26"      # most recent COMPLETE season -- 2026/27 is still in progress, held out entirely
TRAIN_SEASONS_BEFORE = TEST_SEASON  # train on every season strictly before this one


def time_based_split(dataset):
    train = dataset[dataset["Season"] < TRAIN_SEASONS_BEFORE]
    test = dataset[dataset["Season"] == TEST_SEASON]
    return train, test


def train_models(train):
    X = train[FEATURE_COLUMNS]
    model_home = XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.05,
                               objective="count:poisson", subsample=0.8, colsample_bytree=0.8, random_state=42)
    model_away = XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.05,
                               objective="count:poisson", subsample=0.8, colsample_bytree=0.8, random_state=42)
    model_home.fit(X, train["FTHG"])
    model_away.fit(X, train["FTAG"])
    return model_home, model_away


def static_poisson_baseline(test):
    """
    A single, un-updated Poisson snapshot fit ONLY on the season before
    the test season (like a preseason prediction that never adjusts) --
    isolates "is the goal-creation function itself better" from "does it
    update weekly", which is a separate, fairer comparison for later.
    """
    from data_load import load_all_seasons
    all_matches = load_all_seasons()
    prior_season_matches = all_matches[all_matches["Season"] < TEST_SEASON].tail(380)  # last complete season only
    attack, defense, avg_home, avg_away = compute_team_strengths(prior_season_matches)

    # A team with ZERO games in the single prior season (a true preseason
    # snapshot, unlike the live model's 2-season window) never gets an
    # entry at all -- must be newly promoted, so fall back to the same
    # empirical promoted-team prior the live model uses.
    test_teams = set(test["HomeTeam"]) | set(test["AwayTeam"])
    for team in test_teams - set(attack.keys()):
        attack[team] = PROMOTED_ATTACK_PRIOR
        defense[team] = PROMOTED_DEFENSE_PRIOR

    home_preds, away_preds = [], []
    for row in test.itertuples():
        h_xg, a_xg = expected_goals(row.HomeTeam, row.AwayTeam, attack, defense, avg_home, avg_away)
        home_preds.append(h_xg)
        away_preds.append(a_xg)
    return np.array(home_preds), np.array(away_preds)


if __name__ == "__main__":
    dataset = build_feature_dataset()
    train, test = time_based_split(dataset)
    print(f"Train: {len(train)} matches (everything before {TEST_SEASON})")
    print(f"Test:  {len(test)} matches ({TEST_SEASON}, held out entirely)\n")

    model_home, model_away = train_models(train)
    ml_home_pred = model_home.predict(test[FEATURE_COLUMNS])
    ml_away_pred = model_away.predict(test[FEATURE_COLUMNS])

    naive_home_pred = np.full(len(test), train["FTHG"].mean())
    naive_away_pred = np.full(len(test), train["FTAG"].mean())

    poisson_home_pred, poisson_away_pred = static_poisson_baseline(test)

    results = pd.DataFrame({
        "Model": ["Naive (league avg goals)", "Static Poisson (preseason snapshot)", "XGBoost (ML)"],
        "HomeGoalsMAE": [
            mean_absolute_error(test["FTHG"], naive_home_pred),
            mean_absolute_error(test["FTHG"], poisson_home_pred),
            mean_absolute_error(test["FTHG"], ml_home_pred),
        ],
        "AwayGoalsMAE": [
            mean_absolute_error(test["FTAG"], naive_away_pred),
            mean_absolute_error(test["FTAG"], poisson_away_pred),
            mean_absolute_error(test["FTAG"], ml_away_pred),
        ],
    })
    results["CombinedMAE"] = (results["HomeGoalsMAE"] + results["AwayGoalsMAE"]) / 2

    print("Mean Absolute Error predicting actual goals scored (lower is better):")
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(results.to_string(index=False))

    print("\nFeature importance (XGBoost, home-goals model):")
    importances = pd.Series(model_home.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
    print(importances.to_string())
