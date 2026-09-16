"""
Phase 8: historical backtest of the full season simulator.

Everything so far has been sanity-checked one number at a time (does Hull's
defense look sane, does the Elo favorite win more often than not). This is
the real test: rewind to gameweek 5 of a season we already know the ending
of, run the whole pipeline exactly as if it were live, and compare the
predicted title/top4/relegation odds against what actually happened.

Leakage warning this script exists to handle: PROMOTED_ATTACK_PRIOR and
PROMOTED_DEFENSE_PRIOR (in poisson_model.py) were fit on ALL THREE
historically-promoted cohorts at once (2023/24, 2024/25, 2025/26). If we
backtest on, say, 2024/25, using that same prior would mean the model is
partly graded on data it was tuned with. So this script recomputes a
LEAVE-ONE-OUT prior -- built only from the OTHER two seasons' promoted
teams -- before running the backtest.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_load import load_all_seasons
from poisson_model import compute_team_strengths
from simulate import get_current_standings, precompute_fixture_xg, simulate_seasons
from team_news import expected_goals_for_fixture

GAMEWEEK_CUTOFF_MATCHES = 50  # 5 gameweeks x 10 matches -- how far into the test season we "know"

# All historically promoted cohorts in our data (see poisson_model.py's own
# derivation). Used to build a leave-one-out prior per test season.
PROMOTED_COHORTS = {
    "2023/24": ["Burnley", "Luton", "Sheffield United"],
    "2024/25": ["Ipswich", "Leicester", "Southampton"],
    "2025/26": ["Burnley", "Leeds", "Sunderland"],
}


def leave_one_out_promoted_prior(all_matches, test_season):
    """
    Average attack/defense strength of promoted teams in every cohort
    EXCEPT test_season, using each team's full season of real data (not
    shrunk -- this is fitting the prior itself, not applying it).
    """
    attacks, defenses = [], []
    for season, teams in PROMOTED_COHORTS.items():
        if season == test_season:
            continue
        season_matches = all_matches[all_matches["Season"] == season]
        avg_home = season_matches["FTHG"].mean()
        avg_away = season_matches["FTAG"].mean()
        for team in teams:
            home = season_matches[season_matches["HomeTeam"] == team]
            away = season_matches[season_matches["AwayTeam"] == team]
            attack = (home["FTHG"].mean() / avg_home + away["FTAG"].mean() / avg_away) / 2
            defense = (home["FTAG"].mean() / avg_away + away["FTHG"].mean() / avg_home) / 2
            attacks.append(attack)
            defenses.append(defense)
    return sum(attacks) / len(attacks), sum(defenses) / len(defenses)


def actual_final_table(season_matches):
    """The real final standings for a completed season, for comparison."""
    standings = get_current_standings(season_matches)
    teams = sorted(standings.keys(), key=lambda t: (
        -standings[t]["Points"], -(standings[t]["GF"] - standings[t]["GA"]), -standings[t]["GF"]
    ))
    return teams, standings


def run_backtest(test_season, prior_season, n_simulations=10_000):
    all_matches = load_all_seasons()
    season_matches = all_matches[all_matches["Season"] == test_season].sort_values("Date").reset_index(drop=True)

    known_matches = season_matches.iloc[:GAMEWEEK_CUTOFF_MATCHES]
    future_matches = season_matches.iloc[GAMEWEEK_CUTOFF_MATCHES:]

    # "recent" window for strength estimation: prior full season + what we know so far
    strength_window = pd.concat([all_matches[all_matches["Season"] == prior_season], known_matches])

    attack_prior, defense_prior = leave_one_out_promoted_prior(all_matches, test_season)
    attack, defense, avg_home, avg_away = compute_team_strengths(
        strength_window, promoted_attack_prior=attack_prior, promoted_defense_prior=defense_prior
    )

    standings = get_current_standings(known_matches)
    fixtures_df = future_matches[["Date", "HomeTeam", "AwayTeam"]].reset_index(drop=True)
    fixture_pairs = list(zip(fixtures_df["HomeTeam"], fixtures_df["AwayTeam"]))

    empty_news = pd.DataFrame(columns=["Team", "StartDate", "EndDate", "AttackMultiplier", "DefenseMultiplier", "Note"])
    home_xg, away_xg = precompute_fixture_xg(fixtures_df, attack, defense, avg_home, avg_away, empty_news)

    predictions = simulate_seasons(standings, fixture_pairs, home_xg, away_xg, n_simulations=n_simulations)
    actual_order, actual_standings = actual_final_table(season_matches)

    return predictions, actual_order, attack_prior, defense_prior


def brier_score(predictions, actual_order, column, actual_slice):
    """
    Mean squared error between predicted probability and the actual 0/1
    outcome, across all teams. 0 = perfect, 0.25 = "no better than a coin
    flip guessed at 50% for everyone", lower is better. This is the
    standard way to score a probabilistic forecast -- accuracy alone isn't
    enough because a forecast can be "right" for the wrong reasons.
    """
    actual_set = set(actual_order[actual_slice])
    errors = [(predictions.loc[team, column] - (1.0 if team in actual_set else 0.0)) ** 2
              for team in predictions.index]
    return sum(errors) / len(errors)


if __name__ == "__main__":
    TEST_SEASON = "2024/25"
    PRIOR_SEASON = "2023/24"

    print(f"Backtesting: rewind to gameweek 5 of {TEST_SEASON}, simulate the rest, "
          f"compare against what actually happened.\n")

    predictions, actual_order, attack_prior, defense_prior = run_backtest(TEST_SEASON, PRIOR_SEASON)

    print(f"Leave-one-out promoted-team prior (excludes {TEST_SEASON}'s own cohort): "
          f"attack={attack_prior:.3f} defense={defense_prior:.3f}\n")

    print("Actual final table:")
    for i, team in enumerate(actual_order, 1):
        marker = " (promoted)" if team in PROMOTED_COHORTS[TEST_SEASON] else ""
        print(f"  {i:2d}. {team}{marker}")

    print(f"\nPredicted odds (made after just 5 gameweeks) vs what actually happened:")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    comparison = predictions[["CurrentPoints", "TitleChance", "Top4Chance", "RelegationChance"]].copy()
    comparison["ActualPosition"] = comparison.index.map(lambda t: actual_order.index(t) + 1)
    comparison["ActualChampion"] = comparison.index == actual_order[0]
    comparison["ActualTop4"] = comparison.index.map(lambda t: actual_order.index(t) < 4)
    comparison["ActualRelegated"] = comparison.index.map(lambda t: actual_order.index(t) >= 17)
    comparison = comparison.sort_values("ActualPosition")
    pd.set_option("display.float_format", lambda x: f"{x:.1%}" if x <= 1 else f"{x:.1f}")
    print(comparison.to_string())

    title_brier = brier_score(predictions, actual_order, "TitleChance", slice(0, 1))
    top4_brier = brier_score(predictions, actual_order, "Top4Chance", slice(0, 4))
    releg_brier = brier_score(predictions, actual_order, "RelegationChance", slice(17, 20))

    n = len(predictions)
    baseline_title_brier = brier_score(
        pd.DataFrame({"TitleChance": [1 / n] * n}, index=predictions.index), actual_order, "TitleChance", slice(0, 1))
    baseline_top4_brier = brier_score(
        pd.DataFrame({"Top4Chance": [4 / n] * n}, index=predictions.index), actual_order, "Top4Chance", slice(0, 4))
    baseline_releg_brier = brier_score(
        pd.DataFrame({"RelegationChance": [3 / n] * n}, index=predictions.index), actual_order, "RelegationChance", slice(17, 20))

    print(f"\nBrier scores (lower is better, 0 = perfect):")
    print(f"  Title:      model={title_brier:.4f}   naive baseline (1/20 for everyone)={baseline_title_brier:.4f}")
    print(f"  Top 4:      model={top4_brier:.4f}   naive baseline (4/20 for everyone)={baseline_top4_brier:.4f}")
    print(f"  Relegation: model={releg_brier:.4f}   naive baseline (3/20 for everyone)={baseline_releg_brier:.4f}")
