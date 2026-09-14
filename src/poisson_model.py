"""
Phase 3: Poisson goal model.

Elo tells us roughly "who's better", but nothing about draws or scorelines.
This model is a different, complementary idea: estimate each team's
attacking and defensive strength directly from actual goals scored/conceded,
then use those to predict the probability of every possible scoreline for
an upcoming match.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_load import load_all_seasons

# Window used to estimate *current* team strength. Older seasons include
# squads that have changed too much to be representative. Newly promoted
# teams (no data in 2025/26) fall back to "average" (1.0) until they've
# played enough 2026/27 matches -- a known limitation of this simple
# version, worth revisiting later.
RECENT_SEASONS = ["2025/26", "2026/27"]

MAX_GOALS = 8  # scorelines beyond this are practically impossible; no need to compute them


def poisson_pmf(k, lam):
    """
    Poisson probability of observing exactly k goals when the expected
    (average) number of goals is `lam`.

        P(X = k) = (lam ** k) * e^(-lam) / k!

    e.g. if a team's expected goals is 1.4, P(exactly 2 goals) = poisson_pmf(2, 1.4)
    """
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def compute_team_strengths(matches):
    """
    Attack strength: how many goals a team scores relative to the league
    average (blended across home and away matches).

    Defense strength: how many goals a team CONCEDES relative to league
    average. Note a GOOD defense has a strength BELOW 1.0 -- they concede
    less than an average team.
    """
    league_avg_home_goals = matches["FTHG"].mean()
    league_avg_away_goals = matches["FTAG"].mean()

    teams = set(matches["HomeTeam"]) | set(matches["AwayTeam"])
    attack, defense = {}, {}

    for team in teams:
        home = matches[matches["HomeTeam"] == team]
        away = matches[matches["AwayTeam"] == team]

        home_attack = home["FTHG"].mean() / league_avg_home_goals if len(home) else 1.0
        away_attack = away["FTAG"].mean() / league_avg_away_goals if len(away) else 1.0
        attack[team] = (home_attack + away_attack) / 2

        # "goals conceded at home" = the AWAY team's goals in home matches
        home_defense = home["FTAG"].mean() / league_avg_away_goals if len(home) else 1.0
        away_defense = away["FTHG"].mean() / league_avg_home_goals if len(away) else 1.0
        defense[team] = (home_defense + away_defense) / 2

    return attack, defense, league_avg_home_goals, league_avg_away_goals


def expected_goals(home_team, away_team, attack, defense, league_avg_home_goals, league_avg_away_goals):
    """
    Expected goals for each side in a specific matchup:
        home_xg = league_avg_home_goals * home_attack * away_defense
        away_xg = league_avg_away_goals * away_attack * home_defense
    """
    home_xg = league_avg_home_goals * attack[home_team] * defense[away_team]
    away_xg = league_avg_away_goals * attack[away_team] * defense[home_team]
    return home_xg, away_xg


def match_outcome_probabilities(home_xg, away_xg, max_goals=MAX_GOALS):
    """
    Build the grid of P(home scores h, away scores a) for h, a in
    0..max_goals, assuming home/away goals are independent Poisson draws.
    Sum the grid into P(home win), P(draw), P(away win).
    """
    p_home_win = p_draw = p_away_win = 0.0

    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = poisson_pmf(h, home_xg) * poisson_pmf(a, away_xg)
            if h > a:
                p_home_win += p
            elif h == a:
                p_draw += p
            else:
                p_away_win += p

    return p_home_win, p_draw, p_away_win


def predicted_result(p_home, p_draw, p_away):
    """Whichever outcome has the highest probability."""
    best = max(p_home, p_draw, p_away)
    if best == p_home:
        return "H"
    elif best == p_draw:
        return "D"
    return "A"


if __name__ == "__main__":
    matches = load_all_seasons()
    recent = matches[matches["Season"].isin(RECENT_SEASONS)]

    attack, defense, avg_home, avg_away = compute_team_strengths(recent)

    # --- Example: a specific upcoming-style matchup ---
    home_team, away_team = "Arsenal", "Chelsea"
    home_xg, away_xg = expected_goals(home_team, away_team, attack, defense, avg_home, avg_away)
    p_home, p_draw, p_away = match_outcome_probabilities(home_xg, away_xg)

    print(f"{home_team} (home) vs {away_team} (away)")
    print(f"Expected goals: {home_team} {home_xg:.2f} - {away_xg:.2f} {away_team}")
    print(f"P(Home win): {p_home:.1%}  P(Draw): {p_draw:.1%}  P(Away win): {p_away:.1%}\n")

    # --- Sanity check: does this beat "always predict home win"? ---
    # NOTE: this checks the model against the SAME matches used to
    # estimate the strengths (in-sample), not a true holdout test -- good
    # enough as a sanity check, not a rigorous accuracy claim.
    correct = 0
    for m in recent.itertuples():
        h_xg, a_xg = expected_goals(m.HomeTeam, m.AwayTeam, attack, defense, avg_home, avg_away)
        ph, pd_, pa = match_outcome_probabilities(h_xg, a_xg)
        if predicted_result(ph, pd_, pa) == m.FTR:
            correct += 1

    accuracy = correct / len(recent)
    home_win_rate = (recent["FTR"] == "H").mean()

    print(f"Poisson model accuracy (all 3 outcomes, in-sample): {accuracy:.1%}")
    print(f"Baseline (always predict home win): {home_win_rate:.1%}")
