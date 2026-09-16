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
# squads that have changed too much to be representative.
RECENT_SEASONS = ["2025/26", "2026/27"]

# Newly promoted teams have almost no data in this window. Shrinking them
# toward "1.0 = league average" was wrong: computed from our own 9 historical
# promoted-team-seasons (2023/24-2025/26: Burnley, Luton, Sheffield United,
# Ipswich, Leicester, Southampton, Burnley, Leeds, Sunderland), promoted
# teams average attack=0.694, defense=1.353 -- meaningfully BELOW average,
# not average. Use that as their shrinkage target instead.
PROMOTED_ATTACK_PRIOR = 0.694
PROMOTED_DEFENSE_PRIOR = 1.353
ESTABLISHED_PRIOR = 1.0

MAX_GOALS = 8  # scorelines beyond this are practically impossible; no need to compute them


def poisson_pmf(k, lam):
    """
    Poisson probability of observing exactly k goals when the expected
    (average) number of goals is `lam`.

        P(X = k) = (lam ** k) * e^(-lam) / k!

    e.g. if a team's expected goals is 1.4, P(exactly 2 goals) = poisson_pmf(2, 1.4)
    """
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


SHRINKAGE_WEIGHT = 6  # treat every team as if it had this many "prior" matches to start


def _shrink(raw_value, n_matches, prior_value=ESTABLISHED_PRIOR, weight=SHRINKAGE_WEIGHT):
    """
    Blend a raw estimate toward `prior_value`, weighted by how much real
    data backs it up.

        shrunk = (n_matches * raw_value + weight * prior_value) / (n_matches + weight)

    A team with 0 matches gets pure `prior_value`. A team with `weight`
    matches gets weighted 50/50 between its own data and the prior. A team
    with 40 matches (a full season) is barely pulled at all.

    Without this, a promoted team that's conceded 0 goals in 3 games gets
    a defense strength of exactly 0.0 -- meaning the model predicts every
    future opponent will score 0 goals against them, forever. That's
    overfitting to a tiny, lucky sample, not a real signal.

    `prior_value` matters as much as the shrinkage itself: shrinking a
    promoted team toward "average" (1.0) is a different, wrong claim than
    shrinking it toward "typical promoted team" (see PROMOTED_*_PRIOR).
    """
    return (n_matches * raw_value + weight * prior_value) / (n_matches + weight)


def _find_promoted_teams(matches):
    """
    A team is "promoted" if it appears in the most recent season in
    `matches` but not in the season immediately before it -- i.e. we have
    no top-flight data on them at all beyond the current season.
    """
    seasons = sorted(matches["Season"].unique())
    if len(seasons) < 2:
        return set()
    latest_season, prior_season = seasons[-1], seasons[-2]
    prior_teams = set(matches[matches["Season"] == prior_season]["HomeTeam"]) | \
        set(matches[matches["Season"] == prior_season]["AwayTeam"])
    latest_teams = set(matches[matches["Season"] == latest_season]["HomeTeam"]) | \
        set(matches[matches["Season"] == latest_season]["AwayTeam"])
    return latest_teams - prior_teams


def compute_team_strengths(matches):
    """
    Attack strength: how many goals a team scores relative to the league
    average (blended across home and away matches).

    Defense strength: how many goals a team CONCEDES relative to league
    average. Note a GOOD defense has a strength BELOW 1.0 -- they concede
    less than an average team.

    Both are shrunk toward a prior based on sample size -- see _shrink().
    Promoted teams shrink toward PROMOTED_*_PRIOR instead of "average",
    since that's what promoted teams actually look like historically.
    """
    league_avg_home_goals = matches["FTHG"].mean()
    league_avg_away_goals = matches["FTAG"].mean()
    promoted_teams = _find_promoted_teams(matches)

    teams = set(matches["HomeTeam"]) | set(matches["AwayTeam"])
    attack, defense = {}, {}

    for team in teams:
        home = matches[matches["HomeTeam"] == team]
        away = matches[matches["AwayTeam"] == team]
        n_matches = len(home) + len(away)
        attack_prior = PROMOTED_ATTACK_PRIOR if team in promoted_teams else ESTABLISHED_PRIOR
        defense_prior = PROMOTED_DEFENSE_PRIOR if team in promoted_teams else ESTABLISHED_PRIOR

        home_attack = home["FTHG"].mean() / league_avg_home_goals if len(home) else attack_prior
        away_attack = away["FTAG"].mean() / league_avg_away_goals if len(away) else attack_prior
        raw_attack = (home_attack + away_attack) / 2
        attack[team] = _shrink(raw_attack, n_matches, prior_value=attack_prior)

        # "goals conceded at home" = the AWAY team's goals in home matches
        home_defense = home["FTAG"].mean() / league_avg_away_goals if len(home) else defense_prior
        away_defense = away["FTHG"].mean() / league_avg_home_goals if len(away) else defense_prior
        raw_defense = (home_defense + away_defense) / 2
        defense[team] = _shrink(raw_defense, n_matches, prior_value=defense_prior)

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


def most_likely_scoreline(home_xg, away_xg, max_goals=MAX_GOALS):
    """The single (home_goals, away_goals) combination with the highest probability."""
    best_score, best_p = (0, 0), 0.0
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = poisson_pmf(h, home_xg) * poisson_pmf(a, away_xg)
            if p > best_p:
                best_p = p
                best_score = (h, a)
    return best_score, best_p


def probability_over_line(home_xg, away_xg, line=2.5, max_goals=MAX_GOALS):
    """
    P(total goals in the match > line). More informative than a single
    "most likely scoreline" -- no individual exact score ever dominates a
    Poisson distribution (see README/chat notes), but "will this match
    have a lot of goals" is a well-behaved, standard summary.
    """
    p_over = 0.0
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            if h + a > line:
                p_over += poisson_pmf(h, home_xg) * poisson_pmf(a, away_xg)
    return p_over


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
