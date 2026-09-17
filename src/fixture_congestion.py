"""
Phase 10: fixture congestion (rest days across ALL competitions).

The core insight that makes this tractable: to know how tired a team is,
we only need to know WHEN they last played -- not who they played or how
strong that opponent was. That means a Champions League away leg in a
country we have zero league data for still counts toward "how many days
of rest did they get", even though we could never model who they played.

What this deliberately does NOT do: comprehensively scrape every team's
cup/European fixtures. That turned out to be impractical with the
sources readily available (see chat) -- Wikipedia's competition pages
don't convert to a clean fixture list, and guessing at match-report URLs
isn't something to do. So, like team_news.csv, this starts as a small,
manually-maintained log that grows as specific fixtures are noticed,
not an exhaustive automated feed.

The multiplier sizes below are a reasonable heuristic, NOT fit from our
own data the way the promoted-team prior was -- we don't have enough
cross-competition fixture history logged yet to backtest this rigorously.
Treat them as a starting point to revisit once more data accumulates.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_load import RAW_DIR

OTHER_COMPETITIONS_PATH = RAW_DIR.parent / "other_competitions.csv"

REST_DAYS_SHORT_THRESHOLD = 4  # fewer than this many days since the last match counts as "short rest"
SHORT_REST_ATTACK_MULT = 0.95
SHORT_REST_DEFENSE_MULT = 1.05


def load_other_competitions():
    if not OTHER_COMPETITIONS_PATH.exists() or OTHER_COMPETITIONS_PATH.stat().st_size == 0:
        return pd.DataFrame(columns=["Team", "Date", "Competition", "Note"])
    df = pd.read_csv(OTHER_COMPETITIONS_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def rest_days_before(team, match_date, pl_matches, other_matches):
    """
    Days since `team`'s most recent match (Premier League or otherwise)
    strictly before `match_date`. None if we have no prior match on record
    at all (e.g. their very first tracked match).
    """
    pl_dates = pl_matches[(pl_matches["HomeTeam"] == team) | (pl_matches["AwayTeam"] == team)]["Date"]
    other_dates = other_matches[other_matches["Team"] == team]["Date"] if len(other_matches) else pd.Series(dtype="datetime64[ns]")
    all_dates = pd.concat([pl_dates, other_dates])
    prior_dates = all_dates[all_dates < match_date]

    if len(prior_dates) == 0:
        return None
    return (match_date - prior_dates.max()).days


def congestion_multiplier(team, match_date, pl_matches, other_matches):
    """(attack_mult, defense_mult) for `team` playing on `match_date`, given their rest since the last match."""
    rest = rest_days_before(team, match_date, pl_matches, other_matches)
    if rest is not None and rest < REST_DAYS_SHORT_THRESHOLD:
        return SHORT_REST_ATTACK_MULT, SHORT_REST_DEFENSE_MULT
    return 1.0, 1.0


if __name__ == "__main__":
    from data_load import load_all_seasons

    matches = load_all_seasons()
    other = load_other_competitions()

    print(f"Loaded {len(other)} other-competition fixture(s):")
    print(other.to_string(index=False))

    print("\nRest days check for Man United's next few fixtures after the Carabao Cup game:")
    for team, date_str in [("Man United", "2026-09-20"), ("Brighton", "2026-09-19")]:
        d = pd.Timestamp(date_str)
        rest = rest_days_before(team, d, matches, other)
        mult = congestion_multiplier(team, d, matches, other)
        print(f"  {team} on {date_str}: rest_days={rest}  congestion_mult={mult}")
