"""
One-off parser: turn the pasted PL 2026/27 fixture list (data/raw/
schedule_2026_27_raw.txt) into a clean CSV (Date, HomeTeam, AwayTeam)
using the same team-name spellings as football-data.co.uk, so it lines
up with everything else in the pipeline.

Run it whenever the raw schedule file changes (e.g. postponements/
rearrangements get corrected -- see the weekly update script).
"""

import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_load import RAW_DIR, load_all_seasons

RAW_SCHEDULE_PATH = RAW_DIR / "schedule_2026_27_raw.txt"
OUTPUT_PATH = RAW_DIR / "schedule_2026_27.csv"

WEEKDAYS = "Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday"
HEADER_RE = re.compile(rf"^(?:{WEEKDAYS})\s+(\d{{1,2}})\s+([A-Za-z]+)(?:\s+(\d{{4}}))?$")
TIME_PREFIX_RE = re.compile(r"^\d{1,2}:\d{2}\s*(?:GMT)?\s*")
TRAILING_JUNK_RE = re.compile(r"\s*\([^)]*\)\s*$")

# Map every spelling variant seen in the raw schedule to the canonical
# name used in our football-data.co.uk match data.
NAME_MAP = {
    "Ipswich Town": "Ipswich", "Ipswich": "Ipswich",
    "Liverpool": "Liverpool",
    "Newcastle United": "Newcastle", "Newcastle": "Newcastle",
    "AFC Bournemouth": "Bournemouth", "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Sunderland": "Sunderland",
    "Brighton & Hove Albion": "Brighton", "Brighton": "Brighton",
    "Leeds United": "Leeds", "Leeds": "Leeds",
    "Fulham": "Fulham",
    "Crystal Palace": "Crystal Palace",
    "Manchester City": "Man City", "Man City": "Man City",
    "Coventry City": "Coventry", "Coventry": "Coventry",
    "Nottingham Forest": "Nott'm Forest", "Nott'm Forest": "Nott'm Forest",
    "Tottenham Hotspur": "Tottenham", "Tottenham": "Tottenham", "Spurs": "Tottenham",
    "Hull City": "Hull", "Hull": "Hull",
    "Aston Villa": "Aston Villa",
    "Everton": "Everton",
    "Manchester United": "Man United", "Man Utd": "Man United", "Man United": "Man United",
    "Arsenal": "Arsenal",
    "Chelsea": "Chelsea",
}


def normalize_team(name: str) -> str:
    name = name.strip()
    if name not in NAME_MAP:
        raise ValueError(f"Unrecognized team name: {name!r}")
    return NAME_MAP[name]


def parse_schedule(raw_text: str) -> pd.DataFrame:
    current_date = None
    rows = []

    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue

        header_match = HEADER_RE.match(line)
        if header_match:
            day, month_name, year = header_match.groups()
            year = int(year) if year else 2026
            current_date = datetime.strptime(f"{day} {month_name} {year}", "%d %B %Y")
            continue

        if " v " not in line:
            continue  # stray text, not a fixture line

        fixture = TIME_PREFIX_RE.sub("", line)
        fixture = TRAILING_JUNK_RE.sub("", fixture)
        home_raw, away_raw = fixture.split(" v ", 1)

        rows.append({
            "Date": current_date,
            "HomeTeam": normalize_team(home_raw),
            "AwayTeam": normalize_team(away_raw),
        })

    return pd.DataFrame(rows)


def fill_gaps_from_played_matches(schedule: pd.DataFrame) -> pd.DataFrame:
    """
    The pasted schedule text may not cover every gameweek (e.g. it started
    from Matchweek 3, skipping the season-opening two rounds). Anything
    missing that we've actually already played has a real date sitting in
    our own results data -- use that instead of leaving it out.
    """
    all_matches = load_all_seasons()
    season_matches = all_matches[all_matches["Season"] == "2026/27"]

    known_pairs = set(zip(schedule["HomeTeam"], schedule["AwayTeam"]))
    to_add = season_matches[
        ~season_matches.apply(lambda r: (r["HomeTeam"], r["AwayTeam"]) in known_pairs, axis=1)
    ][["Date", "HomeTeam", "AwayTeam"]]

    if len(to_add):
        print(f"Filling in {len(to_add)} fixtures from already-played match results "
              f"(not present in the pasted schedule):")
        print(to_add.to_string(index=False))

    return pd.concat([schedule, to_add], ignore_index=True)


if __name__ == "__main__":
    raw_text = RAW_SCHEDULE_PATH.read_text(encoding="utf-8")
    schedule = parse_schedule(raw_text)

    dupes = schedule[schedule.duplicated(subset=["HomeTeam", "AwayTeam"], keep=False)]
    if len(dupes):
        print("WARNING -- duplicate fixtures found (fix schedule_2026_27_raw.txt):")
        print(dupes.to_string(index=False))

    schedule = fill_gaps_from_played_matches(schedule)
    schedule = schedule.sort_values("Date").reset_index(drop=True)

    print(f"\nTotal fixtures: {len(schedule)}")
    print(f"Unique teams: {len(set(schedule['HomeTeam']) | set(schedule['AwayTeam']))}")
    print(f"Date range: {schedule['Date'].min().date()} to {schedule['Date'].max().date()}")

    # Every team should appear exactly 38 times (19 home + 19 away)
    appearances = pd.concat([schedule["HomeTeam"], schedule["AwayTeam"]]).value_counts()
    bad = appearances[appearances != 38]
    if len(bad):
        print("\nWARNING -- teams without exactly 38 fixtures:")
        print(bad)
    else:
        print("All 20 teams have exactly 38 fixtures -- schedule is complete.")

    # Every team should play every other team exactly twice (once home, once away)
    teams = sorted(set(schedule["HomeTeam"]) | set(schedule["AwayTeam"]))
    full_season = {(h, a) for h in teams for a in teams if h != a}
    known = set(zip(schedule["HomeTeam"], schedule["AwayTeam"]))
    missing = full_season - known
    if missing:
        print(f"\nWARNING -- {len(missing)} fixtures still missing: {sorted(missing)}")
    else:
        print("Every team plays every other team home and away -- no missing fixtures.")

    schedule.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")
