"""
Phase 1: load and clean Premier League match data.

football-data.co.uk gives us a LOT of columns (betting odds, shots, cards...).
For the rating model we only need: who played, when, and the final score.
"""

from pathlib import Path
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
SUPPLEMENT_PATH = RAW_DIR / "results_supplement.csv"

# The columns we actually care about, and what they mean:
#   Date     - match date
#   HomeTeam / AwayTeam - team names
#   FTHG / FTAG - Full Time Home/Away Goals
#   FTR      - Full Time Result: 'H' (home win), 'D' (draw), 'A' (away win)
COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
SUPPLEMENT_COLUMNS = COLUMNS + ["Season", "VerifiedSources", "AddedDate"]


def load_season(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[COLUMNS].copy()
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
    # season label from filename, e.g. E0_2627.csv -> "2026/27"
    season_code = csv_path.stem.split("_")[1]
    df["Season"] = f"20{season_code[:2]}/{season_code[2:]}"
    return df


def load_supplement() -> pd.DataFrame:
    """
    A small, manually/agent-maintained bridge for results that football-
    data.co.uk hasn't published yet -- see fixture_congestion.py and
    team_news.csv for the same "manually maintained, not scraped
    comprehensively" pattern. Rows here are only meant to exist briefly:
    once the primary source catches up, load_all_seasons() automatically
    stops using them (see the dedup there) -- no manual cleanup needed,
    though update_weekly.py prunes them physically too, for tidiness.
    """
    if not SUPPLEMENT_PATH.exists() or SUPPLEMENT_PATH.stat().st_size == 0:
        return pd.DataFrame(columns=SUPPLEMENT_COLUMNS)
    df = pd.read_csv(SUPPLEMENT_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def load_all_seasons() -> pd.DataFrame:
    frames = [load_season(p) for p in sorted(RAW_DIR.glob("E0_*.csv"))]
    all_matches = pd.concat(frames, ignore_index=True)

    supplement = load_supplement()
    if len(supplement):
        # The primary source always wins once it has a fixture -- a
        # supplement row only fills a gap, it never overrides real data.
        # Scoped by (Season, HomeTeam, AwayTeam), NOT just the team pair --
        # the same fixture (e.g. Arsenal v Leeds) recurs across different
        # seasons, so team-pair-only matching would wrongly treat a
        # current-season gap as already covered by an old season's match.
        known_triples = set(zip(all_matches["Season"], all_matches["HomeTeam"], all_matches["AwayTeam"]))
        still_needed = supplement[
            ~supplement.apply(lambda r: (r["Season"], r["HomeTeam"], r["AwayTeam"]) in known_triples, axis=1)
        ]
        if len(still_needed):
            all_matches = pd.concat([all_matches, still_needed[COLUMNS + ["Season"]]], ignore_index=True)

    all_matches = all_matches.sort_values("Date").reset_index(drop=True)
    return all_matches


def prune_superseded_supplement_rows():
    """
    Physically remove supplement rows the primary source now covers for
    real. Not required for correctness (load_all_seasons already ignores
    superseded rows), just keeps the file from accumulating dead entries
    forever. Safe to call every week.
    """
    supplement = load_supplement()
    if not len(supplement):
        return 0

    primary_frames = [load_season(p) for p in sorted(RAW_DIR.glob("E0_*.csv"))]
    primary = pd.concat(primary_frames, ignore_index=True)
    known_triples = set(zip(primary["Season"], primary["HomeTeam"], primary["AwayTeam"]))

    still_needed = supplement[
        ~supplement.apply(lambda r: (r["Season"], r["HomeTeam"], r["AwayTeam"]) in known_triples, axis=1)
    ]
    removed = len(supplement) - len(still_needed)
    if removed:
        still_needed.to_csv(SUPPLEMENT_PATH, index=False)
    return removed


if __name__ == "__main__":
    matches = load_all_seasons()
    print(f"Loaded {len(matches)} matches across seasons: {matches['Season'].unique()}")
    print("\nMost recent 5 matches:")
    print(matches.tail(5).to_string(index=False))
    print("\nMatches per season:")
    print(matches.groupby("Season").size())
