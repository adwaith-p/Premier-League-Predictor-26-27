"""
Phase 1: load and clean Premier League match data.

football-data.co.uk gives us a LOT of columns (betting odds, shots, cards...).
For the rating model we only need: who played, when, and the final score.
"""

from pathlib import Path
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# The columns we actually care about, and what they mean:
#   Date     - match date
#   HomeTeam / AwayTeam - team names
#   FTHG / FTAG - Full Time Home/Away Goals
#   FTR      - Full Time Result: 'H' (home win), 'D' (draw), 'A' (away win)
COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]


def load_season(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[COLUMNS].copy()
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
    # season label from filename, e.g. E0_2627.csv -> "2026/27"
    season_code = csv_path.stem.split("_")[1]
    df["Season"] = f"20{season_code[:2]}/{season_code[2:]}"
    return df


def load_all_seasons() -> pd.DataFrame:
    frames = [load_season(p) for p in sorted(RAW_DIR.glob("E0_*.csv"))]
    all_matches = pd.concat(frames, ignore_index=True)
    all_matches = all_matches.sort_values("Date").reset_index(drop=True)
    return all_matches


if __name__ == "__main__":
    matches = load_all_seasons()
    print(f"Loaded {len(matches)} matches across seasons: {matches['Season'].unique()}")
    print("\nMost recent 5 matches:")
    print(matches.tail(5).to_string(index=False))
    print("\nMatches per season:")
    print(matches.groupby("Season").size())
