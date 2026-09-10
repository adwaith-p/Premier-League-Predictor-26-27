"""
Phase 2.5: does the Elo model actually predict anything?

For every match, we already know the pre-match ratings (stored by
run_ratings_history). We ask: "who did the model expect to win?" and
check that against what actually happened.

Draws are excluded from the accuracy number below, because plain Elo
only gives us a single expected-score number, not separate Win/Draw/Loss
probabilities -- that's exactly what Phase 3 (the Poisson model) adds.
For now we're just sanity-checking the ratings themselves.
"""

from elo import HOME_ADVANTAGE, expected_score, run_ratings_history
from data_load import load_all_seasons


def backtest(history):
    decisive = history[history["FTR"] != "D"].copy()

    decisive["HomeExpected"] = expected_score(
        decisive["PreHomeElo"] + HOME_ADVANTAGE, decisive["PreAwayElo"]
    )
    decisive["Predicted"] = decisive["HomeExpected"].apply(lambda e: "H" if e > 0.5 else "A")

    accuracy = (decisive["Predicted"] == decisive["FTR"]).mean()
    home_win_rate = (decisive["FTR"] == "H").mean()

    return accuracy, home_win_rate, len(decisive)


if __name__ == "__main__":
    matches = load_all_seasons()
    _, history = run_ratings_history(matches)

    total_matches = len(history)
    draw_rate = (history["FTR"] == "D").mean()

    accuracy, home_win_rate, n_decisive = backtest(history)

    print(f"Total matches: {total_matches}")
    print(f"Draw rate: {draw_rate:.1%}")
    print(f"Decisive (non-draw) matches: {n_decisive}")
    print(f"\nModel accuracy on decisive matches: {accuracy:.1%}")
    print(f"Baseline (always pick home team): {home_win_rate:.1%}")
