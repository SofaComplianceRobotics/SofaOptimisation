"""In-memory tracking of the optimization run: generation, best score, history."""

from __future__ import annotations

from typing import Iterable

from sofaopt.project import TestSpec


class TrialState:
    """Tracks current generation, trial count, best score, and all scores."""

    def __init__(self) -> None:
        self.gen: int = 0
        self.trial_count: int = 0
        self.best_score: float = float("-inf")
        self.best_gen: int = 0
        self.all_scores: list[float] = []
        self.gated_tests_enabled: bool = False
        self._test_max_scores: dict[str, float] = {}
        self._test_aggregations: dict[str, str] = {}

    def load_test_specs(self, tests: Iterable[TestSpec]) -> None:
        """Cache per-test max-score and aggregation method from the selected tests."""
        for spec in tests:
            self._test_max_scores[spec.name] = spec.max_score
            self._test_aggregations[spec.name] = spec.score_aggregation

    @property
    def test_max_scores(self) -> dict[str, float]:
        return self._test_max_scores

    @property
    def test_aggregations(self) -> dict[str, str]:
        return self._test_aggregations

    def record_score(self, score: float) -> None:
        """Record a completed trial's final score and update best tracking."""
        self.all_scores.append(score)
        self.trial_count += 1
        if score > self.best_score:
            self.best_score = score
            self.best_gen = self.gen

    def advance_gen(self) -> None:
        self.gen += 1

    def compute_rolling_stats(self, window: int = 20) -> dict:
        """Rolling performance metrics over the most recent trials."""
        return _compute_rolling_stats(self.all_scores, window)


def _compute_rolling_stats(all_scores: list[float], window: int = 20) -> dict:
    """Rolling performance metrics over recent trials (standalone helper)."""
    recent = all_scores[-window:] if all_scores else []
    valid = [s for s in recent if s != float("-inf")]
    if not valid:
        return {"rolling_avg": None, "rolling_best": None, "window": window}
    return {
        "rolling_avg": round(sum(valid) / len(valid), 4),
        "rolling_best": round(max(valid), 4),
        "window": len(valid),
    }
