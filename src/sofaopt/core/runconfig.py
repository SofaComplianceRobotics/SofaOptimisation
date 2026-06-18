"""Per-run configuration assembled from a project + UI/CLI selections.

This replaces the old module-global ``config.py``: instead of import-time
constants, the orchestrator builds one :class:`RunConfig` and threads it
through the loop. It captures *which* tests are selected, their weights, the
flattened run plan, and the derived runtime paths — everything the generic
loop needs that depends on a specific project + selection.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from sofaopt.core import envkeys
from sofaopt.project import ParamSpec, SofaOptProject, TestSpec


@dataclass(frozen=True)
class RunConfig:
    project: SofaOptProject
    selected_tests: tuple[TestSpec, ...]
    test_weights: dict[str, float]  # fractions, sum to 1.0
    gated_test_names: tuple[str, ...]

    # ---- constructors -----------------------------------------------------
    @classmethod
    def from_project(
        cls,
        project: SofaOptProject,
        *,
        selected_names: Sequence[str] | None = None,
        weights: dict[str, float] | None = None,
        gated_names: Sequence[str] | None = None,
    ) -> "RunConfig":
        """Build a RunConfig, defaulting to all of the project's tests.

        Args:
            selected_names: Subset of test names to run (default: all).
            weights: Per-test weight fractions; normalized to sum to 1.0
                (default: each test's declared ``weight``).
            gated_names: Tests to gate (default: each test's ``gated`` flag).
        """
        if selected_names:
            chosen = tuple(project.test(n) for n in selected_names)
        else:
            chosen = tuple(project.tests)

        if weights:
            raw = {t.name: float(weights.get(t.name, 0.0)) for t in chosen}
        else:
            raw = {t.name: float(t.weight) for t in chosen}
        total = sum(raw.values())
        if total <= 0:
            norm = {t.name: 1.0 / len(chosen) for t in chosen}
        else:
            norm = {name: w / total for name, w in raw.items()}

        if gated_names is not None:
            gated = tuple(n for n in gated_names if n in {t.name for t in chosen})
        else:
            gated = tuple(t.name for t in chosen if t.gated)

        return cls(
            project=project,
            selected_tests=chosen,
            test_weights=norm,
            gated_test_names=gated,
        )

    @classmethod
    def from_env(cls, project: SofaOptProject) -> "RunConfig":
        """Build a RunConfig from selection forwarded via environment variables.

        Used when the dashboard launches the optimizer subprocess: it sets
        ``OPT_SELECTED_TESTS`` / ``OPT_TEST_WEIGHTS`` / ``OPT_GATED_TESTS``.
        """
        raw_sel = os.environ.get(envkeys.SELECTED_TESTS, "").strip()
        selected = [s for s in raw_sel.split(",") if s] or None

        weights = None
        raw_w = os.environ.get(envkeys.TEST_WEIGHTS, "").strip()
        if raw_w:
            try:
                parsed = json.loads(raw_w)
                if isinstance(parsed, dict):
                    weights = {k: float(v) for k, v in parsed.items()}
            except Exception:
                weights = None

        raw_g = os.environ.get(envkeys.GATED_TESTS, "").strip()
        gated = [s for s in raw_g.split(",") if s] if raw_g else None

        return cls.from_project(
            project,
            selected_names=selected,
            weights=weights,
            gated_names=gated,
        )

    # ---- derived views ----------------------------------------------------
    @property
    def selected_names(self) -> tuple[str, ...]:
        return tuple(t.name for t in self.selected_tests)

    @property
    def param_specs(self) -> list[dict]:
        return [p.to_dict() for p in self.project.params]

    @property
    def run_plan(self) -> tuple[tuple[str, int, int], ...]:
        return tuple(
            (t.name, run_index, t.run_count)
            for t in self.selected_tests
            for run_index in range(1, t.run_count + 1)
        )

    @property
    def n_repeats(self) -> int:
        return len(self.run_plan)

    @property
    def test_max_scores(self) -> dict[str, float]:
        return {t.name: t.max_score for t in self.selected_tests}

    @property
    def test_aggregations(self) -> dict[str, str]:
        return {t.name: t.score_aggregation for t in self.selected_tests}

    # ---- environment for scene subprocesses -------------------------------
    def base_scene_env(self) -> dict[str, str]:
        """Project env + forwarded selection. Per-run keys are added by the runner."""
        env = self.project.scene_env()
        env[envkeys.SELECTED_TESTS] = ",".join(self.selected_names)
        env[envkeys.TEST_WEIGHTS] = json.dumps(
            {name: round(frac * 100) for name, frac in self.test_weights.items()}
        )
        if self.gated_test_names:
            env[envkeys.GATED_TESTS] = ",".join(self.gated_test_names)
        return env
