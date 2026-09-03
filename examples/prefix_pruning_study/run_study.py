"""Drive a trace-collecting campaign for the prefix-pruning investigation.

Runs one of the study platforms (``trunk_control`` or ``liver_elastography``)
through the REAL orchestrator with anytime-score traces enabled
(``OPT_SCORE_TRACE=1``), into an isolated work dir, then replays the traces
(see ``replay.py``) and writes ``report.json`` next to the run.

Deliberate deviations from the platforms' shipped configs (all so the traces
measure what the design doc needs):

- ``n_parallel=8`` — the population size pruning is designed for (mu=4 keeps
  CMA-ES ranks meaningful; the shipped 4 is too small to halve).
- restarts OFF — a mid-campaign IPOP restart changes the population size and
  would mix regimes inside one trace set.
- recording OFF — 8 parallel GL contexts wedge scene startup on Windows (the
  known preview/GL contention); replay any single trial interactively with the
  example README's runSofa command instead.

Usage::

    python run_study.py trunk            # ~10 min: 16 gens x 8 trials x ~3 s
    python run_study.py liver --gens 12  # 3 runs per trial (3 load cases)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

import replay  # noqa: E402

from sofaopt import run_optimization  # noqa: E402
from sofaopt.project import SofaOptProject, load_project_file  # noqa: E402

EXAMPLES = {"trunk": "trunk_control", "liver": "liver_elastography"}


def _scene_pythonpath() -> str:
    """PYTHONPATH for scene subprocesses: sofaopt src + SOFA site-packages."""
    parts = [str(REPO / "src")]
    sofa_site = Path(os.environ.get("SOFA_ROOT", "")) / "lib" / "python3" / "site-packages"
    if sofa_site.is_dir():
        parts.append(str(sofa_site))
    existing = os.environ.get("PYTHONPATH", "")
    return os.pathsep.join(parts + ([existing] if existing else []))


def build_project(example: str, out_dir: Path, gens: int, n_parallel: int) -> SofaOptProject:
    project = load_project_file(REPO / "examples" / EXAMPLES[example] / "project.py")
    sofa_env = {
        **{k: str(v) for k, v in project.sofa_env.items()},
        "PYTHONPATH": _scene_pythonpath(),
        "OPT_SCORE_TRACE": "1",
    }
    return dataclasses.replace(
        project,
        work_dir=out_dir,
        sofa_env=sofa_env,
        n_parallel=n_parallel,
        n_generations=gens,
        cmaes_restarts=0,
        restart_on_convergence=False,
        warm_restarts=False,
        stall_generations=0,
        record_frames=False,
        run_script=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example", choices=sorted(EXAMPLES))
    parser.add_argument("--gens", type=int, default=16, help="generations (default 16)")
    parser.add_argument("--n-parallel", type=int, default=8, help="population size (default 8)")
    parser.add_argument("--out", type=Path, default=None,
                        help="work dir (default runs/<example>_<timestamp>)")
    args = parser.parse_args()

    out_dir = args.out or HERE / "runs" / f"{args.example}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    project = build_project(args.example, out_dir, args.gens, args.n_parallel)

    started = time.time()
    run_optimization(project)
    wall = time.time() - started

    report = replay.analyze(project.trials_dir)
    report["example"] = args.example
    report["wall_seconds"] = round(wall, 1)
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[study] campaign wall time {wall:.0f}s")
    print(replay.format_report(report))
    print(f"[study] full report: {report_path}")


if __name__ == "__main__":
    main()
