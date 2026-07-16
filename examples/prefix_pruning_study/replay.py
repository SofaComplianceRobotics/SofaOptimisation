"""Offline trace replay for the multi-fidelity (prefix-pruning) investigation.

Reads the anytime-score traces (``score_trace_run*.json``, written by the study
platforms when ``OPT_SCORE_TRACE`` is set) out of a finished campaign's trials
tree and answers the two questions the design doc
(``docs/design/multi-fidelity.md``) poses BEFORE any pruning machinery is built:

1. **Anytime rank validity** — at simulation step k, how well do the partial
   scores rank a generation's candidates against their final scores
   (Spearman rho, per test)?
2. **Simulated pruning** — had the bottom of each generation been killed at
   rung step k (always keeping ``keep`` trials — CMA-ES's mu), how many
   simulation steps would have been saved, and how often would a
   top-``keep``-by-final trial (or the generation winner, or the eventual
   study best) have been killed by mistake?

One recorded campaign replays EVERY candidate rung schedule — cheaper and
stronger than a live shadow run per configuration. A live shadow pass remains
the final pre-flight for the chosen config once the machinery exists.

Trial-level values are the unweighted mean of the trial's per-run values,
which equals the recorded final score for the study platforms (equal test
weights, ``max_score=100``) — the same transform on both sides of every rank
comparison.

The analysis itself (``load_traces``, ``rank_validity``, ``simulate_schedule``,
...) lives in ``sofaopt.pruning_trace`` — promoted out of this script so the
dashboard's Optimization Health panel shares the same implementation instead
of duplicating it. This file is now just the CLI.

Usage::

    python replay.py <work_dir>/runtime/trials [--keep-fraction 0.5]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sofaopt.pruning_trace import analyze, format_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trials_dir", type=Path, help="a campaign's runtime/trials dir")
    parser.add_argument("--keep-fraction", type=float, default=0.5,
                        help="survivor fraction per rung (default 0.5 = CMA-ES mu)")
    parser.add_argument("--json", type=Path, default=None,
                        help="also write the full report to this JSON file")
    args = parser.parse_args()
    report = analyze(args.trials_dir, args.keep_fraction)
    if report is None:
        raise SystemExit(f"No score_trace_run*.json under {args.trials_dir} — "
                         "was the campaign run with OPT_SCORE_TRACE=1?")
    print(format_report(report))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"report written to {args.json}")


if __name__ == "__main__":
    main()
