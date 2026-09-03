# Prefix-pruning study — measuring multi-fidelity gains before building them

Investigation tooling for `docs/design/multi-fidelity.md`: would killing the
bottom of each generation early (successive halving on the **simulation-step
prefix**) save real budget on the study platforms, without killing eventual
winners? This directory measures that from data instead of assuming it.

## How it works

1. The study platforms (`liver_elastography`, `trunk_control`) report an
   **anytime score** every `TRACE_EVERY` steps: `partial_score` in the live
   run status, and — when `OPT_SCORE_TRACE=1` — an on-disk trace
   (`score_trace_run<slot>.json` per run, written just before `write_score`).
2. `run_study.py` drives a real campaign with traces on (isolated work dir,
   `n_parallel=8`, restarts off — see the module docstring for why).
3. `replay.py` replays ANY candidate rung schedule offline from one campaign:
   per-step **rank validity** (Spearman of partial vs final score, per test)
   and **simulated pruning** (steps saved; regret = killed trials that finish
   in the generation's top-mu; generation-winner / study-best kills).

Trace replay evaluates every rung schedule from one recorded campaign — a live
shadow run tests only one configuration per campaign. Live shadow remains the
final pre-flight once the pruning machinery exists.

## Commands

```bash
# from this directory (needs SOFA_ROOT; ~5-15 min each)
python run_study.py trunk                # 16 gens x 8 trials, 1 run each
python run_study.py liver --gens 12      # 3 load-case runs per trial

# re-analyze an existing campaign (any rung schedule, no re-simulation)
python replay.py runs/<name>/runtime/trials --keep-fraction 0.5 --json report.json
```

Campaign outputs land in `runs/` (gitignored). Measured results are recorded
in `docs/design/multi-fidelity.md` §"Measured gains"; keep that section as the
single source of truth rather than duplicating numbers here.

**Recording note**: campaigns run with `record_frames=False` (8 parallel GL
contexts wedge scene startup on Windows — the known preview contention). To
review a trial visually, replay it interactively with the runSofa command in
the example's README, using the params from the trial dir's `params.json`.
