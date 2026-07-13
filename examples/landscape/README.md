# Analytic landscape harness

A validation/benchmark example whose **score is a classic optimization test
function** of the parameters — so it is genuinely multimodal (and optionally
noisy), unlike the deterministic `cube_drop` toy. Because the global optimum is
**known**, this is where we *measure* whether the optimizer features actually
help, not just check that the plumbing runs.

## What it exercises

| Feature | How this example shows it |
|---|---|
| **IPOP restarts** (`cmaes_restarts`) | multimodal functions (rastrigin, schwefel, himmelblau) where restarts escape local basins — on by default |
| **`run_until_converged`** | `--variant converged` self-sizes: keeps restarting until restarts stop paying off |
| **Racing** (`run_count_min`) | `--variant noisy` adds score noise, so racing skips wasted repeats |
| **perf / scaling** | edit `DIM`; `SETTLE_STEPS` in `scene.py` dials per-eval cost |

## The functions (`benchmark_functions.py`, pure + unit-tested)

`sphere` (smooth unimodal — the control), `rosenbrock` (curved valley),
`rastrigin` (many regular basins), `ackley` (funnel + fine ripples), `schwefel`
(deceptive — optimum far from centre), `himmelblau` (2-D, four equal optima).
Each maps to a 0–100 score with **100 at the global optimum**.

## Run it

```bash
python run.py                     # rastrigin, IPOP restarts on (python runner, records video)
python run.py --variant converged # stop when restarts stop improving
python run.py --variant plain     # plain CMA-ES baseline (no restarts)
python run.py --variant noisy      # noisy objective -> racing
```

Change the landscape by editing `FUNCTION` / `DIM` / `NOISE` at the top of
`project.py`. Launch the dashboard with `python -m sofaopt.dashboard <project.py>`
(or wire `run_script`) to watch restart markers appear on the convergence curve
and the restart-status panel update live.

Interactive single trial (visible GUI, default params; the marker parks at a
height ∝ its score, then freezes):

```bash
runSofa -l SofaPython3 -g imgui scene.py
```
(with `src/` and this folder on `PYTHONPATH`, and `OPT_LANDSCAPE_FN` set).

## Measuring the advantage

The SOFA-free comparison bench prints plain-CMA-ES vs IPOP across all functions:

```bash
python ../../tests/test_landscape_features.py
```

A representative run (2-D, local start, mean best score over 12 reps):

| function | plain CMA-ES | IPOP restarts | gain |
|---|---|---|---|
| sphere (unimodal) | 99.5 | 99.4 | −0.1 |
| rosenbrock | 97.5 | 99.2 | +1.7 |
| rastrigin | 80.6 | 92.5 | **+11.9** |
| ackley | 59.2 | 52.5 | −6.7 |
| schwefel (deceptive) | 55.4 | 71.0 | **+15.6** |
| himmelblau (4 optima) | 86.1 | 94.4 | **+8.2** |

Restarts clearly help on multimodal/deceptive landscapes, are neutral on the
unimodal control, and can even *hurt* on a single-funnel landscape (ackley) —
useful guidance for *when* to enable them, and the reason a real benchmark (not
a rigged toy) is worth having.
