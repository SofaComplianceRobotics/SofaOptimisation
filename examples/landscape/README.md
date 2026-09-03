# Analytic landscape harness

A validation/benchmark example whose **score is a classic optimization test
function** of the parameters — so it is genuinely multimodal (and optionally
noisy), unlike the deterministic `cube_drop` toy. Because the global optimum is
**known**, this is where we *measure* whether the optimizer features actually
help, not just check that the plumbing runs.

## What it exercises

| Feature | How this example shows it |
|---|---|
| **IPOP restarts** (`cmaes_restarts`) | multimodal functions (rastrigin, schwefel, himmelblau) where restarts escape local basins — configured with the convergence trigger + warm-start that the benchmark showed actually helps (below) |
| **`restart_on_convergence`** | restart on CMA-ES's *real* convergence, not a best-plateau — the fix that stops restarts hurting |
| **`warm_restarts`** | re-seed a restart from the incumbent, not a random jump |
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
python run.py                       # convergence-triggered warm restarts (records video)
python run.py --variant converged   # stop when restarts stop improving
python run.py --variant plain       # plain CMA-ES baseline (no restarts)
python run.py --variant stall-restart # OLD behavior (cold restart on a plateau) — the harm
python run.py --variant noisy        # noisy objective -> racing
```

Overlay `default` vs `plain` vs `stall-restart` on the dashboard's **Archives**
tab to see the convergence trigger do no harm where the plateau trigger hurt.

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

## What the benchmark taught us about restarts

This harness overturned a plausible-looking assumption and produced a concrete
fix. The story is worth reading before you enable restarts on a real project.

**1. Cold restarts on a best-plateau (the original behavior) *hurt*.** At equal
budget, restarting whenever the best score plateaued for `stall_generations`
made CMA-ES *worse* across the board — because the plateau heuristic fires while
the search is still productive, throwing away a converging run to jump to a
random (usually worse) point. Fair equal-budget comparison, mean best over reps:

| dim / budget | rastrigin | ackley | schwefel |
|---|---|---|---|
| dim 5, 240 evals (plain → cold-restart) | 81 → 74 | **90 → 67** | 48 → 49 |

**2. The fix — restart on real convergence, not a plateau.** With
`restart_on_convergence=True`, a restart only fires once CMA-ES's own
convergence signal trips (step size collapsed) *and* budget remains. A wider
sweep (`plain` vs `cold` vs `warm`, convergence-triggered, mean best over 15
reps) shows the corrected picture:

| regime | rastrigin | ackley | schwefel |
|---|---|---|---|
| **tight budget** (200×dim evals) | +0 / +0 | +0 / +0 | +0 / +0 |
| **large budget** (1000×dim evals), cold / **warm** | +3 to +8 | ~+0.5 | **+4 to +25** |

At tight budgets restarts simply **don't fire** (nothing converged with budget
to spare) → *do no harm*. At large budgets they **help**, biggest on the
deceptive/multimodal cases — and **`warm_restarts=True` (re-seed from the
incumbent) matches or beats cold random restarts** almost everywhere.

**Guidance:** enable restarts (`cmaes_restarts>0`) *with*
`restart_on_convergence=True` and `warm_restarts=True`, and only expect gains on
a **large budget** (thousands of evaluations) over a **multimodal** landscape. On
tight budgets (e.g. a few hundred evals) they are correctly inert.

Reproduce the equal-budget plain / stall-trigger / convergence-trigger table:

```bash
python ../../tests/test_landscape_features.py
```

This is exactly why a real benchmark (not a rigged toy) is worth having: it
caught a shipped feature that was net-negative, and pointed at the fix.

### Alternatives tested and rejected (don't re-litigate without new data)

Each was prototyped on this harness and measured against the shipped
convergence-trigger + warm-IPOP configuration at equal budget:

- **`lr_adapt` (CMA-ES learning-rate adaptation)** — helps schwefel (+7) but
  hurts ackley (−12) and is slightly negative elsewhere. Net-negative.
- **BIPOP (budget-matched large-cold / small-warm regimes)** — ties or loses to
  warm-IPOP on rastrigin at every dimension (e.g. 94.0 vs 97.1 at dim 10); only
  a mixed ±3 picture on schwefel. Not worth the extra restart-state complexity.
- **Flat-popsize warm restarts (no IPOP growth)** — clearly worse on rastrigin
  (92.7 vs 97.1 at dim 10). The growing population is doing real work.
- **Warm-sigma inflation recalibration** — ×1.0/×1.5/×2.0/×3.0 all within ±2
  with no consistent direction; the shipped ×2.0 stands.

The shipped defaults are therefore at the measured optimum of this design
space; a future challenger should beat them *here* before shipping.
