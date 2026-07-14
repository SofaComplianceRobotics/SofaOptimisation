# Caduceus wrapped-pose match (heavy real-SOFA tier, contact)

Ported from SOFA's shipped `examples/Demos/caduceus.scn` — the iconic default
demo: a deformable FEM snake drops onto the SOFA pod and wraps itself around
it under frictional contact (collision pipeline + LCP constraint solver every
step). The observation is the snake's pose (all 184 sparse-grid FEM dofs) at a
**fixed horizon** after the drop; the optimizer must recover
(young_modulus, friction_mu, total_mass) that reproduce a **target pose
measured once from the shipped values** — contact-parameter identification
from an observed rest pose.

Deviations from the source `.scn` (documented on purpose): visuals, lights and
camera are dropped (headless; the observation is the mechanical dofs), and
`parallelCollisionDetectionAndFreeMotion` is disabled — measured result: trials
are then **bit-deterministic** (two reference runs, rms diff 0.0), which is
what makes provenance bands possible on a contact scene.

## What it validates

- The full **frictional-contact stack** headless (CollisionPipeline +
  MinProximityIntersection + LCPConstraintSolver + FrictionContactConstraint)
  — the heaviest rung of the ladder (`landscape` → tetra → liver → finger →
  **caduceus**), and the visual showcase for recordings.
- A **fixed-horizon observation protocol**: contact chatter keeps a residual
  max dof speed (~0.5–1.4) long after the pose is stable, so unlike the
  material tiers the score is taken at a fixed step, not behind a settle gate
  (the settle early-exit remains as a cheap-out for truly dead motion).
- A landscape with a real **basin cliff**, not a smooth ridge: below
  `friction_mu ≈ 0.05` the snake slides off the pod entirely (rms 57000 →
  score 0). Restart/convergence logic meets a genuinely multimodal,
  discontinuous objective here.

## Parameters and target

| param | range | reference (target) | search default |
|---|---|---|---|
| `young_modulus` | 5000 – 100000 | **30000** | 60000 |
| `friction_mu` | 0.0 – 0.6 | **0.2** | 0.4 |
| `total_mass` | 0.3 – 3.0 | **1.0** | 1.0 |

Target in `target.json` — **measured 2026-07-14** (Win 11 dev machine, python
runner, fixed-horizon observation at step 600, ~2.5 s wall per trial).
Measured anchors: reference **~100**; search defaults **36.0** (rms 2.04);
box corners 4–20; the sub-mu cliff **0**. Re-measure and update after any
*intentional* physics change; never widen to silence a failure. The e2e
re-runs the anchors through the production launch:

```bash
python -m pytest tests/test_e2e_caduceus.py -q -s   # from the repo root, SOFA_ROOT set
```

To regenerate `target.json` after an intentional change: run the reference
values with `OPT_CADU_DUMP=<out.json>` set, wrap the dumped positions as
`{"positions": [...]}` with a dated provenance string.

## Run it

```bash
python run.py    # CMA-ES contact-parameter identification
```

Interactive replay (visible GUI, default params, stops at the horizon):

```bash
runSofa -l SofaPython3 -g imgui scene.py
```
(with `src/` on `PYTHONPATH`.)
