# Soft finger actuated reach (soft-robot real-SOFA tier)

Ported from the SoftRobots plugin's shipped finger part
(`softrobots/parts/finger/finger.py` — same volume mesh, fixing box, cable
geometry and material defaults). A controller **ramps the cable displacement**
to a commanded value at a fixed rate (production-style walked command, no
teleports), waits for the fingertip to reach steady state, and scores how
closely the tip lands on a **target position measured once from reference
values** — actuated-reach calibration mixing control, material and design
parameters.

Deviations from the source (documented on purpose): the three collision meshes
and the contact header are dropped — in the moderate-bend regime commanded
here the finger touches nothing, while the Lagrangian stack
(FreeMotionAnimationLoop + constraint solver) stays because `CableConstraint`
is a Lagrangian actuator. Keyboard jogging is replaced by the ramp controller.

## What it validates

- The first **actuated** rung of the ladder (`landscape` → tetra → liver →
  **finger** → caduceus): drives an actuator the way a production pipeline
  does and measures at the real interface (mechanical dofs), per the
  framework's testing rules (~2 s per trial).
- The **Lagrangian constraint stack** (FreeMotionAnimationLoop +
  BlockGaussSeidelConstraintSolver + CableConstraint) and **third-party plugin
  loading** (SoftRobots plugin, stlib3 prefabs) under headless parallel trials.
- A redundant control/material/design triple — many
  (displacement, stiffness, pull point) combinations reach the same tip —
  the covariance structure CMA-ES is built to learn.

## Parameters and target

| param | range | reference (target) | search default |
|---|---|---|---|
| `young_modulus` | 5000 – 40000 | **18000** | 30000 |
| `cable_displacement` | 0 – 25 | **15** | 5 |
| `pull_point_y` | −5 – 20 | **0** | 10 |

Target `TARGET_TIP = [-94.696, 33.445, -4.488]` — **measured 2026-07-14**
(Win 11 dev machine, python runner, steady at step 255). Measured anchors:
reference scores **~100**, search defaults **37.3**, far corners ~9–11 (tip
span over the box is ~85 mm). The e2e re-runs the anchors through the
production launch:

```bash
python -m pytest tests/test_e2e_finger.py -q -s   # from the repo root, SOFA_ROOT set
```

Re-measure and update after any *intentional* physics change; never widen to
silence a failure.

## Run it

```bash
python run.py    # CMA-ES reach calibration
```

Interactive replay (visible GUI, default params, freezes at steady state):

```bash
runSofa -l SofaPython3 -g imgui scene.py
```
(with `src/` and SOFA's `python3/site-packages` on `PYTHONPATH`.)
