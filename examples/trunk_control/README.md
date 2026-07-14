# Soft trunk control allocation (control study platform, 8 params)

Ported from the SoftRobots plugin's Trunk tutorial
(`examples/tutorials/Trunk/trunk.py` — the elephant-trunk soft robot: a VTK
tetrahedral continuum actuated by **8 cables**, 4 long + 4 short, arranged in
4 directions). A controller ramps all 8 cable displacements production-style
(walked command, no teleports), waits for the trunk to settle, and scores how
closely its **backbone curve** (6 points sampled along the axis) matches a
target backbone measured once from a reference actuation — **control
allocation**: choose 8 actuator commands to shape one continuum.

This is the **control-flavored** in-depth study platform, the companion to
[`liver_elastography`](../liver_elastography/) (the **identification-flavored**
one). Together they cover the two hard 8-parameter regimes an optimizer meets.

Deviations from the source (documented on purpose): visual + collision models
and the AnimationManager are dropped (the observation is the mechanical
backbone); the Lagrangian stack (FreeMotionAnimationLoop + constraint solver)
stays because `CableConstraint` is a Lagrangian actuator. Solver, mesh,
material and the 8-cable geometry are verbatim from `trunk.py`.

## Why these 8 parameters

| param | range | role |
|---|---|---|
| `cable_L0..L3` | 0 – 40 | the 4 **long** cables — pull the whole trunk (long moment arm, wide range) |
| `cable_S0..S3` | 0 – 30 | the 4 **short** cables — bend the base section only (finer, local control) |

Young's modulus is fixed at the tutorial value (450); the **cables are the only
knobs**, so this is a pure control problem, not identification. The set was
chosen for the two properties that make control allocation hard:

- **Redundancy**: 8 cables shape a backbone that lives in far fewer effective
  dimensions, so many command vectors reach the same shape — a large optimum
  set, not a point (the campaign finds a valid allocation, not necessarily the
  reference one).
- **Antagonism**: cables on opposite sides (e.g. L0 vs L2) partly cancel, so
  the objective is non-convex with interacting axes — exactly the covariance
  structure and multi-basin terrain the restart/warm-start machinery targets.

## Target and provenance

Target backbone `TARGET_BACKBONE` (in `scene.py`) — **measured 2026-07-14**
(Win 11 dev machine, python runner) from the reference actuation
(`cable_L0=30, cable_S1=20`, all other cables slack; settled at step 79),
**bit-deterministic** (two runs, backbone rms diff 0.000000). Measured anchors:
reference **~100**; blind start (all cables slack) rms 60.0 → **22.3**; the
opposite actuation (L2, S3 instead of L0, S1) rms 97.6 → **8.7**. Re-measure and
update after any *intentional* physics change; never widen to silence a failure.

```bash
python -m pytest tests/test_e2e_trunk.py -q -s   # from repo root, SOFA_ROOT set
```

To regenerate the target: run the reference actuation with `OPT_TRUNK_DUMP=<out.json>`
set and copy the printed backbone into `TARGET_BACKBONE`.

## Run it

```bash
python run.py    # CMA-ES cable allocation against the target backbone
```

Interactive replay (visible GUI, default = all cables slack; freezes when
settled):

```bash
runSofa -l SofaPython3 -g imgui scene.py
```
(with `src/` and SOFA's `python3/site-packages` on `PYTHONPATH`; the trunk mesh
is found under `$SOFA_ROOT/../plugins/SoftRobots/...` or set `OPT_TRUNK_MESH_DIR`.)
