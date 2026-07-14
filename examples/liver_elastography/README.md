# Liver elastography — in-depth study platform (8 params, 3 tests)

Extends [`liver_registration`](../liver_registration/) from a **scalar**
material to a **regional stiffness field**. The 596 tetrahedra of the liver
mesh are partitioned into 6 committed regions (`regions.json`); each region has
its own Young's modulus, passed to `TetrahedronFEMForceField` as a per-element
list. The reference field is homogeneous E=3000 with one **soft lesion**
(region 5, E=1000); the optimizer must localize it from the settled shapes
under **three** patient orientations — inverse elastography.

Unlike the four regression examples (each narrow by design), this one is built
for **depth**: 8 coupled parameters, graded sensitivity, and a load-case count
that is itself an experimental variable.

## Why these 8 parameters

| param | range | reference | role |
|---|---|---|---|
| `young_region_0` | 500 – 12000 | 3000 | **weak**: hugs the fixed nodes, barely deforms — nearly unobservable |
| `young_region_1..4` | 500 – 12000 | 3000 | mid-signal bulk regions |
| `young_region_5` | 500 – 12000 | **1000** | **the lesion**: deepest region, strongest signal — the thing to find |
| `poisson_ratio` | 0.05 – 0.45 | 0.30 | compressibility; reshapes sag differently per orientation |
| `mass_density` | 0.2 – 3.0 | 1.0 | body-force load (forms the exact E/ρ ridge, as in liver_registration) |

The **six regional moduli** turn a scalar identification into a *field*
reconstruction: because region 0 sits against the ligament constraints and
region 5 is deep in the free bulk, the parameter sensitivities span orders of
magnitude — the regime where optimizer features (restarts, warm-start,
covariance adaptation) actually differentiate, on real SOFA at ~1 s per launch.
A **soft** lesion is the standard elastography phantom and was measured to give
~2x the shape signal of a stiff one (a stiff inclusion saturates toward rigid;
a soft one keeps deforming).

## What it validates / enables

- **8-dim search on a real scene** with a real budget (`n_generations=60`,
  `cmaes_restarts=4`, convergence trigger + warm-start — the studied config).
- **Load-case count as a variable**: different orientations load different
  regions, so a soft lesion in region 5 vs region 4 is barely separable from
  supine alone (measured: 86.6 vs 85.1) but separates once orientations are
  combined. Select 1 / 2 / 3 tests on the dashboard (or via the run-config
  env) to study identifiability vs number of tests quantitatively.
- **Sensitivity tooling ground truth**: the dashboard importance analysis
  should rank region 5 high and region 0 low — a known answer to check it
  against.
- A natural **multi-fidelity** axis (settle-step prefix, fewer load cases) for
  the `investigate` branch.

## Targets and provenance

Targets in `targets.json` — **measured 2026-07-14** (Win 11 dev machine, python
runner; reference settles supine 310 / lateral 300 / tilt 215). Measured
anchors (supine): reference **~100**; blind uniform-field defaults (E=5000
everywhere, ν=0.10) **46.2**; correct-elsewhere-but-missed-lesion **86.6**;
lesion-in-the-wrong-region **85.1**. The region partition
(`regions.json`) is committed and deterministic (k-means seed 42); regenerating
it must reproduce the file byte-identically. Re-measure and update after any
*intentional* physics change; never widen to silence a failure.

```bash
python -m pytest tests/test_e2e_elastography.py -q -s   # from repo root, SOFA_ROOT set
```

To regenerate `regions.json`: k-means (seed 42, 50 iters) on the tet centroids
of `liver.msh`, regions renumbered by centroid (x,y,z) order. To regenerate
`targets.json`: run each case with `OPT_LIVER_DUMP=<out.json>` and the reference
field, merge the three dumps with a dated provenance string.

## Run it

```bash
python run.py    # CMA-ES field identification against all three load cases
```

Interactive replay (visible GUI, default params, supine case; freezes when
settled):

```bash
runSofa -l SofaPython3 -g imgui scene.py
```
(with `src/` on `PYTHONPATH`; set `OPT_LIVER_CASE=lateral|tilt` for the others.)
