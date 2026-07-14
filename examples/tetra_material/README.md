# Single-tetra material identification (light real-SOFA tier)

Ported from SOFA's shipped `examples/Demos/oneTetrahedron.scn` — the smallest
possible FEM (one tetrahedron, 4 nodes, near-instant trials). The base triangle
is fixed and the apex sags under gravity; the amount of sag identifies the
material. The optimizer must recover material parameters that reproduce a
**target apex settle position** measured once from reference values — the
classic inverse material-identification problem in miniature.

Deviation from the source `.scn` (documented on purpose): the original fixes the
*apex* and lets the body dangle; we fix the *base* so the settled apex position
is a smooth, monotone function of the material — an objective instead of a demo.
Damping is raised (equilibrium unchanged — damping vanishes at zero velocity) so
every material settles inside the horizon.

## What it validates

- The genuine SOFA path (subprocess launch, settle **self-stop**, the
  `trial_state.json` contract) at negligible per-trial cost — the light tier of
  the example ladder (`landscape` → **tetra** → liver → soft finger → caduceus).
- A real **identifiability ridge**: static sag ≈ load/stiffness, so
  `(young_modulus, total_mass)` trade off — the covariance structure CMA-ES is
  built to learn (watch the Importance/Interactions tab pick it up).
- `--variant noisy` adds sensor-style score noise → **racing**
  (`run_count=5, run_count_min=2`) skips repeats on hopeless candidates.

## Parameters and target

| param | range | reference (target) | search default |
|---|---|---|---|
| `young_modulus` | 2 – 50 | **4.0** | 30.0 |
| `poisson_ratio` | 0.05 – 0.45 | **0.30** | 0.10 |
| `total_mass` | 0.5 – 8 | **5.0** | 5.0 |

Target `TARGET_APEX = [0, 9.457, 0]` — **measured 2026-07-13** (Win 11 dev
machine, python runner, settled at step 339). The e2e test re-runs both anchor
trials through the same production-shaped launch and prints the apex/score:

```bash
python -m pytest tests/test_e2e_tetra.py -q -s   # from the repo root, SOFA_ROOT set
```

Measured landscape anchors (same setup): reference scores **98.9**, the search
defaults **40.8**, the softest corner (E=2, m=6) **20.9**. Re-measure and update
after any *intentional* physics change; never widen to silence a failure.

## Run it

```bash
python run.py                  # CMA-ES identification (a short real campaign)
python run.py --variant noisy  # noisy variant -> racing
```

Interactive replay (visible GUI, default params, freezes when settled):

```bash
runSofa -l SofaPython3 -g imgui scene.py
```
(with `src/` on `PYTHONPATH`.)
