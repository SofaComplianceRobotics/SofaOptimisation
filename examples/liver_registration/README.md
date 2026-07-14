# Liver shape registration (medium real-SOFA tier, multi-test)

Ported from SOFA's shipped `examples/Demos/liver.scn` — the classic liver demo
(181-node tetrahedral organ mesh, corotational FEM, three fixed "ligament"
nodes). The liver sags under gravity; the settled shape of **all mesh nodes**
is the observation. The optimizer must recover the material that reproduces
**target settled shapes measured once from the reference values shipped in the
`.scn`** — surgical-registration-style inverse identification.

The same trial evaluates **two load cases** — the patient orientation changes
the gravity vector — as two weighted `TestSpec`s sharing one scene
(`trial.test_name` selects the case). This is the framework's reference example
for the **multi-test scoring pipeline** (normalize → weight → combine).

Deviations from the source `.scn` (documented on purpose): the collision
pipeline, sphere collision models and OBJ visual mapping are dropped — the
liver interacts with nothing here and the observation is the mechanical dofs.
Solver, damping, mesh, FEM and fixed nodes are unchanged.

## What it validates

- The **multi-test weighted scoring pipeline** with two real SOFA launches per
  trial (the other examples are single-test) — the medium rung of the ladder
  (`landscape` → tetra → **liver** → soft finger → caduceus), ~1 s per launch.
- A **perfectly flat identifiability ridge**: gravity is a body force ∝
  density, so scaling (E, ρ) together leaves the settled shape *exactly*
  unchanged (measured: E=6000, ρ=2 reproduces the E=3000, ρ=1 target to
  rms 0.0000). CMA-ES must learn a degenerate optimum manifold; the second
  orientation pins down `poisson_ratio` but — intentionally — cannot break
  the ridge.

## Parameters and targets

| param | range | reference (target) | search default |
|---|---|---|---|
| `young_modulus` | 500 – 10000 | **3000** | 8000 |
| `poisson_ratio` | 0.05 – 0.45 | **0.30** | 0.10 |
| `mass_density` | 0.2 – 3.0 | **1.0** | 1.0 |

Targets in `targets.json` — **measured 2026-07-14** (Win 11 dev machine, python
runner; reference settles at step 268 supine / 296 lateral). Measured anchors:
reference scores **~100** on both cases; search defaults **37.8** (supine) /
**32.9** (lateral); stiff+light corner 25; soft+heavy corner ~0 (horizon).
Re-measure and update after any *intentional* physics change; never widen to
silence a failure. The e2e re-runs the anchors through the production launch:

```bash
python -m pytest tests/test_e2e_liver.py -q -s   # from the repo root, SOFA_ROOT set
```

To regenerate `targets.json` after an intentional change: run each case with
`OPT_LIVER_DUMP=<out.json>` set and the reference material, then merge the two
dumps (`{"cases": {"supine": [...], "lateral": [...]}}`) with a dated
provenance string.

## Run it

```bash
python run.py    # CMA-ES identification against both load cases
```

Interactive replay (visible GUI, default params, supine case; freezes when
settled):

```bash
runSofa -l SofaPython3 -g imgui scene.py
```
(with `src/` on `PYTHONPATH`; set `OPT_LIVER_CASE=lateral` for the other case.)
