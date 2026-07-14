# sofaopt examples — the validation ladder

Two groups. The **regression ladder** (top table) is six examples ordered by
cost, each adding exactly one new demand on the framework, so a regression
shows up at the cheapest rung that exercises it. The **study platforms** (second
table) are two deliberately harder 8-parameter examples for in-depth optimizer
research rather than quick regression. Every objective is a **measured target
with recorded provenance** (never a guessed number), and every scene
**self-stops** (settle detection or a fixed observation horizon) — no wasted
steps.

## Regression ladder

| example | source | scenario | params | first to validate | ~trial |
|---|---|---|---|---|---|
| [`landscape`](landscape/) | analytic functions | known-optimum benchmarks (sphere, rastrigin, ackley, ...) | 2–10 floats | optimizer features in isolation (restarts, racing, convergence) — no SOFA | ms |
| [`cube_drop`](cube_drop/) | original | cube falls, height score | 3 floats | the runSofa runner path end-to-end | ~s |
| [`tetra_material`](tetra_material/) | `Demos/oneTetrahedron.scn` | inverse material ID from a target apex settle | E, ν, mass | python-runner SOFA plumbing at minimal cost; racing under noise (`--variant noisy`) | ~0.5 s |
| [`liver_registration`](liver_registration/) | `Demos/liver.scn` | settled-shape registration under **two gravity load cases** | E, ν, density | the multi-test weighted scoring pipeline (2 launches/trial) | ~1 s ×2 |
| [`soft_finger`](soft_finger/) | SoftRobots `parts/finger` | **actuated** cable reach: ramped command → target tip | E, cable displacement, pull point | actuation + Lagrangian constraints + third-party plugins (SoftRobots, stlib3) | ~2 s |
| [`caduceus_settle`](caduceus_settle/) | `Demos/caduceus.scn` | wrapped-pose match under frictional **contact** | E, friction μ, mass | collision + LCP contact stack; a basin *cliff* (μ→0 slides off); fixed-horizon observation | ~2.5 s |

Landscape structure is deliberate and documented per example: the material
tiers carry (E, load) identifiability *ridges* (exact on the liver), the
finger a redundant control/material/design triple, the caduceus a
discontinuous slide-off basin — between them they cover the geometries an
optimizer meets in practice.

## Study platforms (8 params, for in-depth research)

| example | source | scenario | params | studies | ~trial |
|---|---|---|---|---|---|
| [`liver_elastography`](liver_elastography/) | `Demos/liver.scn` + regional field | **identification**: localize a soft lesion in a 6-region stiffness field, 3 load cases | 6 regional E + ν + density | graded sensitivity, load-case-count vs identifiability, sensitivity-tooling ground truth | ~1 s ×3 |
| [`trunk_control`](trunk_control/) | SoftRobots Trunk tutorial | **control allocation**: 8 cables shape a soft continuum to a target backbone | 8 cable displacements | redundancy + antagonism, non-convex control landscape, restart/covariance value | ~3 s |

The two platforms are complementary halves of the hard 8-D regime:
identification (a hidden parameter field to recover, with an active E/ρ ridge
that makes lesion localization genuinely ambiguous — a measured 40-generation
run mislocated the lesion to the adjacent region while scoring 91) vs control
(a redundant, antagonistic actuator map with many equivalent optima). They are
where optimizer features actually differentiate, on real SOFA scenes cheap
enough for repeated-seed studies.

Each example directory has a README with the parameter table, measured
anchors + provenance, the regeneration recipe for its target, and the
interactive `runSofa` replay command. The `tests/test_e2e_<name>.py` files
re-run the anchor trials through the production launch primitives and assert
the provenance bands.

## Feature studies

[`prefix_pruning_study/`](prefix_pruning_study/) drives trace-collecting
campaigns on the study platforms and replays candidate multi-fidelity pruning
schedules offline (rank validity + simulated savings/regret) — the measurement
half of `docs/design/multi-fidelity.md`. Both platforms report an anytime
`partial_score` in their live status and, under `OPT_SCORE_TRACE=1`, write
per-run score traces for it.
