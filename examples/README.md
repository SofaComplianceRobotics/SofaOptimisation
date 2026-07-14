# sofaopt examples — the validation ladder

Five examples, ordered by cost. Each rung adds exactly one new demand on the
framework, so a regression shows up at the cheapest rung that exercises it.
Every objective is a **measured target with recorded provenance** (never a
guessed number), and every scene **self-stops** (settle detection or a fixed
observation horizon) — no wasted steps.

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

Each example directory has a README with the parameter table, measured
anchors + provenance, the regeneration recipe for its target, and the
interactive `runSofa` replay command. The `tests/test_e2e_<name>.py` files
re-run the anchor trials through the production launch primitives and assert
the provenance bands.
