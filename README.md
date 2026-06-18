# sofaopt

**Parallel CMA-ES optimization and a live dashboard for [SOFA](https://www.sofa-framework.org/) simulations — bring your own scene, your own SOFA build.**

sofaopt drives the hard, generic part of optimizing a SOFA simulation:

- samples parameters with **CMA-ES** (via Optuna),
- runs many candidates **in parallel** as headless `runSofa` subprocesses,
- collects, normalizes, weights and **aggregates scores** across multiple tests,
- shows **live progress, leaderboards and parameter bounds** in a web dashboard.

---

## Install

```bash
pip install -e .            # core optimizer
pip install -e .[dashboard] # + web UI
pip install -e .[preview]   # + STL preview rendering (shape projects)
```

sofaopt does **not** depend on SOFA as a Python package. It launches whatever
`runSofa` you point it at, so any build with the `SofaPython3` plugin works.

## The contract

You write one `project.py` and one or more `scene.py` files.

1. **`SofaOptProject`** — declares parameters (`ParamSpec`), tests
   (`TestSpec` → a `scene.py`), and how to reach SOFA (`runsofa_exe`,
   `sofa_env`). See [`src/sofaopt/project.py`](src/sofaopt/project.py).
2. **Each scene** reads the trial and writes a score:

   ```python
   from sofaopt.scene import open_trial
   trial = open_trial()
   k = trial.params["stiffness"]
   # ... build & run your SOFA scene ...
   trial.write_score(score, reason="...")
   ```

3. *(optional)* a **prepare hook** turns sampled params into an asset (e.g.
   generate a mesh) before the scene launches — for shape optimization.

## Run

```python
from sofaopt import run_optimization, launch_dashboard
from project import PROJECT

run_optimization(PROJECT)            # headless
# or
launch_dashboard(PROJECT, port=8050) # web UI
```

## Examples

A runnable example needing only a SOFA install with SofaPython3:

- [`examples/cube_drop/`](examples/cube_drop/) — Demo: a cube falls and the optimizer learns to make it bigger
  and heavier. Uses a **prepare hook** that generates a scaled cube mesh per
  trial, the optimiser is rewarded by having the cube touch the groun sooner, so is ensentivized to scale the cube and make it heavier

## Porting guide

_(full step-by-step guide — see [`docs/porting-guide.md`](docs/porting-guide.md))_
