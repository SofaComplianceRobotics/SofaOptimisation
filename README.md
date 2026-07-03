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
pip install -e .[video]     # + trial recording / video generation (pygame, PyOpenGL, ...)
pip install -e .[analysis]  # + fANOVA importance & interaction analysis (scikit-learn)
```

`ffmpeg` (for the `[video]` features) is an external tool: install it separately
and make sure it is on `PATH`.

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

## Optimizer settings

The search is **CMA-ES** (via Optuna). A few fields on `SofaOptProject` control it:

| Field | Default | What it does |
|-------|---------|--------------|
| `n_parallel` | 5 | CMA-ES population size — candidates per generation, run as parallel `runSofa` processes. |
| `n_generations` | 100 | How many generations to run. |
| `cmaes_startup_trials` | auto | **Random startup phase** (see below). `None` auto-sizes from the number of searched params (`resolve_startup_trials()`). |
| `cmaes_sigma0` | 1.0 | Initial spread (std-dev) of the search once CMA-ES begins. |

**Random startup phase.** CMA-ES needs a few evaluated points before its model is
meaningful, so the **first `cmaes_startup_trials` trials are sampled uniformly at
random** within each parameter's bounds; only after that does the CMA-ES
algorithm take over (sampling around an adapting mean). Raise it for more upfront
exploration on rugged landscapes, lower it to converge sooner.

Two related details:
- **Starting point** — CMA-ES is *centered on each `ParamSpec`'s `default`*, not on
  a random point. Frozen params (`min == max`) are excluded from the search.
- Once CMA-ES is active it explores with spread `cmaes_sigma0` around that
  evolving center.

**How a trial is scored.** Per run the scene writes one raw score. The pipeline
then runs in exactly this order: repeats of a test are combined by its
`score_aggregation` (`"mean"` | `"median"` | `"sum"`) → the per-test aggregate is
normalized by `max_score` (clamped at 1.0) → tests are combined by `weight`
(renormalized over the tests actually counted, e.g. when a gated test is
skipped) → the 0–100 study objective. Every display (dashboard, videos) reads
this recorded score — nothing recomputes its own.

**Failure semantics.** A trial whose prepare hook raises, or whose runs all
crash, is reported to the optimizer as a real observation of
`hard_fail_score` (default −3.0) so the sampler learns to avoid that region.
A trial killed by the `sofa_realtime_timeout` backstop is reported as
*pruned* instead — a wall-clock timeout says the run wedged, not that the
parameters were bad. This split is intentional.

## Trial recording (Python runner)

Requires the `[video]` extra and `ffmpeg` on PATH. When using
`runner="python"`, set `record_frames=True` to capture a video of every trial:

```python
PROJECT = SofaOptProject(
    ...
    runner="python",
    record_frames=True,
    record_frame_skip=16,       # capture every 16th step (default)
    record_frame_size=(640, 480),
    record_keep_top_n=15,       # keep recordings for 15 best trials
    record_keep_bottom_n=5,     # keep recordings for 5 worst trials
    record_prune_every_n=20,    # prune every 20 completed trials (mid-run)
)
```

Each trial writes a `trial.mp4` to its directory. After each generation the framework
burns gen/trial/score/params text into the video via ffmpeg. Excess recordings are pruned
mid-run (every `record_prune_every_n` completed trials) to keep disk usage bounded.
At the end of the run a summary video is generated from the top+bottom trials.

## Dashboard

```python
launch_dashboard(PROJECT, port=8050)
```

The web UI provides:

- **Performance graph** — score over trials, click any point to select it
- **"Test it" button** — click a trial in the graph then press "Test it" to launch
  `runSofa -g imgui` with that trial's params and the scene its first run used
  (loads `SofaImGui` automatically). Useful for visually inspecting a candidate.
  Viewer windows are attached to a kill-on-close job, so they never outlive the
  dashboard; the headless optimization run itself is *not* — it survives closing
  the dashboard.
- **"View recording" link** — if the trial has a recorded `trial.mp4`, a direct link
  appears next to the "Test it" button.
- **"Generate Summary" button** — concatenates the top+bottom trial recordings into a
  single summary MP4.
- **Live leaderboard, progress, parameter bounds, and Pareto front** tabs.
- **Archives tab** — archive the current run (with a name and notes), restore or
  delete archives, and **compare runs**: overlaid best-so-far convergence curves
  plus a summary and best-params diff table.

## Archiving runs

Archiving *moves* `work_dir/runtime/` into `work_dir/archives/<timestamp>_<name>/`
(instant, no copy) together with an `archive.json` manifest (settings snapshot,
best score/params, notes) — and thereby resets the workspace. **Starting a fresh
run auto-archives any existing run first**, so a new run can never destroy a
previous one. Restoring moves an archive back to `runtime/` (auto-archiving the
current run first); a restored run can be resumed since its `study.db` is intact.

```python
from sofaopt import archive_run, list_archives, restore_archive, delete_archive

archive_run(PROJECT, name="baseline", notes="before widening bounds")
for info in list_archives(PROJECT):
    print(info.name, info.best_score, info.n_trials)
restore_archive(PROJECT, list_archives(PROJECT)[0].path.name)
```

Comparison (also available in the dashboard's Archives tab) reads each run's
*recorded* scores — the same numbers the studies optimized.

## Examples

A runnable example needing only a SOFA install with SofaPython3:

- [`examples/cube_drop/`](examples/cube_drop/) — Demo: a cube falls and the optimizer learns to make it bigger
  and heavier. Uses a **prepare hook** that generates a scaled cube mesh per
  trial; the optimizer is rewarded for the cube touching the ground sooner, so it
  is incentivized to scale the cube up and make it heavier. Variants demonstrate
  the TPE sampler, the Python runner with recording, multi-objective NSGA-II,
  and OAT sensitivity analysis.

## Porting guide

_(full step-by-step guide — see [`docs/porting-guide.md`](docs/porting-guide.md))_
