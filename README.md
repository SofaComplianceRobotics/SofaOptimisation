# sofaopt

**Parallel black-box optimization and a live dashboard for [SOFA](https://www.sofa-framework.org/) simulations — bring your own scene, your own SOFA build.**

sofaopt drives the hard, generic part of optimizing a SOFA simulation:

- samples parameters with **CMA-ES** (default), **GP Bayesian optimization**,
  **TPE**, **Random**, or **NSGA-II** for multi-objective Pareto search (all via Optuna),
- runs many candidates **in parallel** as headless `runSofa` subprocesses (or an
  in-process Python runner with per-trial video recording),
- collects, normalizes, weights and **aggregates scores** across multiple tests,
- shows **live progress, leaderboards, parameter importance and Pareto fronts**
  in a web dashboard, and **archives runs** for later comparison.

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

From a terminal — no `run.py` needed:

```bash
sofaopt path/to/project.py                       # headless (resumes if study.db exists)
sofaopt path/to/project.py --sampler tpe --parallel 8 --gens 50
sofaopt path/to/project.py --prune-mode shadow   # multi-fidelity dry-run
# also: python -m sofaopt path/to/project.py ...
```

Or from Python:

```python
from sofaopt import run_optimization, launch_dashboard
from project import PROJECT

run_optimization(PROJECT)            # headless
# or
launch_dashboard(PROJECT, port=8050) # web UI
```

A run started any of these ways writes one log at `work_dir/logs/optimize.log`,
so the dashboard's log window tails it whichever way the run was launched.

## Optimizer settings

The search is controlled by fields on `SofaOptProject`. The full "which sampler,
which knobs, and why" discussion lives in the
[optimization guide](docs/optimization-guide.md); the surface:

| Field | Default | What it does |
|-------|---------|--------------|
| `sampler` | `"cmaes"` | Search algorithm: `"cmaes"`, `"gp"` (Gaussian-process BO — sample-efficient for expensive sims, <20-D), `"tpe"`, or `"random"`. |
| `n_parallel` | 5 | Population size — candidates per generation, run as parallel `runSofa` processes (≥ 4 for CMA-ES). |
| `n_generations` | 100 | How many generations to run. |
| `cmaes_startup_trials` | auto | **Space-filling startup phase** (see below). `None` auto-sizes from the number of searched params (`resolve_startup_trials()`). |
| `cmaes_sigma0` | 1.0 | Initial spread (std-dev) of the search once CMA-ES begins. |
| `cmaes_with_margin` | `False` | CMA-ES *with Margin* — keeps low-cardinality integer params from stagnating. |
| `seed_sampler` | `"random"` | Startup design for `cmaes`/`gp`: `"sobol"` gives an evenly space-filling (QMC) design; `seed_sampler_seed` re-scrambles it for an independent exploration. |
| `multi_objective` | `False` | Each test becomes a Pareto objective (NSGA-II); gating/weights disabled. |
| `dedup_trials` | `False` | Reuse the recorded score of an identical completed candidate instead of re-simulating (deterministic objectives only). |
| `stall_generations` | 0 | Stop early after N generations without best-score improvement (0 = off). |
| `max_active_sofa_procs` | 12 | Hard cap on concurrent SOFA processes across all tests/repeats. |

**Startup phase.** CMA-ES and GP need evaluated points before their model is
meaningful, so the **first `cmaes_startup_trials` trials are sampled
independently** (uniform random, or a Sobol' space-filling design with
`seed_sampler="sobol"`) within each parameter's bounds; only after that does the
model-based sampler take over. Raise it for more upfront exploration on rugged
landscapes, lower it to converge sooner.

Two related details:
- **Starting point** — CMA-ES is *centered on each `ParamSpec`'s `default`*, not on
  a random point. Frozen params (`min == max`) are excluded from the search.
- Once CMA-ES is active it explores with spread `cmaes_sigma0` around that
  evolving center.

**How a trial is scored.** Per run the scene writes one raw score. The pipeline
then runs in exactly this order: repeats of a test are combined by its
`score_aggregation` (`"mean"` | `"median"` | `"sum"` | `"exponential_coverage"`)
→ the per-test aggregate is normalized by `max_score` (clamped at 1.0) → tests
are combined by `weight` (renormalized over the tests actually counted, e.g.
when a gated test is skipped) → the 0–100 study objective. Every display
(dashboard, videos) reads this recorded score — nothing recomputes its own.

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

The web UI tabs are **Run · Monitor · Parameters · Results · (Pareto) · Archives**:

- **Run** — the single place to launch work. Lists the project's tests once;
  each row has a **Preview** button (opens that scene in an interactive
  `runSofa -g imgui` window — viewer windows die with the dashboard; the headless
  run does not), a **Gate** toggle, and a **weight** slider. Below: the optimizer
  settings (sampler, initial design, parallelism, CMA-ES margin, run-until-converged,
  **pruning** mode), **Start / Pause / Resume** (pausing tree-kills in-flight SOFA
  processes; resuming re-evaluates the interrupted trials, which are excluded from
  rankings), and a shared **log window** with an All / Warnings / Errors filter that
  tails the run — including one started from the `sofaopt` CLI.
- **Monitor** — live per-generation trial grid, restart status, jump-to-running.
- **Parameters** — a table of every parameter (including *frozen* ones), the
  sampled-value bounds heatmap, and (single-objective) fANOVA importance +
  interaction map.
- **Results** — score-over-trials graph (click a point to select a trial), a
  per-trial detail panel with **"Test it"** (re-launch that trial's params in a
  viewer) and **"View recording"** / **"Generate Summary"** video controls, the
  live leaderboard, and the optimization-health panel.
- **Archives** — archive the current run (with a name and notes), restore or
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
  the TPE and GP samplers, the Sobol' startup design, the Python runner with
  recording, multi-objective NSGA-II, and OAT sensitivity analysis.

## Tests

```bash
pytest                           # unit suite — fast, no SOFA needed (the SOFA process is faked)
ruff check src tests examples    # lint gate — clean is part of "done"
```

Two end-to-end tests ([`tests/test_e2e_cube_drop.py`](tests/test_e2e_cube_drop.py))
additionally run one **deterministic real trial** of `examples/cube_drop` — once
through a real `runSofa` (score written, child stops itself, run archives
intact) and once through the in-process Python runner (same score, plus a
recorded `trial.mp4`). They skip automatically unless a SOFA build is reachable
(`SOFA_ROOT` / `RUNSOFA_EXE`) and the dev-only toolkit providing the `sofa`
pytest marker is installed — a plain public checkout stays green.

When to run what:

| Trigger | Tier |
|---|---|
| any commit touching Python | `ruff check` clean + `pytest` (e2e auto-skips without SOFA) |
| changes to the trial contract (runner, scoring, trial_state, archiving) | `pytest` on a machine with SOFA, so the e2e actually executes |
| before a merge / PR / release | full suite with the e2e executing |
| changes to the pruning or finalize machinery | additionally `tests/test_e2e_prune_trunk.py` (~2 min: two full trunk campaigns, shadow vs kill at equal budget) |

## Documentation

- [`docs/porting-guide.md`](docs/porting-guide.md) — step-by-step: plug *your*
  SOFA project in (scene contract, project file, prepare hooks, running).
- [`docs/optimization-guide.md`](docs/optimization-guide.md) — choosing and
  tuning the search: which sampler for which problem, budget/parallelism,
  noise & repeats, scoring/gating, failure semantics, analyzing results.
