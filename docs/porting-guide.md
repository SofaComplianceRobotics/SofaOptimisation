# Porting guide: optimize *your* SOFA project with sofaopt

This guide takes you from an existing SOFA scene to a fully optimized,
dashboard-driven project. You bring the simulation and the score; sofaopt
brings CMA-ES, parallel execution, aggregation, gating, and the web UI.

It assumes nothing about your robot, your geometry, or your SOFA build.

---

## 0. Mental model

The framework runs one loop:

```
sample params ─▶ (optional) prepare hook ─▶ launch scene.py via runSofa
                                                   │
                  collect score from trial_state.json ◀─ scene writes score
```

So you implement **three things**:

1. a **`SofaOptProject`** — what to tune, what to run, how to reach SOFA;
2. one or more **scenes** that read the sampled params and write a score;
3. *(only for shape optimization)* a **prepare hook** that turns params into an
   asset (e.g. a mesh) before the scene launches.

Everything else is provided.

> The smallest complete example lives in [`examples/cantilever/`](../examples/cantilever).
> Read it alongside this guide — it is ~40 lines of project + scene.

---

## 1. Install

```bash
pip install -e /c/dev/sofaopt[dashboard]   # add ,preview for STL previews
```

sofaopt does **not** install SOFA. It launches whatever `runSofa` you point it
at, so any build with the `SofaPython3` plugin works (emiolabs build, an
official binary release, a source build — all fine).

---

## 2. Make your scene speak the contract

In each scene's `createScene(root)`, read the trial and report a score. This is
the *entire* scene-side API:

```python
from sofaopt.scene import open_trial

def createScene(root):
    trial = open_trial(root)            # params + run metadata from the env
    stiffness = trial.params["stiffness"]
    mesh = trial.env.get("OPT_MESH")    # only if a prepare hook produced one

    # ... build your SOFA graph using those values ...

    root.addObject(MyController(trial=trial))   # see below
    return root
```

A controller (or any code) reports exactly **one** outcome per run:

```python
class MyController(Sofa.Core.Controller):
    def onAnimateEndEvent(self, e):
        # optional: live progress for the dashboard
        trial.write_status({"state": "running",
                            "current_frame": self.step,
                            "total_frames": self.horizon},
                           min_interval=0.2)
        if done:
            trial.write_score(score, reason="held 4.2 s")   # writes + stops
        # or trial.prune("unstable")  to discard the run unscored
```

Key properties:

- `trial.write_score(...)` is **idempotent** and **terminates** the process —
  call it once when the run has produced its result.
- When the scene is opened **outside** the optimizer (e.g. `runSofa scene.py`
  to debug), `trial.params == {}` and `trial.is_optimizing == False`, so guard
  scoring with `if trial.is_optimizing:` and your scene still runs interactively.

### What the scene receives (env, set automatically)

| `trial.` field | from env key | meaning |
|----------------|--------------|---------|
| `params` | `OPT_PARAMS_PATH` (a `params.json`) | this trial's sampled values |
| `run_slot` | `OPT_RUN_SLOT` | which slot to write the score into |
| `test_name` | `OPT_TEST_NAME` | which test is running |
| `gen`/`trial`/`run` | `OPT_GEN`/`OPT_TRIAL`/`OPT_RUN` | identifiers |
| `env["..."]` | prepare-hook env | e.g. `OPT_MESH` |

---

## 3. Describe the project

Create `project.py` (anywhere — your repo, not sofaopt's):

```python
import os
from pathlib import Path
from sofaopt import SofaOptProject, ParamSpec, TestSpec

HERE = Path(__file__).resolve().parent

PROJECT = SofaOptProject(
    name="my_robot",
    work_dir=HERE,                      # runtime/ (trials, db, progress) goes here
    params=[
        ParamSpec("stiffness", "float", 1e3, 1e6, 1e4),
        ParamSpec("n_layers",  "int",   1,   6,   3),
        ParamSpec("use_brace", "bool",  default=True),
    ],
    tests=[
        TestSpec("reach", scene_file=HERE / "scenes/reach.py",
                 max_score=100, weight=2, default_selected=True),
        TestSpec("hold",  scene_file=HERE / "scenes/hold.py",
                 max_score=100, weight=1, gated=True),
    ],
    runsofa_exe=Path(os.environ.get("RUNSOFA_EXE", "runSofa")),
    sofa_env={k: os.environ[k] for k in ("SOFA_ROOT",) if k in os.environ},
    n_parallel=5,
    n_generations=100,
    run_script=HERE / "run.py",         # enables the dashboard's Run button
)
```

### Parameters (`ParamSpec`)

- types: `"float"`, `"int"`, `"bool"`.
- a float/int with `low == high` is **frozen** (reported to the scene but not
  searched) — handy to keep every parameter in one list and toggle which are live.
- already have a dataclass with `metadata={"opt": {...}}` fields? Use
  `sofaopt.param_specs_from_dataclass(instance)`.

### Tests (`TestSpec`)

- `scene_file` is launched via `runSofa`.
- `run_count > 1` repeats a test (e.g. randomized scenarios) and aggregates with
  `score_aggregation` (`"mean"`, `"median"`, `"min"`, `"max"`, `"sum"`,
  `"exponential_coverage"`).
- `max_score` normalizes the test to `[0,1]`; `weight` combines tests.
- `gated=True`: only run this (often expensive) test once an *ungated* test has
  scored above zero for the candidate.
- `relaunchable=True`: advanced — a run that exits non-terminally but set
  `trial.mark_probe_finished()` is relaunched (up to `max_run_relaunches`), for
  iterative probes across several short `runSofa` invocations.

---

## 4. Reaching your SOFA build

Three things must agree (same build): `runsofa_exe`, the plugin tree under
`SOFA_ROOT`, and the SofaPython3 site-packages the scene imports. Set them via
`runsofa_exe` + `sofa_env`:

```python
runsofa_exe=Path(r"C:/sofa/bin/runSofa.exe"),
sofa_env={
    "SOFA_ROOT": r"C:/sofa",
    "PYTHONPATH": r"C:/sofa/lib/python3/site-packages;C:/path/to/your/modules",
},
```

`PYTHONPATH` must let the scene `import sofaopt` and your own modules. With an
editable `pip install -e` of sofaopt into the same interpreter SofaPython3 uses,
`import sofaopt` already works.

---

## 5. Shape optimization? Add a prepare hook

If a parameter changes **geometry**, generate it per trial with a hook that
returns a `TrialPrep`:

```python
from sofaopt import TrialPrep

def prepare(params, trial_dir):
    mesh = trial_dir / "shape.stl"
    build_mesh(params, out=mesh)        # your generator (CadQuery, gmsh, a CLI…)
    return TrialPrep(
        env={"OPT_MESH": str(mesh)},    # injected into the scene process
        cleanup=[mesh],                 # deleted after the trial's runs finish
        preview_image=mesh,             # optional: shown in the dashboard (.stl or .png)
    )

PROJECT = SofaOptProject(..., prepare_trial=prepare)
```

The scene then loads `trial.env["OPT_MESH"]`. Raise inside the hook to hard-fail
a candidate (e.g. invalid geometry). Parameter-only projects (stiffness, mass,
gains, …) need **no hook at all** — the cantilever example has none.

Other optional hooks: `constrain_params(params)->params` (enforce
cross-parameter relationships before use) and
`on_generation_end(gen, paths)` (cross-generation carryover).

---

## 6. Run it

Headless (`run.py`):

```python
from sofaopt import run_optimization
from project import PROJECT
run_optimization(PROJECT)
```

Dashboard (`dashboard.py`): select tests + weights, Run/Stop, live progress,
leaderboard, parameter-bounds heatmap:

```python
from sofaopt import launch_dashboard
from project import PROJECT
launch_dashboard(PROJECT, port=8050)
```

Artifacts land under `work_dir/runtime/` (`trials/gen_XXXX/trial_YY/…`,
`study.db`, `trials/progress.json`).

---

## 7. Checklist

- [ ] `pip install -e sofaopt[dashboard]` into the interpreter SofaPython3 uses.
- [ ] `runsofa_exe` + `sofa_env` point at one consistent SOFA build with SofaPython3.
- [ ] Each scene calls `open_trial(root)` and writes exactly one `write_score`/`prune`.
- [ ] `params` and `tests` declared; `max_score`/`weight` set per test.
- [ ] Shape project only: `prepare_trial` writes the asset + returns its env.
- [ ] `run_script` set so the dashboard can Run.
- [ ] Sanity-check a single scene first: `runSofa -l SofaPython3 -g imgui scenes/reach.py`.

---

## 8. Troubleshooting

- **Scene exits immediately / plugin ABI errors** — `runsofa_exe`, `SOFA_ROOT`
  and the SofaPython3 site-packages are from different builds. Make them one build.
- **`ModuleNotFoundError: sofaopt` inside the scene** — add sofaopt (and your
  modules) to `sofa_env["PYTHONPATH"]`, or install sofaopt into SOFA's Python.
- **All trials hard-fail** — open one scene by hand under `runSofa`; the per-run
  log is `runtime/trials/gen_*/trial_*/sofa_run*.log`.
- **Dashboard Run does nothing** — set `run_script` on the project.
- **A run never ends** — ensure every code path eventually calls `write_score`
  or `prune`; `sofa_realtime_timeout` prunes a stuck run as a backstop.
