# Porting guide: optimize *your* SOFA project with sofaopt

This guide takes you from an existing SOFA scene to a fully optimized,
dashboard-driven project. You bring the simulation and the score; sofaopt
brings CMA-ES, parallel execution, aggregation, gating, and the web UI.


---

## 0. Mental model

The framework runs one loop:

```
sample params ─▶ (optional) prepare hook ─▶ launch scene.py via runSofa
                                                                      │
        collect score from trial_state.json ◀─ scene writes score  ◀─
```

So you implement **three things**:

1. a **`SofaOptProject`** — what to tune, what to run, how to reach SOFA;
2. one or more **scenes** that read the sampled params and write a score;
3. *(only for shape optimization)* a **prepare hook** that turns params into an
   asset (e.g. a mesh) before the scene launches.

Everything else is provided.

> A complete, runnable example lives in [`examples/cube_drop/`](../examples/cube_drop).
> Read it alongside this guide — it is a small project + scene that exercises the
> whole pipeline (including the prepare hook).

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
        ParamSpec("stiffness", "float", 1e3, 1e6, 1e4), #name, type, min, max, default
        ParamSpec("n_layers",  "int",   1,   6,   3),
        ParamSpec("use_brace", "bool",  default=True),#name, type, default
    ],

    tests=[
        TestSpec("reach", scene_file=HERE / "scenes/reach.py",
                 max_score=100, weight=2, default_selected=True),
        TestSpec("hold",  scene_file=HERE / "scenes/hold.py",
                 max_score=100, weight=1, gated=True),
    ],
    # add more tests with max_score=, weight=, gated=, run_count=, default_selected=

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
- `gated=True`: only run this test once an *ungated* test has
  scored above zero for the candidate.
- `relaunchable=True`: advanced — runs an *iterative probe* across several short
  `runSofa` launches, carrying state between them. See
  [§5b. Iterative probes](#5b-iterative-probes-relaunchable-tests).

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
    )

PROJECT = SofaOptProject(..., prepare_trial=prepare)
```

The scene then loads `trial.env["OPT_MESH"]`. Raise inside the hook to hard-fail
a candidate (e.g. invalid geometry). Parameter-only projects (stiffness, mass,
gains, …) need **no hook at all** — just read `trial.params` in the scene.

Other optional hooks: `constrain_params(params)->params` (enforce
cross-parameter relationships before use) and
`on_generation_end(gen, paths)` (cross-generation carryover).

---

## 5b. Iterative probes (relaunchable tests)

Most tests run once: the scene starts, simulates, writes one score, exits. A
**relaunchable** test instead runs the *same* scene several times in a row,
carrying state forward each time — useful for a search/probe that refines a
value across attempts (e.g. "find the heaviest cube this gripper can still
hold" by trying weights one after another).

Turn it on in two places:

```python
# in project.py
TestSpec("hold_probe", scene_file=..., relaunchable=True)   # this test
PROJECT = SofaOptProject(..., max_run_relaunches=20)        # safety cap on retries
```

**The key fact: each relaunch is a brand-new `runSofa` process, so nothing in
memory survives.** You carry state across launches by writing it to disk. The
scene API gives you two helpers that do exactly that, scoped to this run:

- `trial.load_carry()` → the dict you saved last launch (`{}` on the first).
- `trial.save_carry(d)` → persist a dict for the next launch.

And you end each iteration one of two ways:

- `trial.relaunch(carry={...})` — save state and **go again** (another launch).
- `trial.write_score(score, reason)` — **done**, stop relaunching, report.

So the scene's controller looks like this:

```python
from sofaopt.scene import open_trial

def createScene(root):
    trial = open_trial(root)
    state = trial.load_carry()                 # {} on the first launch
    rung   = state.get("rung", 0)              # which step of the probe we're on
    best   = state.get("best", 0.0)

    weight_to_try = 5.0 + 5.0 * rung           # <-- modify values between relaunches here
    # ... build the scene using weight_to_try (and trial.params, the mesh, etc.) ...
    root.addObject(ProbeController(trial, root, rung, best, weight_to_try))
    return root

class ProbeController(Sofa.Core.Controller):
    def onAnimateEndEvent(self, e):
        if not finished_this_attempt:
            return
        if attempt_succeeded and rung < 9:           # keep climbing
            trial.relaunch(carry={"rung": rung + 1, "best": weight_to_try})
        else:                                        # converged / ran out
            trial.write_score(best, reason=f"max held weight {best}")
```

Mechanics, in one paragraph: when you call `relaunch()`, sofaopt sees the run
exited *without* a final score but *with* a "probe finished" flag, so it starts
the slot again (clearing the flag); your next `open_trial().load_carry()` sees
the values you saved, and you adjust the scene accordingly. If a relaunchable
run ever exits *without* calling `relaunch()` or `write_score()` (i.e. it
crashed), sofaopt fails it rather than retrying a deterministic crash. The
`max_run_relaunches` cap stops a probe that never converges from looping
forever.

> Leave `relaunchable=False` (the default) unless you specifically need this —
> a normal one-shot scene is simpler in every way.

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
