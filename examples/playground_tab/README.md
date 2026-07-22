# Optimizer Playground — an externally-supplied dashboard tab

An interactive teaching sandbox for **how optimizers explore a search space**:
generate a synthetic 2D landscape, run a real Optuna sampler over it, and watch
the search path animate frame by frame while best-so-far convergence curves
accumulate across runs.

It is also the worked example of the dashboard's **extension point** — the whole
tab is added from this directory, with no change to `sofaopt` itself.

## Run it

```bash
python dashboard.py          # then browse http://localhost:8050
```

The tab bar shows the `cube_drop` project's built-in tabs plus **Playground**,
inserted before Archives.

## Adding it to your own dashboard

```python
from sofaopt import launch_dashboard
from playground import playground_tab

launch_dashboard(PROJECT, extra_tabs=[playground_tab()])
```

`playground_tab()` returns a `sofaopt.DashboardTab`, which carries a `build`
callable for the layout and a `register(app)` hook for the callbacks. Pass
`before="archives"` (or any built-in tab id) to control placement; the default
appends after the built-ins. The mirror knob is `hide_tabs=`, which suppresses a
built-in tab — together they let a project replace a built-in with its own.

## Why it lives here and not in the core package

The playground reads **no project data**: no study, no trials, no parameters —
its landscapes are synthetic. It teaches sampler behaviour rather than reporting
on a run, so it does not belong in a package whose other tabs all describe a
real optimization. Keeping it outside the core also keeps it honest as a test of
the extension point: if `DashboardTab` were not sufficient, this could not work.

Note the deliberate consequence: the algorithms offered here are *not* the same
set the real optimizer offers. This tab is for intuition, not for previewing a
run's configuration.

## Layout

| Path | Role |
|---|---|
| `playground/objectives.py` | Synthetic multi-peak landscape generation and scoring. Pure numpy. |
| `playground/optimizers.py` | Runs real Optuna samplers (CMA-ES, TPE, GP-BO, Random) over a landscape. |
| `playground/tab.py` | Dash layout — controls, landscape heatmap, convergence panel, playback transport. |
| `playground/callbacks.py` | Callbacks. The transport runs clientside so playback never floods the server. |
| `playground/__init__.py` | Exports `playground_tab()`, the `DashboardTab` factory. |
| `dashboard.py` | Launcher wiring the tab into the `cube_drop` dashboard. |

`objectives.py` and `optimizers.py` have no Dash dependency and no `sofaopt`
import, so the engine can be exercised directly:

```python
from playground.objectives import make_landscape, score_at
from playground.optimizers import run_optimization
```
