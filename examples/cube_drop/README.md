# Cube drop — the "does it climb the hill?" demo

A deliberately rigged sanity check. Two parameters — a cube's **size** and
**mass** — and one rule: **the sooner the cube reaches the floor, the better.**
The optimum is obvious by construction, so a healthy optimizer should visibly
drive **both size and mass up**.

It also exercises the **prepare hook**: every trial generates a scaled cube
`.obj` (the "model"), so it covers the full pipeline including geometry
generation. (A project that only tunes scene quantities needs no hook.)

## Why bigger + heavier wins

| Knob | Effect |
|------|--------|
| **size ↑** | Cube spawns at a fixed *center* height, so a bigger cube's bottom face starts lower → reaches the floor sooner. |
| **mass ↑** | A fixed upward force `F` opposes gravity, so net downward accel is `g − F/m`. Heavier → faster. Too light (`m·g < F`) → it floats and scores **0**. |

Score = how early the bottom face crosses `y = 0` (earlier → higher).

## Run

Install once from the repo root: `pip install -e .[dashboard]`, then point at
your SOFA.

**PowerShell** (Windows default shell — use `$env:`, not `export`):

```powershell
$env:SOFA_ROOT   = "C:/path/to/sofa"           # any build with SofaPython3
$env:RUNSOFA_EXE = "$env:SOFA_ROOT/bin/runSofa.exe"

python run.py        # headless - watch best size/mass climb each generation
python dashboard.py  # or the web UI at http://localhost:8050
```

### Variants

```powershell
python run.py --variant default    # CMA-ES / runSofa (this README's default)
python run.py --variant tpe        # TPE Bayesian sampler
python run.py --variant python     # in-process Python runner + trial recording
python run.py --variant pareto     # multi-objective NSGA-II (fall_fast vs compact)
python run.py --variant pareto-python  # multi-objective + Python runner
python sensitivity_test.py         # OAT sensitivity analysis
```

Extras per variant: `default`/`tpe`/`pareto` need only the base install
(`[dashboard]` for the web UI); `python` records videos and therefore needs
`pip install -e .[video]` **and** `ffmpeg` on PATH; `sensitivity_test.py`
needs the base install only, while the dashboard's Importance/Interactions
tab needs `pip install -e .[analysis]`.

**Git Bash / Linux / macOS:**

```bash
export SOFA_ROOT="/path/to/sofa"
export RUNSOFA_EXE="$SOFA_ROOT/bin/runSofa.exe"
python run.py
```

Watch one cube fall interactively (uses param defaults):
`runSofa -l SofaPython3 -g imgui scene.py`

In the headless log, the best trial's `params` should trend toward
`cube_size ≈ 50` and `cube_mass ≈ 50` (the range maxima) within a few
generations.

## Files

| File | Role |
|------|------|
| `project.py` | The `SofaOptProject` + the prepare hook that writes `cube.obj`. |
| `scene.py` | Rigid cube + gravity + buoyancy; scores on contact. |
| `run.py` / `dashboard.py` | Headless / web entry points. |

> Plugin names target SOFA v23.06+. If a `RequiredPlugin` errors, adjust the
> list at the top of `scene.py` for your version.
