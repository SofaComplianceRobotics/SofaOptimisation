# Cantilever — the minimal sofaopt example

Tunes a soft beam's **Young's modulus** and **Poisson ratio** so its tip sags to
a target deflection under gravity. It is deliberately tiny:

- **2 parameters**, **1 scene**, **1 score**, **no prepare hook** (nothing is
  generated — the scene just reads the sampled params).
- No collisions, no external meshes — runs on any SOFA build with `SofaPython3`.

Use it as a template: copy this folder, then change `project.py` (your params +
tests + SOFA paths) and `scene.py` (your simulation + score).

## Files

| File | What it is |
|------|------------|
| `project.py` | The `SofaOptProject` — the whole contract in one place. |
| `scene.py` | The SOFA scene. Reads `open_trial(root)`, scores at the horizon. |
| `run.py` | Headless optimization: `python run.py`. |
| `dashboard.py` | Web UI with Run/Stop + live charts: `python dashboard.py`. |

## Setup

```bash
pip install -e /c/dev/sofaopt[dashboard]   # the framework
```

Point sofaopt at your SOFA build (any build with SofaPython3).

**PowerShell** (Windows default shell — use `$env:`, not `export`):

```powershell
$env:SOFA_ROOT   = "C:/path/to/sofa"
$env:RUNSOFA_EXE = "$env:SOFA_ROOT/bin/runSofa.exe"
```

**Git Bash / Linux / macOS:**

```bash
export SOFA_ROOT="/path/to/sofa"
export RUNSOFA_EXE="$SOFA_ROOT/bin/runSofa.exe"
```

The scene must be able to `import sofaopt`; with an editable install into the
same Python SofaPython3 uses, it already can.

## Run

Headless:

```bash
python run.py
```

Watch one scene interactively (no optimizer — falls back to defaults):

```bash
runSofa -l SofaPython3 -g qt scene.py
```

Dashboard (select tests/weights, Run/Stop, live progress + leaderboard):

```bash
python dashboard.py        # then open http://localhost:8050
```

## How scoring works

`scene.py` adds a controller that counts steps, and at `HORIZON_STEPS` measures
the free tip's vertical drop and reports:

```python
score = 100 * exp(-|deflection - TARGET| / 4)
trial.write_score(score, reason=...)
```

`write_score` writes into the trial's slot in `trial_state.json` and stops the
process; the optimizer reads it, feeds CMA-ES, and the dashboard shows it.

> **Plugin names**: targets SOFA v23.06+. If a `RequiredPlugin` errors on your
> build, adjust the list at the top of `scene.py` for your version.
