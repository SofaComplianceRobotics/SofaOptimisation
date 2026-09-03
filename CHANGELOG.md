# Changelog

Notable changes to `sofaopt`, newest first. Loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); entries are grouped by
feature arc rather than by release, since the package is pre-1.0 and ships from
`dev`.

## [Unreleased]

Work merged since PR #4 (adaptive repeat racing + IPOP CMA-ES restarts).

### Dashboard rework and `sofaopt` CLI — 2026-07-17

The dashboard went from ten tabs to five, and the terminal became a first-class
launch path.

- **Added** a `sofaopt` console command (`sofaopt run`, `sofaopt dashboard`), so
  a study no longer needs a hand-rolled `run.py` per project.
- **Added** a single run log at `<work_dir>/logs/optimize.log` written by *both*
  launch paths — the dashboard can now tail a run started from the terminal.
  It lives beside `runtime/`, not inside it, so archiving can still move
  `runtime/` atomically on Windows.
- **Changed** Scenes and Optimise into one **Run** tab: the test catalog is
  listed once, each row carrying its own scene **Preview** button, with the
  optimizer settings, launch controls and a filterable log view below.
- **Added** a unified **Parameters** tab: every parameter's range, default and
  *frozen* state, the sampled-value bounds heatmap, and fANOVA importance plus
  the pairwise interaction map.
- **Changed** tab names and order for a clearer flow —
  Run → Monitor → Parameters → Results → (Pareto) → Archives.
  (`Performance` became `Results`, `Progress` became `Monitor`.)
- **Added** restart and convergence controls to the Run tab: IPOP restart cap,
  stall-generation trigger, population growth, and the restart-on-convergence /
  warm-restart / trial-dedup switches, forwarded as `OPT_*` overrides with
  matching CLI flags. Controls gray out when their dependencies are unmet.
- **Added** `prune_mode` as a runtime toggle (`OPT_PRUNE_MODE`, a CLI flag and a
  Run-tab dropdown) — multi-fidelity pruning no longer needs a source edit.
- **Added** hover documentation to every Run-tab control, and a simplified
  layout for single-test projects (no gating, weights or pie chart, which are
  meaningless when one test carries the whole objective).
- **Removed** the Playground tab from the core dashboard; it is a self-contained
  teaching sandbox and belongs outside `sofaopt`.
- **Fixed** scene Preview launching `runSofa` with `-g imgui` without ensuring
  the `SofaImGui` plugin was loaded — it exited instantly while reporting a PID.
  Preview output is now captured and an immediate exit is surfaced with its log.
- **Fixed** the run-until-converged pre-flight rejecting `restart_on_convergence`
  configurations that the project validator accepts.

### Dashboard reliability, performance and archives — 2026-07-16 → 07-17

- **Added** an Optimization Health panel and an on-demand search-space
  convergence report.
- **Added** zoom and a range slider to the score graph for candidate selection.
- **Added** richer archives: a convergence/diversity summary per run,
  cross-run comparison, per-archive search-space reports, and preservation of a
  run's name and notes across restore → re-archive.
- **Fixed** SQLite contention under many concurrent workers (WAL journal mode
  and a longer busy timeout).
- **Fixed** the dashboard serving requests one at a time, which blanked live
  views during heavy runs.
- **Fixed** archiving from the dashboard failing because the `study.db` handle
  was still open, and `runtime/` moves being non-atomic — a half-archived run
  could be stranded.
- **Fixed** a browser tab left open across a dashboard restart silently serving
  stale state, and a port already in use failing quietly instead of loudly.
- **Performance**: cached per-tick filesystem work that scaled with trial count.

### Multi-fidelity step pruning — 2026-07-15

Successive halving over the simulation-step prefix, off by default.

- **Added** `SofaOptProject.prune_mode` (`off` / `shadow` / `kill`) with a
  quantile rung rule: at each calibrated rung, candidates are ranked by their
  anytime partial score and the weakest are stopped early. `shadow` runs the
  full decision path but only logs what it *would* kill — a risk-free dry run.
- **Added** `Trial.report_progress()` for scenes to publish an anytime partial
  score, plus `TestSpec.prunable` and `TestSpec.prune_rungs`.
- **Added** a prefix-pruning trace study with offline replay, so rung schedules
  are calibrated against recorded campaigns instead of guessed.
- Measured: 40% of simulation steps saved with zero rank regret on
  `liver_elastography`, 28% on `trunk_control`; a closed-loop end-to-end run
  saved 35% at exact best-score parity with no winning trial killed.

### SOFA examples ladder — 2026-07-14

A graded set of runnable examples, from seconds-per-trial to full study
platforms.

- **Added** `tetra_material` (light — inverse material identification),
  `liver_registration` (medium — multi-test registration),
  `soft_finger` (actuated soft robot — cable reach), and
  `caduceus_settle` (heavy contact).
- **Added** two in-depth study platforms: `liver_elastography` (8 parameters,
  3 tests) and `trunk_control` (8-cable control allocation).
- **Added** an `examples/README.md` index describing the ladder and when to
  reach for each tier.

### Optimizer restart tuning and benchmark harness — 2026-07-13 → 07-14

- **Added** `examples/landscape`, an analytic benchmark harness (no SOFA) for
  validating optimizer features against known optima in seconds.
- **Added** `restart_on_convergence`: trigger IPOP restarts on CMA-ES's own
  convergence signal instead of the best-score plateau. The plateau heuristic
  fires while the search is still productive; measurement showed it made
  restarts net-harmful. Recommended whenever `cmaes_restarts > 0`.
- **Added** `warm_restarts`: re-seed each restart from the incumbent with an
  inflated spread rather than a uniform-random point — measured to match or
  beat cold restarts across the benchmark, most clearly on deceptive
  landscapes.
- **Documented** the rejected alternatives (BIPOP, flat population, sigma A/B)
  and why the measurements ruled them out.
