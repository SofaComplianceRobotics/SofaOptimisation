# Multi-fidelity optimization — investigation & design

*Status: three rounds. Round 1 (2026-07-11, branch `investigate`): design
analysis, no code. Round 2 (2026-07-15, on `feature/benchmark-examples`): the
measurement half — study platforms report an anytime `partial_score`,
`examples/prefix_pruning_study/` replays recorded campaigns against candidate
schedules, §7 has the measured numbers. Round 3 (same day, user go-ahead):
**Phase 1 is implemented** — `core/generation/pruning.py`, `prune_mode`
off/shadow/kill (off by default everywhere), `TestSpec.prunable` +
`prune_rungs`, `Trial.report_progress()`, shadow validated live, and the
closed-loop equal-budget e2e (§8) passing: 35% of startup-generation steps
saved at score parity with zero winner kills.*

SOFA trials are expensive and naturally fidelity-scalable: fewer simulation
steps, coarser mesh, shorter horizon, fewer repeats. A successive-halving
scheme (cheap screen → full evaluation for survivors) can cut campaign cost
substantially. This document maps the idea onto sofaopt's architecture,
recommends what to build, and now grounds the decision in measurements on the
8-parameter study platforms.

---

## 1. The three fidelity axes, and why their *structure* decides the design

| Axis | Who controls it | Structure | Cost to exploit |
|---|---|---|---|
| Simulation horizon (steps) | Scene | **Prefix**: a low-fidelity eval is the first k steps of the full eval | Small — "promote" = *don't kill* the running process |
| Mesh / solver coarseness | Prepare hook + scene | **Restart**: each fidelity is a separate run from scratch | Large — fidelity plumbing through prepare hook, env, scoring; per-project bias correction |
| Repeats / tests per trial | Framework (`run_plan`) | Restart-ish | Covered: `TestSpec.gated` (tests axis) and **racing** (`run_count_min`, repeats axis) |

The prefix structure of the step axis is the decisive observation. For
Hyperband/BOHB-style *restart* fidelities you must launch a candidate at
budget b, get a score, then relaunch survivors at budget ηb — paying process
startup, scene build, and (for shape projects) mesh generation each rung, and
managing cross-fidelity score bias between separate runs. On the step axis
none of that exists: every trial is already running toward full fidelity, and
the scheduler's only action is to **kill the losers early**. Promotion is
free, and the "fidelity gap" reduces to one assumption — that the score-so-far
at step k ranks candidates roughly like the final score does (anytime
validity) — which is exactly what §7 measures.

A structural fact the study platforms added: **the example scenes are
settle-gated** — they self-stop when motion calms, typically far before the
safety horizon (liver: median end ~240 of 500; trunk settles after its command
ramp, well short of 700). Two consequences: (1) rungs expressed as *fractions
of `total_frames`* (the round-1 sketch) are dead — 0.25 × horizon can exceed
the median settle step; rungs must be **absolute steps per test**, calibrated
from data; (2) savings math runs against *measured* end steps, not horizons —
which the trace replay does.

**Recommendation (unchanged from round 1, now with evidence): build
early-kill pruning on the step axis. Defer restart fidelities until pruning
is measured insufficient (Phase 2, likely never).**

## 2. What already exists (inventory, updated 2026-07-15)

- **A polling scheduler with a pruning seam.** `_GenerationFinalizer` scans
  every run every 0.2 s; `_scan_runs` already wall-clock-timeout-prunes via
  `prune_trial()` (kills the process tree, marks slots + trial pruned). A
  rung rule is a second branch at the same seam.
- **A pruned lifecycle end-to-end.** `prune_trial()`, scene-side
  `trial.prune()`, `_is_pruned` → `_tell_pruned` → `study.tell(PRUNED)`,
  dashboard rendering, resume treating `pruned` as terminal.
- **The full signal, on the study platforms.** Both `liver_elastography` and
  `trunk_control` now report `current_frame/total_frames` **and an anytime
  `partial_score`** (refreshed every `TRACE_EVERY=5` steps) through
  `write_status`, plus on-disk score traces under `OPT_SCORE_TRACE=1`.
- **An inert sampler flag waiting for this feature.**
  `CmaEsSampler(consider_pruned_trials=True)` is already set in
  `build_study()`; it only acts once `trial.report()` intermediate values
  exist — rung pruning is what activates it.
- **Sibling budget features, landed since round 1.** Racing
  (`TestSpec.run_count_min`: CI-vs-incumbent early stop on the *repeats*
  axis, measured −40% launches on noisy tetra) and IPOP restarts
  (convergence-triggered + warm). §6 covers the interplay. The
  `SofaOptProject.__post_init__` validation is now split per concern with an
  explicit comment that multi-fidelity pruning adds its own validator.
- **Two-stage screening precedent.** `TestSpec.gated` is multi-fidelity
  *across tests*; step pruning is the same idea *within* a test.

## 3. The anytime score (the once-missing half) — contract

Frame progress says how far a run is, not how good it is. Pruning needs an
**anytime objective**: the score the run would receive if it ended now. Only
the scene can define this. As implemented on the platforms (the pattern for
the porting guide):

```python
# in the scene controller, at TRACE_EVERY-step cadence:
status = {"state": "running", "current_frame": self.step, "total_frames": HORIZON_STEPS}
if self.step % TRACE_EVERY == 0:
    self.partial_score = score_from_rms(rms_to_target(current, TARGET))
if self.partial_score is not None:
    status["partial_score"] = round(self.partial_score, 4)
trial.write_status(status, min_interval=0.2)
```

Observation only — it must never feed back into the physics. Cost on the
platforms: unmeasurable next to the per-step settle detection they already
do. Framework-side contract still to add in Phase 1:
`TestSpec.prunable: bool = False` (a test whose scenes don't report
`partial_score` must never be step-pruned) and a `Trial.report_progress()`
convenience wrapper.

For shape-match objectives the anytime score is the natural "score the
current shape" (both platforms). For end-state or accumulation objectives the
scene reports its best current estimate or stays non-prunable. **Actuated
scenes have a ramp prefix** (trunk: ~60–75 steps of cable walking) during
which the partial score reflects the command schedule more than candidate
quality — rungs must sit after the ramp (§7 shows this in the data).

## 4. The design (phased, updated)

### Phase 0 — contract (scene side DONE on the platforms)

Both study platforms report `partial_score` + traces. Remaining: the
`TestSpec.prunable` flag, `Trial.report_progress()` sugar, porting-guide
section. No framework behavior change.

### Phase 0.5 — measure before building (DONE, `examples/prefix_pruning_study/`)

Round 1 proposed "shadow mode first" — run the real pruning rule live, log
would-kill decisions, kill nothing. The implemented approach is stronger and
cheaper: **record traces once, replay every schedule offline.**

- `run_study.py` drives a real campaign (isolated work dir, `n_parallel=8`,
  restarts off, traces on) through the production orchestrator.
- `replay.py` computes, per candidate rung step: per-test **rank validity**
  (mean per-generation Spearman of partial-vs-final) and **simulated
  pruning** for any `[(step, keep), ...]` schedule — steps saved, kills,
  regrets (killed trials that would have finished in their generation's
  top-µ), generation-winner kills, study-best kills.

One campaign evaluates the entire schedule space; a live shadow run tests one
configuration. Shadow mode is demoted to the final pre-flight of the chosen
config once Phase 1 exists (it additionally validates the *machinery* —
timing, capacity interaction, tell path — which replay cannot).

Replay's known limit: it is **open-loop** — it assumes killing a trial does
not change what CMA-ES samples next (in reality pruned trials feed the
sampler their last partial value via `consider_pruned_trials`). Closed-loop
effect on convergence is measured in the Phase-1 e2e (§8), not here.

### Phase 1 — early-kill pruning in the finalize loop (IMPLEMENTED 2026-07-15)

As shipped (`src/sofaopt/core/generation/pruning.py`; hook: the
`_GenerationFinalizer` calls `pruner.check()` at the top of every settle
pass, next to the wall-clock timeout):

- **Rule**: rung-synchronized quantile within the generation. A rung
  `(step, keep_fraction)` fires once every non-terminal run has reached
  `step` (or finished); candidates rank by `partial_score` (terminal ones by
  their final score, unscored failures at −inf, occupying the bottom without
  being "killable"), and the bottom beyond `ceil(population × keep_fraction)`
  are stopped. A wedged run cannot deadlock a rung — the wall-clock timeout
  eventually makes it terminal.
- **Config as shipped** (leaner than the round-2 sketch): per test,
  `TestSpec.prunable` + `prune_rungs: ((step, keep_fraction), ...)` — the
  keeps live with the steps, so there is no separate `prune_eta` /
  `prune_min_survivors`; project-level, only
  `prune_mode: "off" | "shadow" | "kill"` (default off; every project stays
  behavior-neutral until it opts in). Validators: prunable requires
  `run_count == 1`, ungated, non-relaunchable, ascending steps,
  non-increasing fractions; `prune_mode` requires the v1 single-test shape;
  warnings on `kill` below λ=8 and on a last keep fraction < 0.5 (the CMA-ES
  µ rule). The Optuna ASHA/median rules for TPE/GP are **deferred** — a
  `prune_rule` field arrives with them.
- **Safety ladder**: `"shadow"` runs the full decision path but only logs and
  marks the run slot (`shadow_prune_step`/`shadow_prune_note`); `"kill"` goes
  through the existing `prune_trial()` (process tree killed, slots + trial
  pruned, told PRUNED). Before any rung's kills, every live candidate's
  partial is bridged to Optuna via `trial.report(value, step)`, so a pruned
  trial's last intermediate value is its partial at the rung — this is what
  finally activates `CmaEsSampler(consider_pruned_trials=True)`.
- **Scene sugar**: `Trial.report_progress(score, frame, total)` (porting
  guide §2). A prunable test whose scene reaches a rung without ever
  reporting `partial_score` disables pruning for the generation with a
  warning — never guess.
- `trunk_control` ships the §7-calibrated schedule
  (`prune_rungs=((117, 0.75), (257, 0.5))`, `prune_mode` still off).

**v1 scope** (validator-enforced): single-objective, exactly one test,
prunable, `run_count == 1` — `trunk_control`'s shape. The liver (3 weighted
tests sharing params) needs a trial-level partial aggregate; the replay
already ranks trials by the unweighted mean of per-run partials (= the
recorded final combine for equal weights), and §7 shows it works — the
multi-test extension is data-supported and next in line.

### Phase 2 — restart fidelities (deferred, likely never)

Unchanged from round 1: explicit fidelity ladders (`OPT_FIDELITY`, prepare-
hook fidelity, Hyperband brackets across generations) only if step pruning
proves insufficient *and* per-step cost dominates. Nothing in the round-2
measurements argues for opening this.

## 5. Why this is safe for CMA-ES (and where it isn't)

CMA-ES is rank-based within a generation: the update consumes the ranking of
the λ candidates, weighted toward the top µ. The quantile rule prunes only
bottom-ranked candidates and `prune_min_survivors = µ` keeps every
recombination-weight carrier fully evaluated. Pruned candidates enter via
their last intermediate value; systematic low-bias in partial scores cannot
disturb the update unless it *re-ranks across the cut* — the event the replay
counts directly (regret / winner kills). Measured on liver: zero winner kills
from step 65 on, rank validity ≥ +0.9 from step 45.

Caveats (updated):

- **Small populations.** At `n_parallel = 5`, "kill the bottom half" is 2
  candidates on a noisy signal. The feature wants λ ≥ 8; the trace campaigns
  ran λ = 8, and the design should warn (or refuse `"kill"`) below 8.
  Economics: pruning makes larger populations cheap — the right campaign
  shape shifts toward "wider λ, mostly pruned".
- **Ramp prefixes.** Actuated scenes are uninformative during command
  walking; the earliest rung must clear the ramp (trunk data, §7). This is
  the per-test reason `prune_rung_steps` is per test.
- **Noisy objectives / repeats.** Unchanged: v1 excludes `run_count > 1`;
  when lifting it, prune on partial aggregates and mind racing (§6).
- **GP sampler.** Unchanged: pruned trials are lost to the surrogate; prefer
  `"median"` conservatively there; never tell biased partials as COMPLETE.
- **Startup phase.** Prune during startup too (screening is most valuable
  there); the liver campaign's first ~4 generations were Sobol' startup and
  show the same rank validity.

## 6. Interplay with racing, restarts, and the rest (checked against the code)

The budget features operate on orthogonal axes and compose:

| Feature | Axis | Decision signal | Mechanism |
|---|---|---|---|
| Gating (`TestSpec.gated`) | tests | ungated test scored > 0 | don't launch gated runs |
| Racing (`run_count_min`) | repeats | 95% t-CI upper bound vs study incumbent | don't launch more repeats |
| **Step pruning (this design)** | steps within a run | partial-score rank vs generation peers | kill the running process |

- **Racing** decides whether to *launch* another repeat after runs exit
  (`_race_next_repeat` in `_settle_trial`); pruning decides whether to *keep*
  a run that is still executing (`_scan_runs`). No shared state. One
  interaction to guard when v1's scope widens: a rung-pruned slot reads as
  score `None` → aggregates as 0.0 in `_read_run_results`, which would
  depress the racing CI — in v1 they cannot co-occur (`run_count == 1` means
  no racing), and the composition rule (pruned slots excluded from the racing
  aggregate, or racing evaluated before rungs) is decided when repeats enter
  scope. Racing's CI-vs-incumbent pattern is also the template for a later
  margin-based prune rule (kill only when partial + margin < cut).
- **Restarts (IPOP)**: a restart grows the internal CMA population across
  sofaopt generations; the quantile rule reads λ from the actual generation
  size, so nothing special. Restart scoping (`cma:rN:` attr prefixes) means a
  fresh sampler never sees pre-restart pruned trials — no contamination.
  With `run_until_converged`, pruning's per-generation savings buy more
  restarts inside the same budget — compounding. The trace campaigns
  deliberately ran restarts OFF so one trace set is one regime.
- **Wall-clock timeout prune**: same `prune_trial()` path, different reason.
- **Gating**: v1 scope (one ungated test) never races the gate.
- **`dedup_trials`**: cache indexes COMPLETE only — pruned vectors re-run.
- **Resume**: `pruned` already terminal; RUNNING-only recovery unaffected.
- **Stall tracker / `study.best_value`**: COMPLETE only. Unaffected.
- **Dashboard**: pruned states render today; `partial_score` now flows
  through run slots — a Progress-tab sparkline is a cheap follow-up.
- **Recording**: killed runs already leave valid fragmented MP4s.
- **Relaunchable probes**: excluded from prunable (per-launch frame reset).

## 7. Measured gains (2026-07-15, study platforms, λ=8, keep=µ=4)

Method: real campaigns through the production orchestrator
(`run_study.py`, restarts off, Sobol' startup ≈ first 4 generations), traces
replayed over the full rung grid. "Regret" = a killed trial that would have
finished in its generation's top-µ; "winner/best kills" = killed the
generation winner / the campaign's best trial. Campaign artifacts live in
`examples/prefix_pruning_study/runs/` (gitignored); the numbers of record are
here.

### liver_elastography (12 gens × 8 trials × 3 load-case runs = 288 runs, 55 s wall)

Median end step 238 (of horizon 500). Anytime rank validity (mean
per-generation Spearman, per test): ≈ 0 at step 5, **+0.84…+0.97 by step
45–65, ≥ +0.92 from step 85, ≈ +1.00 from step 145**.

| single rung step | saved | regrets | winner kills | study-best kills |
|---|---|---|---|---|
| 25 | 52.8% | 8 | 0 | 0 |
| 65 | 47.6% | 3 | 0 | 0 |
| 85 | 42.6% | 2 | 0 | 0 |
| **145** | **33.2%** | **0** | **0** | **0** |

Staged schedules (keeps 8→6→4):

| schedule | saved | regrets | winner kills |
|---|---|---|---|
| (45, 6), (105, 4) | 45.1% | 4 | 0 |
| **(65, 6), (145, 4)** | **40.3%** | **0** | **0** |
| (85, 6), (185, 4) | 35.2% | 1 | 0 |

**Headline: a two-rung schedule at steps 65/145 saves 40% of all simulation
steps with zero measured regret** — across 46 kills in 12 generations, no
killed trial would have finished in its generation's top half. That is ≈1.7×
effective budget, or the same budget driving a ~1.7× larger population.

### trunk_control (16 gens × 8 trials × 1 run = 128 runs, 273 s wall)

Median end step 111 (of horizon 700). The **ramp prefix is visible exactly as
predicted**: rank validity +0.23 at step 5 and +0.66 at 33 (mid-ramp, the
partial score is mostly command schedule), +0.90 at 61 (ramp over), then a
long +0.93…0.96 plateau, reaching +1.00 only very late (~590). The redundant,
antagonistic control landscape churns ranks among close candidates well after
settle onset — the platform doing what it was designed for.

| single rung step | saved | regrets | winner kills | study-best kills |
|---|---|---|---|---|
| 61 | 38.1% | 7 | 0 | 0 |
| 89 | 32.6% | 4 | 0 | 0 |
| **145** | **28.0%** | **2** | **0** | **0** |
| 593 (first zero-regret) | 4.3% | 0 | 0 | 0 |

Staged schedules: (117, 6), (257, 4) → 28.0% saved, 2 regrets; more
aggressive staging buys ~5 points of savings for 3–5 extra regrets.

**Headline: ~28–33% saved with 0 winner kills but a floor of ~2–5 mid-rank
regrets** — unlike the liver, trunk has no zero-regret schedule at useful
savings. Killing a would-be top-µ (non-winner) candidate weakens the CMA-ES
recombination rather than breaking it (the pruned trial still contributes its
partial value); whether that measurably slows convergence is precisely what
the closed-loop Phase-1 e2e (§8) must answer at equal budget. Until then,
trunk's calibration is the conservative rung 145 (or staged 117/257).

### Honesty notes

- Replay is open-loop (§4, Phase 0.5): it does not model the sampler seeing
  pruned-trial partials instead of finals. The closed-loop check is the
  Phase-1 e2e.
- These are settle-gated scenes: savings are measured against *actual* end
  steps, and part of the win comes from killing slow-settling candidates
  early — the expensive tail. Fixed-horizon scenes (caduceus) should see
  *larger* fractional savings for the same rung placement, since every
  survivor pays the full horizon.
- Round-1's analytic estimate (1.5–2× per generation) is confirmed at the low
  end by the zero-regret schedules and exceeded by the aggressive ones;
  "several-fold" remains out of reach without rungs the data marks unsafe.

## 8. Verification checklist (all Phase-1 items verified 2026-07-15)

Measured/validated by round 2:

- [x] Anytime score is cheap and behavior-neutral in the scenes (platform
      e2e bands unchanged).
- [x] Anytime rank validity supports rungs well before settle (liver ≥ +0.9
      from step 45; trunk: §7).
- [x] Zero-regret schedules with ~40% savings exist (liver).
- [x] Trial-level mean-of-partials ranks correctly for equal-weight
      multi-test trials (liver replay).

Verified by round 3 (Optuna 4.9 pinned; `tests/test_pruning.py` +
`tests/test_e2e_prune_trunk.py`):

- [x] `trial.report(value, step)` + `study.tell(trial, state=PRUNED)` under
      ask/tell records intermediate values
      (`test_pruned_trial_carries_partial_as_intermediate_value`).
- [x] `CmaEsSampler(consider_pruned_trials=True)` keeps sampling correctly
      across generations containing PRUNED trials with intermediate values
      (tripwire `test_cmaes_keeps_asking_with_pruned_intermediate_values`;
      trials pruned without values — today's timeout prunes — unaffected).
- [x] **Closed-loop equal-budget e2e (Gate 2)**: two trunk campaigns, shadow
      vs kill, seeded Sobol' startup asking identical params in both arms
      (bit-deterministic scene ⇒ the shadow arm is ground truth for every
      trial the kill arm terminated). Measured: 4 trials killed, **35% of
      startup-generation steps saved**, no generation winner killed,
      best-score parity exact, every PRUNED study trial carrying its rung
      partial as an intermediate value, shadow arm killing nothing while
      marking would-kills. Runtime ~103 s — merge-tier, and the permanent
      regression gate for the pruning/finalize machinery.
- [ ] `trial.should_prune()` under ask/tell — deferred with the
      `"asha"`/`"median"` rules for TPE/GP (not in v1).

## 9. Rejected alternatives (round 1, still standing)

- **Optuna `HyperbandPruner` as the only mechanism** — brackets need many
  async trials; protects no CMA-ES rank invariant. ASHA/median stay available
  for non-population samplers.
- **Restart rungs via relaunchable probes** — SOFA can't checkpoint;
  "continuation" would re-simulate from scratch, strictly worse than not
  killing.
- **Mesh-coarseness fidelity first** — highest ceiling, highest cost,
  different-simulation bias; nothing in the measurements argues for it.
- **Fixed scene-side kill thresholds** — no cross-trial information; the
  scene-side `trial.prune()` stays for *physics* aborts.
- *(new, round 2)* **Live shadow mode as the primary validation** —
  superseded by trace replay (one campaign ⇒ all schedules); shadow survives
  as the machinery pre-flight.
