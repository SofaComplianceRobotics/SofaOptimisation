# Optimization guide: choosing and tuning the search

The [porting guide](porting-guide.md) gets your project *running*; this guide
helps you make the search *good*: which sampler fits your problem, how to size
the budget, how to handle noisy scores, and how to read the results. Everything
here is a field on `SofaOptProject` (see
[`src/sofaopt/project.py`](../src/sofaopt/project.py)) unless said otherwise.

---

## 1. Which sampler? — decision table

| Your situation | Recommended setup |
|---|---|
| Continuous parameters (stiffness, gains, geometry), the common case | `sampler="cmaes"` (default) |
| Several **integer** params with few values each (2–8 layers, hole counts) | `sampler="cmaes", cmaes_with_margin=True` |
| **Expensive** simulations (minutes each), small budget, ≲ 20 searched params | `sampler="gp"` |
| Mixed / conditional spaces, ≤ ~5 params, cheap-ish evaluations | `sampler="tpe"` |
| Two+ objectives that genuinely compete and you can't pick weights | `multi_objective=True` (NSGA-II) |
| Smoke test, baseline for comparison, or pure space exploration | `sampler="random"` |
| Not sure the score even responds to your params yet | run an [OAT sensitivity sweep](#8-understanding-the-results) first |

Rules of thumb behind the table:

- **Budget per dimension** decides more than anything else. If you can afford
  hundreds of trials per searched parameter, CMA-ES will out-refine everything;
  if you can afford tens, a model-based sampler (`gp`) extracts more per
  simulation.
- **Frozen params don't count** — a `ParamSpec` with `low == high` is reported
  to the scene but excluded from the search, so the *searched* dimensionality
  `d` is what matters everywhere below.

## 2. How each sampler works (and when it breaks)

### CMA-ES — `sampler="cmaes"` (default)

**Intuition.** CMA-ES maintains a multivariate Gaussian "searchlight" over
parameter space. Each generation it draws `n_parallel` candidates from it,
scores them, and shifts/reshapes the Gaussian toward where the good ones were —
the covariance matrix learns the local geometry of the landscape (long valleys,
couplings between parameters) without any gradients.

**Formal idea.** Covariance Matrix Adaptation Evolution Strategy: rank-based
updates of mean, step size and covariance from the population — invariant to
monotone score transformations and to rotations of the space, which is why it
is a robust default for physical-simulation objectives.

Practical constraints, all enforced or wired by the framework:

- Population = concurrency: `n_parallel` is λ, the candidates per generation,
  each a parallel SOFA process. **Must be ≥ 4** (construction raises otherwise)
  — the covariance update is ill-defined below that.
- The search **starts at your `ParamSpec` defaults** (`x0`), not at a random
  point, with initial spread `cmaes_sigma0` (§4). Set defaults to your
  best-known design.
- Boolean params are not part of the CMA-ES Gaussian — they are sampled by the
  independent/startup sampler throughout.
- Pruned trials are considered by the sampler (`consider_pruned_trials=True`),
  so a run killed by the timeout backstop doesn't stall the update.

**When it breaks:** very small budgets (it needs generations to adapt),
low-cardinality integer parameters (see the margin variant next), and
multimodal landscapes — a converged searchlight is stuck in one basin. For
the latter, pair `stall_generations` with `cmaes_restarts` (§5): stagnation
then triggers an IPOP restart (fresh CMA-ES, doubled population) instead of
ending the run.

### CMA-ES with Margin — `cmaes_with_margin=True`

**Intuition.** With a naïve continuous relaxation, an integer parameter whose
range maps to few distinct values can trap CMA-ES: the Gaussian shrinks until
every sample rounds to the *same* integer, and that dimension stops being
searched at all. The *margin* correction (Hamano et al., GECCO 2022) keeps a
minimum probability of sampling the neighboring integers, so discrete
dimensions keep moving.

Use it whenever integer parameters with ≲ 10 distinct values matter to your
search; it only applies when `sampler="cmaes"`.

### Gaussian-process Bayesian optimization — `sampler="gp"`

**Intuition.** Fit a probabilistic surrogate (a Gaussian process) to every
(params → score) pair seen so far, then pick the next candidate by expected
improvement — balancing "promising" against "unexplored". Because the
surrogate interpolates *all* history, each new simulation is chosen with
maximum information; that's why it wins when simulations are expensive and the
budget is small.

**When it breaks.** GP fitting cost grows steeply with the number of trials and
its predictive power fades in high dimensions — beyond ~20 searched params the
surrogate stops being better than CMA-ES's population statistics. The startup
design (§3) is auto-sized to `10·d` for GP (a surrogate needs a real coverage
map before its predictions mean anything). Unlike CMA-ES it does not take an
`x0` — your defaults enter only through the startup design's coverage.

### TPE — `sampler="tpe"`

**Intuition.** Tree-structured Parzen Estimator: instead of modeling the score
surface, model *where the good trials live* vs *where the bad ones live* as two
densities, and sample where the ratio good/bad is highest. Handles mixed and
conditional spaces naturally and is cheap per suggestion.

Optuna's TPE defaults apply as-is (its own internal random startup of ~10
trials; `cmaes_startup_trials` and `seed_sampler` do **not** affect TPE).
Often converges faster than CMA-ES with ≤ ~5 parameters; in more dimensions
its independent-density assumption ignores parameter couplings that CMA-ES's
covariance would capture.

### Random — `sampler="random"`

Uniform sampling forever. Two legitimate uses: a smoke test of the pipeline,
and a baseline — if your tuned sampler doesn't beat random over the same
budget, the landscape is flat (or the score is broken) and no sampler choice
will save you.

### NSGA-II — `multi_objective=True`

**Intuition.** When two tests genuinely compete (fast *vs* compact, strong *vs*
light) any fixed weighting bakes in a trade-off you haven't seen yet. NSGA-II
instead evolves a population toward the **Pareto front** — the set of
candidates where improving one objective must cost another — and returns the
whole front so you choose the trade-off *after* seeing it.

Mechanics in sofaopt: each selected `TestSpec` becomes one objective with its
own `direction` (`"maximize"`/`"minimize"`); population size is `n_parallel`;
**gating and score weighting are disabled** (there is no single combined score
to gate on); `dedup_trials` and `stall_generations` are ignored. Requires ≥ 2
tests. Results land in the dashboard's **Pareto front** tab. See
`examples/cube_drop --variant pareto`.

If you *can* state a sane weighting, prefer weighted single-objective — one
number gives every other feature (gating, leaderboard, stall stop, archive
comparison) full traction.

## 3. The startup design — exploration before the model

CMA-ES and GP both begin with an **independent startup phase**: the first
`cmaes_startup_trials` completed trials are sampled space-fillingly, and only
then does the model-based sampler take over. The model needs a coverage map of
the landscape before its update/surrogate means anything.

- **`cmaes_startup_trials=None` (default) auto-sizes from dimensionality**
  (`project.resolve_startup_trials()`): the power of two nearest to `5·d`
  (CMA-ES) or `10·d` (GP), `d` = searched params. Examples: CMA-ES `d=6` → 32,
  `d=12` → 64; GP `d=6` → 64. Powers of two because the Sobol' design below is
  exactly balanced there. Adding or freezing parameters rescales the
  exploration automatically; set an explicit int to override.
- **`seed_sampler="sobol"`** replaces uniform-random startup with a scrambled
  Sobol' (quasi-Monte-Carlo) design: points that cover the space *evenly*,
  including the interactions between parameters — random points cluster and
  leave holes, QMC points don't. Recommended whenever you use the startup phase
  at all; `"random"` (default) preserves historical behavior.
- **`seed_sampler_seed`** (default 1234) scrambles the Sobol' design. A
  different seed gives a *different but equally balanced* exploration — use it
  when a validation run should not revisit the previous run's startup points.

Sizing intuition: a generation is `n_parallel` trials, so
`cmaes_startup_trials = K × n_parallel` buys roughly **K fully-random
generations** before the model engages. At least one population (≥
`n_parallel`); a 2–4× multiple for rugged or higher-dimensional landscapes.
Bigger = less risk of committing early to a poor basin; smaller = converges
sooner.

## 4. Budget, parallelism, and where CMA-ES starts

```python
PROJECT = SofaOptProject(
    ...,
    n_parallel=6,             # population size (also = concurrent SOFA procs)
    n_generations=120,        # optimizer update steps
    cmaes_sigma0=0.3,         # initial spread (CMA-ES only)
    max_active_sofa_procs=12, # hard cap on concurrent SOFA processes
)
```

### The budget — how many simulations you're committing to

```
trials evaluated  = n_parallel × n_generations
runSofa launches  = trials × Σ(run_count over selected tests)
```

6 × 120 = 720 candidates; one test with `run_count=3` makes that ~2160 SOFA
runs. Multiply by a scene's wall-time to estimate the run, and size
`n_generations` to the time you actually have.

### `n_parallel` — population size (λ)

Candidates per generation, launched concurrently. Bigger populations give a
steadier, more robust CMA-ES update (better on noisy or rugged landscapes) at
the cost of more simulations per generation. Match it to the CPU cores you can
spare; must be ≥ 4 for CMA-ES.

### `n_generations` — how long it refines

Your main "search longer" dial. Pair it with `stall_generations` (§5) so a
converged run stops early instead of burning the rest of the budget.

### Where it starts — `x0` (your `ParamSpec` defaults)

CMA-ES's initial mean is each parameter's `default` (clamped into bounds), so
**set defaults to your best-known / baseline design** — the search begins there
and improves outward. Frozen params (`low == high`) are held fixed and excluded.

### `cmaes_sigma0` — initial spread

The initial standard deviation in Optuna's internally-normalized space (each
range mapped to ~`[0, 1]`):

- `1.0` (default) is **broad** — first CMA-ES samples spread across most of
  each range.
- `0.2–0.3` starts **local**, clustered near your defaults — right when you
  trust the baseline and want refinement, not a global hunt.

CMA-ES adapts the spread as it goes; `sigma0` only sets the starting width.

### `max_active_sofa_procs` — concurrency safety cap

Distinct from `n_parallel`: one trial can launch several runs (multiple tests ×
repeats), so in-flight SOFA processes can exceed the population. This caps the
global total; new launches throttle until others finish. Set it to roughly your
core count (default 12).

### Recipes

- **Quick smoke test:** `n_parallel=4, n_generations=8, cmaes_startup_trials=8`.
- **Real run:** `n_parallel=` cores you can spare, `n_generations=100+`,
  startup auto (`None`) + `seed_sampler="sobol"`, `cmaes_sigma0=1.0`.
- **Trust your baseline, want refinement:** defaults sharp, `cmaes_sigma0≈0.2`,
  `cmaes_startup_trials≈n_parallel`.
- **Rugged / many parameters:** raise the startup design, keep `sigma0` near
  `1.0` for wide exploration.
- **Expensive scene, small budget:** `sampler="gp"`, startup auto,
  `seed_sampler="sobol"`.

## 5. Noise, repeats, duplicates, and stopping early

### `run_count` and `score_aggregation` (per `TestSpec`)

A stochastic scene (randomized scenarios, contact chatter) gives a noisy score;
noise makes every sampler chase phantoms. `run_count=N` runs the test N times
per candidate and `score_aggregation` combines the repeats:

| Mode | Choose it when |
|---|---|
| `"mean"` (default) | plain noise reduction; every repeat equally informative |
| `"median"` | occasional wild outliers (a solver blow-up shouldn't dominate) |
| `"sum"` | repeats are *different scenarios* and total achievement is the goal |
| `"exponential_coverage"` | repeats are different scenarios and you want **breadth**: sum × 1.5 per additional *positive* repeat — a candidate that succeeds in 3 scenarios beats one that triples the score of a single scenario |

Crashed repeats among successful ones count as **0.0** within their test (the
candidate is penalized, not discarded) — see §7 for what happens when *all*
runs fail.

### `run_count_min` — adaptive re-evaluation (racing)

A fixed `run_count` spends the same simulation budget on a hopeless candidate
as on a contender. Racing spends repeats where they matter:

**Intuition.** After a few repeats you know a candidate's score *roughly* — a
mean and a confidence interval. If even the optimistic end of that interval
cannot reach the best score seen so far, more repeats cannot change the
ranking at the top, so stop measuring. Only candidates whose interval
*overlaps the incumbent* — the ones the sampler actually needs to rank
precisely — earn the full repeat count.

**Mechanics.** `TestSpec(run_count=6, run_count_min=2)` launches 2 repeats
per trial; once they finish, repeats are added one at a time while
`mean + t₉₅·s/√n` (propagated through the real normalize/weight pipeline,
clamping included) still reaches `study.best_value`. Skipped repeats show as
`skipped` run slots with the racing reason. Fine print:

- **`score_aggregation="mean"` only** (enforced): with `sum` /
  `exponential_coverage` the repeats are *different scenarios* — skipping
  some would change what the score measures, not its precision.
- A potential **new incumbent keeps running** to its full `run_count`: every
  later racing decision compares against its score, so it must be precise.
- Generation 1 has no incumbent — every candidate runs its full count and
  becomes the baseline.
- Crashed repeats count as 0.0 (as in final scoring), so crash-riddled
  candidates race out quickly. Multi-objective runs ignore racing (no scalar
  incumbent). Works together with `gated=True`: a gate-opened raced test
  still starts at `run_count_min`.

### `dedup_trials` — don't re-simulate identical candidates

With integer/bool-heavy spaces or `float_step` quantization, a converged
CMA-ES endgame can propose the *same* parameter vector for whole generations.
`dedup_trials=True` reuses the recorded score of a completed duplicate: the
trial is told to Optuna immediately and marked `cached`, and no SOFA process
launches. **Only safe when the objective is deterministic** — if you average
noise over `run_count` repeats, keep it off (each duplicate is a fresh noise
sample the cache would destroy). Ignored for multi-objective studies.

### `stall_generations` — stop when converged

`stall_generations=K` (default 0 = off) ends the run after K consecutive
generations without any improvement of the best score. Post-run steps (summary
video, report) still execute. Single-objective only. Pair a generous
`n_generations` with `stall_generations=10–20` and let convergence, not the
clock, decide.

### `cmaes_restarts` — restart instead of stopping (IPOP)

On a **multimodal** landscape a stalled CMA-ES usually means "converged into
one basin", not "nothing left to find". The standard answer (IPOP, Auger &
Hansen 2005) is to restart with a larger population, which searches more
globally each time. `cmaes_restarts=N` re-purposes the stall signal: the
first N stalls each swap in a **fresh** CMA-ES with

- population size × `cmaes_inc_popsize` (default 2 — the IPOP schedule),
- the initial `cmaes_sigma0` (full initial spread again),
- a uniform-random start point (a different one per restart, reproducible).

The run-global best is never forgotten — the study keeps every trial, and a
restart must beat the incumbent within `stall_generations` generations or
the next stall fires (restarting again, or stopping once the budget of N is
spent). `n_generations` still caps the total run. Requires
`stall_generations > 0` and `sampler="cmaes"`; single-objective only.
Resuming a paused run continues from the latest restart's state.

Note: Optuna deprecated its own `restart_strategy` in v4.4, so sofaopt
implements the restart at the orchestrator level (`core/restart.py`) — the
growing population means one internal CMA update spans several sofaopt
generations, which is expected.

## 6. How scores combine — and one objective vs many

Per trial, in exactly this order and nowhere else (`core/scoring.py`):

1. each test's repeat scores → one per-test aggregate via `score_aggregation`;
2. per-test aggregate → normalized by the test's `max_score`, clamped at 1.0;
3. normalized tests → combined by `weight`, renormalized over the tests
   actually counted;
4. the result (0–100) is the study objective, recorded in `trial_state.json`
   (`final_score`) — every display (dashboard, leaderboard, videos, archive
   comparison) reads it verbatim; nothing recomputes its own.

**Gating** (`TestSpec(gated=True)`): a gated (typically expensive) test only
runs — and only counts — once an *ungated* test scored above zero for the
candidate. Use it to spend long validation scenes only on candidates that
passed a cheap sanity test. When the gate stays closed, weights renormalize
over the ungated tests.

**Weighted single-objective vs `multi_objective`.** Weights express "I know the
exchange rate between my tests"; Pareto mode (§2, NSGA-II) expresses "I don't —
show me the trade-off curve". Start weighted; switch to `multi_objective=True`
when you catch yourself re-running with different weights to see what happens.

## 7. Failure semantics — what the sampler learns from a bad trial

Two deliberately different outcomes:

- **Hard failure teaches.** A prepare-hook exception or *all* runs crashing
  reports `hard_fail_score` (default **−3.0**) to the sampler as a *real*
  observation — infeasible geometry and instantly-exploding scenes are
  information, and the sampler learns to avoid that region. Slightly below the
  worst legitimate score is the right magnitude; a huge penalty would distort
  CMA-ES's landscape estimate.
- **A timeout says nothing.** A run killed by the `sofa_realtime_timeout`
  backstop is reported as **pruned** — a wedged process tells you about your
  infrastructure, not your parameters, so it must not push the search away
  from that region.

And in between: crashed repeats among successful ones count as 0.0 within
their test (§5). Any change to this split is a contract change and updates the
docs with it.

## 8. Understanding the results

Four tools, cheapest first:

- **OAT sensitivity sweep** (`sofaopt.sensitivity.run_sensitivity_analysis`,
  before a big run): sweeps each non-frozen param low→high with the others at
  defaults; returns signed, ranked sensitivities. Use it to freeze irrelevant
  parameters, narrow ranges, and estimate the effective dimensionality (which
  picks your sampler, §1). It cannot see interactions — that's the next tool.
- **fANOVA importance + interaction map** (`sofaopt.analysis.analyze`, or the
  dashboard's **Importance / Interactions** tab; needs the `[analysis]` extra):
  reuses the trials the optimizer already ran — no extra simulations. Main
  effects via functional ANOVA; pairwise coupling via a random-forest surrogate
  and Friedman's H-statistic (or surrogate Sobol' indices with `SALib`
  installed). Strong interactions with a small budget argue for CMA-ES (whose
  covariance models coupling) over TPE.
- **Pareto front tab** (multi-objective runs): the recorded front; pick the
  knee point or the trade-off your application tolerates.
- **Archive comparison** (**Archives** tab or `sofaopt.archive_run` /
  `list_archives`): overlaid best-so-far convergence curves and best-params
  diffs across runs — the honest way to compare samplers or settings, since it
  reads each run's *recorded* scores.

A sane campaign: OAT sweep → freeze/narrow → short random or Sobol'-startup run
→ fANOVA on it → pick sampler + budget from what you learned → real run →
archive → compare.
