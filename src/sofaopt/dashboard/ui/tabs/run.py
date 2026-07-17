"""Run tab — one place to launch work on the project's tests.

Merges the former Scenes and Optimise tabs: the test catalog is listed once;
each row carries a Preview button (opens that scene in an interactive runSofa
window), a Gate toggle, and a weight slider. Below sit the optimizer settings,
Start/Pause, and a shared log window that tails the run — whether this dashboard
launched it or a ``sofaopt`` CLI run did.
"""

from dash import dcc, html

from sofaopt.dashboard import context
from sofaopt.dashboard.ui.components import build_log_view

PIE_PALETTE = [
    "#4c8bf5",
    "#e84393",
    "#34a853",
    "#fa7b17",
    "#9c27b0",
    "#00bcd4",
    "#ff5722",
    "#8bc34a",
]


def _equal_split(n: int) -> list[int]:
    """Split 100 into ``n`` integer parts as evenly as possible."""
    if n == 0:
        return []
    base = 100 // n
    rem = 100 - base * n
    return [base + (1 if i < rem else 0) for i in range(n)]


def _num_input(input_id: str, label: str, value, tooltip: str, min_val: int = 0) -> html.Div:
    """One labelled number input with a hover tooltip."""
    return html.Div(
        [
            html.Label(label, className="form-label mb-1 small text-muted"),
            dcc.Input(
                id=input_id, type="number", min=min_val, step=1, value=value,
                className="form-control form-control-sm", style={"width": "90px"},
            ),
        ],
        title=tooltip,
        className="me-3",
    )


def _build_restart_controls(project) -> html.Div:
    """Second optimizer row: IPOP-restart / convergence / dedup knobs.

    Forwarded as ``OPT_*`` env overrides like the first row, so a broad
    run-until-converged campaign can be configured entirely from the UI.
    """
    flags = []
    if project.restart_on_convergence:
        flags.append("conv")
    if project.warm_restarts:
        flags.append("warm")
    if project.dedup_trials:
        flags.append("dedup")
    return html.Div(
        [
            html.Div(
                [
                    _num_input(
                        "opt-cmaes-restarts", "Restarts", project.cmaes_restarts,
                        "Max IPOP restarts (CMA-ES only): each trigger restarts the optimizer "
                        "with the population multiplied by 'Pop growth'. 0 = a stall stops the "
                        "run instead. Set generously for 'Run until converged'.",
                    ),
                    _num_input(
                        "opt-stall-generations", "Stall gens", project.stall_generations,
                        "Consecutive generations without a best-score improvement that trigger "
                        "a restart (or stop the run when Restarts = 0). 0 = off. Not needed as "
                        "a trigger when 'Restart on convergence' is checked.",
                    ),
                    _num_input(
                        "opt-inc-popsize", "Pop growth", project.cmaes_inc_popsize,
                        "Population multiplier applied at each IPOP restart (default 2). The "
                        "internal CMA population grows past Parallel; one CMA update then "
                        "spans several generations — expected.",
                        min_val=1,
                    ),
                    html.Div(
                        dcc.Checklist(
                            id="opt-restart-flags",
                            options=[
                                {"label": " Restart on convergence", "value": "conv"},
                                {"label": " Warm restarts", "value": "warm"},
                                {"label": " Dedup identical trials", "value": "dedup"},
                            ],
                            value=flags,
                            className="mt-4",
                            inputClassName="me-1",
                            labelClassName="me-3",
                        ),
                        title="Restart on convergence: trigger restarts on CMA-ES's own convergence "
                              "signal instead of the stall plateau — recommended whenever Restarts > 0 "
                              "(the plateau fires while the search is still productive). "
                              "Warm restarts: re-seed each restart from the best-so-far with an "
                              "inflated spread instead of a random point. "
                              "Dedup: reuse the recorded score when the exact same parameter vector "
                              "reappears — only safe for a DETERMINISTIC objective.",
                    ),
                ],
                className="d-flex align-items-start flex-wrap",
            ),
        ],
        className="mt-2",
    )


def _build_sampler_controls() -> html.Div:
    """Optimizer-settings row: sampler, initial design, margin, parallelism.

    Defaults are read from the project; the values are forwarded as ``OPT_*``
    env overrides when the Run button launches the headless study, so the UI
    drives the same knobs as the ``sofaopt`` CLI flags.
    """
    project = context.project()
    prunable = (
        len(project.tests) == 1
        and project.tests[0].prunable
        and bool(project.tests[0].prune_rungs)
    )
    return html.Div(
        [
            html.H5("Optimizer", className="mb-2"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Sampler", className="form-label mb-1 small text-muted"),
                            dcc.Dropdown(
                                id="opt-sampler",
                                options=[
                                    {"label": "CMA-ES (covariance, coupled params)", "value": "cmaes"},
                                    {"label": "GP-BO (sample-efficient, expensive sims)", "value": "gp"},
                                    {"label": "TPE (Bayesian)", "value": "tpe"},
                                    {"label": "Random (baseline)", "value": "random"},
                                ],
                                value=project.sampler,
                                clearable=False,
                            ),
                        ],
                        className="col-12 col-md-4",
                        title="The optimization algorithm. CMA-ES adapts a covariance and handles "
                              "coupled parameters; GP-BO is the most sample-efficient for expensive "
                              "sims (small batches); TPE is a Bayesian alternative; Random is the baseline.",
                    ),
                    html.Div(
                        [
                            html.Label("Initial design", className="form-label mb-1 small text-muted"),
                            dcc.Dropdown(
                                id="opt-seed-sampler",
                                options=[
                                    {"label": "Sobol' (space-filling DOE)", "value": "sobol"},
                                    {"label": "Random", "value": "random"},
                                ],
                                value=project.seed_sampler,
                                clearable=False,
                            ),
                        ],
                        className="col-12 col-md-3",
                        title="How the startup trials (before the sampler takes over) are placed: "
                              "Sobol' fills the space evenly — recommended for a broad search; "
                              "Random is plain uniform.",
                    ),
                    html.Div(
                        [
                            html.Label("Parallel / Gens", className="form-label mb-1 small text-muted"),
                            html.Div(
                                [
                                    dcc.Input(
                                        id="opt-n-parallel", type="number", min=1, step=1,
                                        value=project.n_parallel, className="form-control form-control-sm",
                                        style={"width": "80px"},
                                    ),
                                    dcc.Input(
                                        id="opt-n-generations", type="number", min=1, step=1,
                                        value=project.n_generations, className="form-control form-control-sm ms-2",
                                        style={"width": "80px"},
                                    ),
                                ],
                                className="d-flex",
                            ),
                        ],
                        className="col-12 col-md-3",
                        title="Parallel = trials per generation (also the CMA-ES population size). "
                              "Gens = generation budget; with 'Run until converged' it is a safety "
                              "ceiling, not a target. Concurrent SOFA processes stay capped by "
                              "max_active_sofa_procs regardless.",
                    ),
                    html.Div(
                        [
                            html.Label("Pruning", className="form-label mb-1 small text-muted"),
                            dcc.Dropdown(
                                id="opt-prune-mode",
                                options=[
                                    {"label": "Off", "value": "off"},
                                    {"label": "Shadow (dry-run, logs only)", "value": "shadow"},
                                    {"label": "Kill (stop laggards early)", "value": "kill"},
                                ],
                                value=project.prune_mode if prunable else "off",
                                clearable=False,
                                disabled=not prunable,
                            ),
                            html.Small(
                                "" if prunable else "Needs one prunable test with calibrated rungs.",
                                className="text-muted",
                            ),
                        ],
                        className="col-12 col-md-3",
                        title="Multi-fidelity step pruning: at calibrated rungs the generation's "
                              "candidates are ranked by their anytime partial score and the bottom "
                              "are stopped early. Shadow only LOGS would-kill decisions (risk-free "
                              "dry run); Kill actually stops them.",
                    ),
                    html.Div(
                        [
                            html.Div(
                                dcc.Checklist(
                                    id="opt-cmaes-margin",
                                    options=[{"label": " CMA-ES with Margin", "value": "margin"}],
                                    value=["margin"] if project.cmaes_with_margin else [],
                                    className="mt-4",
                                ),
                                title="Margin variant of CMA-ES: keeps sampling diversity when many "
                                      "parameters are integers/discrete. Only applies to CMA-ES.",
                            ),
                            html.Div(
                                dcc.Checklist(
                                    id="opt-run-until-converged",
                                    options=[{"label": " Run until converged", "value": "converged"}],
                                    value=["converged"] if project.run_until_converged else [],
                                    className="mt-1",
                                ),
                                title="Self-size the run: keep restarting on each stall/convergence "
                                      "and stop once 'patience' consecutive restarts fail to improve "
                                      "the best score. Gens becomes a ceiling. Needs CMA-ES, "
                                      "Restarts > 0 and a restart trigger.",
                            ),
                            html.Div(
                                dcc.Input(
                                    id="opt-restart-patience", type="number", min=1, step=1,
                                    value=project.restart_patience,
                                    className="form-control form-control-sm mt-1",
                                    style={"width": "80px"},
                                    placeholder="patience",
                                ),
                                title="Patience: consecutive restarts without a new global best "
                                      "tolerated before the run stops (only used with 'Run until "
                                      "converged'). A restart that improves the best resets the streak.",
                            ),
                        ],
                        className="col-12 col-md-3",
                    ),
                ],
                className="row g-2 align-items-start",
            ),
            _build_restart_controls(project),
            html.Small(
                "GP-BO is most sample-efficient with a small batch — try Parallel ≈ 4. "
                "Margin only applies to CMA-ES. ‘Run until converged’ ignores Gens as a "
                "target and stops once restarts stop improving (needs CMA-ES + restarts). "
                "Hover any control for details.",
                className="text-muted",
            ),
        ],
        className="mb-3 p-3 border rounded bg-light",
    )


def _test_row(name: str, spec, pre_selected: bool, weight: int) -> html.Div:
    """One catalog row: Preview · select · gate · weight."""
    return html.Div(
        [
            html.Button(
                "👁 Preview",
                id={"type": "scene-preview", "test": name},
                n_clicks=0,
                className="btn btn-outline-secondary btn-sm me-2",
                style={"flexShrink": 0, "whiteSpace": "nowrap"},
                title="Open this scene in an interactive SOFA viewer",
            ),
            dcc.Checklist(
                id={"type": "test-check", "test": name},
                options=[{"label": f" {spec.label}", "value": name}],
                value=[name] if pre_selected else [],
                style={"display": "inline-flex", "alignItems": "center", "minWidth": "200px", "flexShrink": 0},
                className="me-2",
            ),
            html.Div(
                dcc.Checklist(
                    id={"type": "gate-check", "test": name},
                    options=[{"label": " Gate", "value": name}],
                    value=[],
                    style={"display": "inline-flex", "alignItems": "center", "minWidth": "90px", "flexShrink": 0},
                    className="me-2 text-muted",
                ),
                title="Gated: this test only runs after one of the ungated tests succeeds in "
                      "the same trial — skips an expensive secondary test when the primary "
                      "already failed. Only useful with several tests; at least one must stay ungated.",
                style={"flexShrink": 0},
            ),
            html.Div(
                dcc.Slider(
                    id={"type": "weight-slider", "test": name},
                    min=0,
                    max=100,
                    step=1,
                    value=weight,
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": True},
                    updatemode="drag",
                ),
                style={"flexGrow": 1},
            ),
        ],
        className="d-flex align-items-center mb-3",
        style={"gap": "8px"},
    )


def build_run_tab(catalog: dict) -> html.Div:
    """Build the Run tab from the project's test catalog."""
    names = list(catalog.keys())
    any_default = any(spec.default_selected for spec in catalog.values())

    selected_names = [
        name for name, spec in catalog.items() if spec.default_selected or not any_default
    ]
    weights = _equal_split(len(selected_names))
    initial_store: dict[str, int] = {}
    wi = 0
    for name in names:
        if name in selected_names:
            initial_store[name] = weights[wi]
            wi += 1
        else:
            initial_store[name] = 0

    test_rows = [
        _test_row(
            name, spec,
            pre_selected=(spec.default_selected or not any_default),
            weight=initial_store[name],
        )
        for name, spec in catalog.items()
    ]

    return html.Div(
        [
            html.H3("Run", className="mb-2"),
            _build_sampler_controls(),
            dcc.Store(id="opt-weights-store", data=initial_store),
            html.Div(
                [
                    html.Div(
                        [
                            html.P("Preview opens a scene in an interactive SOFA window; drag a slider — the others adjust so the total stays at 100%.", className="text-muted mb-2"),
                            html.P("Use Gate to delay a test until one of the ungated tests succeeds.", className="text-muted mb-3"),
                            html.Div(test_rows, className="mb-2"),
                            html.Div(id="run-scene-status", className="mb-2 small"),
                            html.Div(
                                [
                                    html.Button(
                                        "Equal split", id="opt-equal-btn", n_clicks=0,
                                        className="btn btn-outline-secondary btn-sm me-2",
                                        title="Reset the selected tests' weights to an even split (sums to 100%).",
                                    ),
                                    html.Button(
                                        "Normalize", id="opt-normalize-btn", n_clicks=0,
                                        className="btn btn-outline-secondary btn-sm",
                                        title="Rescale the current weights proportionally so they sum to 100%.",
                                    ),
                                ],
                                className="mb-3",
                            ),
                        ],
                        className="col-12 col-md-7",
                    ),
                    html.Div(
                        dcc.Graph(id="opt-pie", config={"displayModeBar": False}, style={"height": "320px"}),
                        className="col-12 col-md-5",
                        title="Objective composition: each selected test's weight share of the "
                              "final score. A configuration view (what you are asking the "
                              "optimizer to maximize), not a results view.",
                    ),
                ],
                className="row g-3 mb-3",
            ),
            html.Div(id="opt-weight-status", className="mb-3"),
            html.Div(
                [
                    html.Button("Start Optimisation", id="opt-start-btn", n_clicks=0, className="btn btn-success me-2"),
                    html.Button("Pause", id="opt-stop-btn", n_clicks=0, disabled=True, className="btn btn-danger"),
                ],
                className="mb-3",
            ),
            html.Div(id="opt-status", className="mb-2 fw-semibold"),
            build_log_view(pre_id="run-log", interval_id="opt-interval", filter_id="run-log-filter"),
        ],
        className="p-3",
    )
