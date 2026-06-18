"""Larger progress sections: per-trial scoring detail, generation stats, grid."""

from dash import html

from sofaopt.dashboard.plotting.colors import C_BANNER, C_BORDER, C_FINAL

from .builders import _build_progress_card


def _build_trial_detail(state: dict, gen_name: str, trial_name: str) -> html.Div:
    """Full per-test scoring breakdown for one trial."""
    final_score = state.get("final_score", 0.0) or 0.0
    test_scores: dict = state.get("test_scores") or {}

    header = html.Div(
        [
            html.Span(f"{gen_name} / {trial_name}", className="fw-semibold me-3"),
            html.Span(
                f"Final score: {final_score:.2f} / 100",
                style={"color": C_FINAL, "fontWeight": 700},
            ),
        ],
        className="d-flex align-items-center mb-3",
        style={"fontSize": "1.05rem"},
    )

    rows = []
    for test_name, info in sorted(
        test_scores.items(), key=lambda kv: kv[1].get("weight_pct", 0), reverse=True
    ):
        if not isinstance(info, dict):
            continue
        agg = float(info.get("aggregate_score", 0.0) or 0.0)
        max_s = float(info.get("max_score", 1.0) or 1.0)
        weight = float(info.get("weight_pct", 0.0) or 0.0)
        norm = min(agg / max_s, 1.0) if max_s > 0 else 0.0
        contribution = norm * weight
        success_pct = norm * 100
        run_scores: list = info.get("run_scores") or []
        run_total = int(info.get("run_total", 1))
        bar_pct = min(norm * 100, 100)

        run_cells = []
        if run_total > 1 and run_scores:
            for idx, rs in enumerate(run_scores):
                if rs == float("-inf") or rs is None:
                    label, color = "FAIL", "#e03131"
                else:
                    label, color = f"{rs:.3f}", "#2f9e44"
                run_cells.append(
                    html.Span(
                        f"run {idx + 1}: {label}",
                        style={
                            "color": color,
                            "background": "#f1f3f5",
                            "borderRadius": "4px",
                            "padding": "2px 8px",
                            "fontSize": "0.78rem",
                            "fontWeight": 600,
                            "marginRight": "6px",
                        },
                    )
                )

        rows.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(test_name, className="fw-semibold me-2"),
                            html.Span(
                                f"{weight:.0f}% of score",
                                style={
                                    "background": "#e7f5ff",
                                    "color": "#1971c2",
                                    "borderRadius": "999px",
                                    "padding": "1px 10px",
                                    "fontSize": "0.78rem",
                                    "fontWeight": 600,
                                },
                            ),
                        ],
                        className="d-flex align-items-center mb-1",
                    ),
                    html.Div(
                        html.Div(
                            style={
                                "width": f"{bar_pct:.1f}%",
                                "height": "100%",
                                "background": C_BANNER,
                                "borderRadius": "999px",
                                "transition": "width 400ms ease",
                            }
                        ),
                        style={
                            "height": "10px",
                            "background": "#dee2e6",
                            "borderRadius": "999px",
                            "overflow": "hidden",
                            "marginBottom": "4px",
                        },
                    ),
                    html.Div(
                        [
                            html.Span(f"{agg:.3f} / {max_s:.3f}", style={"fontWeight": 600, "marginRight": "6px"}),
                            html.Span(f"-> {success_pct:.1f}% success rate", className="text-muted me-3"),
                            html.Span(f"earned {contribution:.2f} / {weight:.1f} pts", style={"color": C_FINAL, "fontWeight": 600}),
                        ],
                        style={"fontSize": "0.82rem"},
                        className="mb-1",
                    ),
                    html.Div(run_cells, className="d-flex flex-wrap") if run_cells else html.Div(),
                ],
                className="mb-3 pb-3",
                style={"borderBottom": f"1px solid {C_BORDER}"},
            )
        )

    return html.Div(
        [header] + rows,
        style={
            "background": "#f8f9fa",
            "border": f"1px solid {C_BORDER}",
            "borderRadius": "8px",
            "padding": "16px 20px",
        },
    )


def _build_progress_stats(current_records: list[dict], all_records: list[dict]) -> html.Div:
    """Compact overview: generation in view + best score observed."""
    gen_index = current_records[0].get("gen_index", -1) if current_records else -1
    gen_label = f"Gen {gen_index}" if gen_index >= 0 else "—"

    best_record = max(
        [r for r in all_records if not r.get("failed", False)],
        key=lambda x: x.get("final_score", 0),
        default=None,
    )
    best_score = best_record.get("final_score", 0) if best_record else 0
    best_trial = (
        f"{best_record.get('gen_name', '')} / {best_record.get('trial_name', '')}"
        if best_record
        else "N/A"
    )

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [html.H6("Generation", className="text-muted"), html.H4(gen_label, className="text-info")],
                        className="col-12 col-md-6",
                    ),
                    html.Div(
                        [
                            html.H6("Best Score", className="text-muted"),
                            html.H4(f"{best_score:.4f}", className="text-warning"),
                            html.Small(best_trial, className="text-muted"),
                        ],
                        className="col-12 col-md-6",
                    ),
                ],
                className="row g-3",
            )
        ],
        className="p-3 bg-light rounded",
    )


def _build_progress_grid(records: list[dict]) -> html.Div:
    """Trial-card grid for the current generation."""
    if not records:
        return html.Div("No trial data found.", className="text-muted")
    rows = [_build_progress_card(record) for record in records]
    return html.Div(rows, style={"display": "grid", "gap": "12px"})
