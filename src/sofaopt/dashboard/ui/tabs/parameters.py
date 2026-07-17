"""Parameters tab — everything about the search-space parameters in one place.

Merges the former Parameter-Bounds and Importance/Interactions tabs:

1. a table of every parameter (name / type / range / default / state / current
   best), including *frozen* ones the bounds heatmap omits;
2. the sampled-value bounds heatmap with the latest-trial marker;
3. (single-objective only) fANOVA main-effect importance + the pairwise
   interaction map — both computed from completed trials, no extra simulations.
"""

from dash import dcc, html

from sofaopt.dashboard import context
from sofaopt.dashboard.context import LIVE_REFRESH_SECONDS


def _interactions_section() -> html.Div:
    return html.Div(
        [
            html.Hr(className="my-4"),
            html.H4("Importance & Interactions", className="mb-2"),
            html.P(
                "Computed from completed trials in study.db — no extra simulations. "
                "Main effects use functional ANOVA; the interaction map shows how "
                "strongly each parameter pair is coupled.",
                className="text-muted",
            ),
            dcc.Graph(id="importance-bar"),
            dcc.Graph(id="interaction-heatmap"),
            dcc.Interval(
                id="interactions-interval",
                # heavier than the live monitors — refresh slowly (figures are cached).
                interval=int(max(10.0, LIVE_REFRESH_SECONDS * 5) * 1000),
                n_intervals=0,
            ),
        ]
    )


def build_parameters_tab() -> html.Div:
    """Build the Parameters tab. The importance/interaction section needs a
    scalar objective, so it is omitted in multi-objective (Pareto) mode."""
    children = [
        html.H3("Parameters", className="mb-2"),
        html.H4("Search space", className="mb-2"),
        html.P(
            "Every parameter the project declares. Frozen parameters "
            "(low == high) are held at their default and are not searched.",
            className="text-muted",
        ),
        html.Div(id="param-table"),
        html.Hr(className="my-4"),
        html.H4("Bounds monitor", className="mb-2"),
        html.P("Where sampled values and the latest trial sit within each active range."),
        dcc.Graph(id="param-bounds-graph"),
        dcc.Interval(
            id="bounds-interval",
            interval=int(max(1.0, LIVE_REFRESH_SECONDS) * 1000),
            n_intervals=0,
        ),
    ]
    if not context.project().multi_objective:
        children.append(_interactions_section())
    return html.Div(children, className="p-3")
