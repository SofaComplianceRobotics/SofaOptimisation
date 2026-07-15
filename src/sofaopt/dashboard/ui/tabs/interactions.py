"""Importance / Interactions tab — fANOVA main effects + interaction map.

Strategy-aligned replacement for the OAT sensitivity view: shows which
parameters matter (main effects) and which act *jointly* (interaction map),
both derived from the trials the optimizer already ran.
"""

from dash import dcc, html

from sofaopt.dashboard.context import LIVE_REFRESH_SECONDS


def build_interactions_tab() -> html.Div:
    return html.Div(
        [
            html.H3("Parameter Importance & Interactions", className="mb-2"),
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
        ],
        className="p-3",
    )
