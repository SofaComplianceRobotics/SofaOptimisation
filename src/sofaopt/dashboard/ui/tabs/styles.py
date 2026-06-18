"""Shared styling constants for dashboard tabs."""

from sofaopt.dashboard.plotting.colors import C_BANNER as ACCENT

LOG_STYLE = {
    "height": "420px",
    "overflowY": "auto",
    "background": "#1e1e1e",
    "color": "#d4d4d4",
    "padding": "12px",
    "borderRadius": "6px",
    "fontSize": "0.82rem",
    "whiteSpace": "pre-wrap",
    "wordBreak": "break-word",
    "fontFamily": "monospace",
}

PAGE_STYLE = {"background": "#f4f5f7", "minHeight": "100vh"}

HEADER_BAR_STYLE = {
    "background": ACCENT,
    "color": "#ffffff",
    "boxShadow": "0 1px 4px rgba(0, 0, 0, 0.25)",
}

HEADER_INNER_STYLE = {
    "maxWidth": "1280px",
    "margin": "0 auto",
    "padding": "14px 24px",
    "display": "flex",
    "alignItems": "baseline",
    "gap": "16px",
}

HEADER_TITLE_STYLE = {"fontSize": "1.45rem", "fontWeight": "700", "letterSpacing": "0.02em"}
HEADER_SUBTITLE_STYLE = {"fontSize": "0.9rem", "color": "rgba(255, 255, 255, 0.75)"}
BODY_STYLE = {"maxWidth": "1280px", "margin": "0 auto", "padding": "0 24px 32px"}

TABS_STYLE = {"borderBottom": "1px solid #d0d3d8", "marginTop": "12px", "marginBottom": "18px"}

TAB_STYLE = {
    "padding": "10px 18px",
    "border": "none",
    "borderBottom": "3px solid transparent",
    "background": "transparent",
    "color": "#5a5f6a",
    "fontWeight": "500",
}

TAB_SELECTED_STYLE = {
    **TAB_STYLE,
    "color": ACCENT,
    "borderBottom": f"3px solid {ACCENT}",
    "fontWeight": "600",
}

TAB_CONTENT_STYLE = {
    "background": "#ffffff",
    "border": "1px solid #e1e4e8",
    "borderRadius": "8px",
    "padding": "20px",
    "boxShadow": "0 1px 2px rgba(0, 0, 0, 0.04)",
}
