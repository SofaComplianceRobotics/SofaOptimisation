"""Color palette for plotting and UI."""

# Per-test series colors; assignment is stable per test name.
TEST_COLORS = [
    "#404867",
    "#6b7aad",
    "#9aa3cc",
    "#2c3e6b",
    "#c0392b",
    "#2ecc71",
    "#e67e22",
    "#9b59b6",
]

# Archive-comparison series colors: a CVD-validated categorical order
# (worst adjacent-pair CVD deltaE 24.2 on white). Assigned by each archive's
# stable position in the full archive list — color follows the run, never its
# rank, so deselecting one run doesn't repaint the others.
ARCHIVE_COLORS = [
    "#2a78d6",  # blue
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#e87ba4",  # magenta
    "#eb6834",  # orange
]

C_BANNER = "#404867"
C_BG = "#ffffff"
C_BORDER = "#d0d3d8"
C_FINAL = "#c0392b"  # final-score tick colour
C_AVG = "#2ecc71"  # rolling average line
C_BEST = "#e74c3c"  # best-so-far line
