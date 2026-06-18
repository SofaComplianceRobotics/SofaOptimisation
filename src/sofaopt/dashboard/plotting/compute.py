"""Data transformation, aggregation and score series for the performance plot."""

from sofaopt.dashboard import context
from sofaopt.dashboard.analyze_io import load_all_trials  # noqa: F401 (re-export)

from .colors import TEST_COLORS


def _collect_all_test_names(records: list[dict]) -> list[str]:
    """Stable ordered list of every unique test name across all records."""
    seen: list[str] = []
    for r in records:
        for name in (r.get("test_scores") or {}).keys():
            if name not in seen:
                seen.append(name)
    return seen


def _test_color(test_name: str, name_order: list[str]) -> str:
    """Hex color for a test, consistent across the whole plot."""
    try:
        idx = name_order.index(test_name)
    except ValueError:
        idx = abs(hash(test_name)) % len(TEST_COLORS)
    return TEST_COLORS[idx % len(TEST_COLORS)]


def _compute_contributions(record: dict) -> dict[str, float]:
    """Per-test weighted contribution for one trial: ``normalize(agg) * weight_pct``."""
    test_scores: dict = record.get("test_scores") or {}
    if not test_scores:
        return {"score": float(record.get("final_score", record.get("score", 0.0)))}

    contributions: dict[str, float] = {}
    for test_name, test_info in test_scores.items():
        if not isinstance(test_info, dict):
            continue
        agg = float(test_info.get("aggregate_score", 0.0) or 0.0)
        raw_max = test_info.get("max_score")
        max_score = float(raw_max) if raw_max is not None else 1.0
        wpct = float(test_info.get("weight_pct", 0.0) or 0.0)
        norm = min(agg / max_score, 1.0) if max_score > 0 else 0.0
        contributions[test_name] = norm * wpct

    return contributions or {
        "score": float(record.get("final_score", record.get("score", 0.0)))
    }


def compute_plot_data(records: list[dict], all_test_names: list[str]) -> dict:
    """Pre-compute every series needed for a full performance-plot redraw."""
    half = context.CENTERED_AVG_HALF_WINDOW
    xs = [r["chron"] for r in records]
    contributions = [_compute_contributions(r) for r in records]
    final_scores = [sum(c.values()) for c in contributions]
    failed_mask = [bool(r.get("failed", False)) for r in records]
    is_complete = [bool(r.get("is_complete", True)) for r in records]

    avg_x, avg_y = [], []
    for i, r in enumerate(records):
        lo = max(0, i - half)
        hi = min(len(records) - 1, i + half)
        scores = [sum(_compute_contributions(w).values()) for w in records[lo : hi + 1]]
        avg_x.append(r["chron"])
        avg_y.append(sum(scores) / len(scores))

    best_x, best_y = [], []
    running_best = None
    for i, r in enumerate(records):
        if not failed_mask[i]:
            fs = final_scores[i]
            if running_best is None or fs > running_best:
                running_best = fs
        if running_best is not None:
            best_x.append(r["chron"])
            best_y.append(running_best)

    per_test_avg: dict[str, tuple[list, list]] = {}
    for test_name in all_test_names:
        avg_x_t, avg_y_t = [], []
        for i, r in enumerate(records):
            lo = max(0, i - half)
            hi = min(len(records) - 1, i + half)
            window_scores = [
                _compute_contributions(records[k]).get(test_name, 0.0)
                for k in range(lo, hi + 1)
            ]
            avg_x_t.append(r["chron"])
            avg_y_t.append(sum(window_scores) / len(window_scores))
        per_test_avg[test_name] = (avg_x_t, avg_y_t)

    gen_tick_positions, gen_tick_labels = [], []
    prev_gen = None
    for r in records:
        if r["gen_index"] != prev_gen:
            gen_tick_positions.append(r["chron"])
            gen_tick_labels.append(str(r["gen_index"]))
        prev_gen = r["gen_index"]

    return {
        "xs": xs,
        "final_scores": final_scores,
        "failed_mask": failed_mask,
        "is_complete": is_complete,
        "contributions": contributions,
        "avg_x": avg_x,
        "avg_y": avg_y,
        "best_x": best_x,
        "best_y": best_y,
        "per_test_avg": per_test_avg,
        "gen_tick_positions": gen_tick_positions,
        "gen_tick_labels": gen_tick_labels,
    }


def _calculate_smart_ticks(
    gen_tick_positions: list, gen_tick_labels: list, visible_range: tuple = None
) -> tuple[list, list]:
    """Stride generation ticks to avoid crowding the X axis."""
    if not gen_tick_positions:
        return [], []

    if visible_range:
        x_min, x_max = visible_range
    else:
        x_min = gen_tick_positions[0]
        x_max = gen_tick_positions[-1]

    visible_indices = [
        i for i, pos in enumerate(gen_tick_positions) if x_min <= pos <= x_max
    ]
    if not visible_indices:
        return [], []

    num_visible = len(visible_indices)
    max_comfortable_labels = 15
    if num_visible <= max_comfortable_labels:
        stride = 1
    else:
        min_stride = num_visible / max_comfortable_labels
        standard_intervals = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
        stride = next(
            (s for s in standard_intervals if s >= min_stride), standard_intervals[-1]
        )

    first_idx = visible_indices[0]
    filtered_indices = [i for i in visible_indices if (i - first_idx) % stride == 0]
    return (
        [gen_tick_positions[i] for i in filtered_indices],
        [gen_tick_labels[i] for i in filtered_indices],
    )
