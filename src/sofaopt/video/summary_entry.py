"""Subprocess entry point for background summary-video generation.

Launched by the dashboard as::

    python -m sofaopt.video.summary_entry <args.json>

where ``args.json`` contains ``{"project": <project_to_jsonable dict>,
"summary_path": "...", "top_n": N, "bottom_n": M}``. JSON replaces the old
pickle+generated-script handoff (§9: no pickle for IPC); hooks are dropped by
serialization, which matches the summary generator's needs (it works from
cached recordings, or re-renders hook-free scenes).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from sofaopt.core.sofa_bootstrap import bootstrap_sofa_env, reconfigure_streams_utf8
from sofaopt.project import project_from_jsonable


def main(args_path: str) -> None:
    reconfigure_streams_utf8()
    print("[summary-gen] Started", flush=True)

    root = bootstrap_sofa_env()
    print(f"[summary-gen] SOFA root: {root or '(not found)'}", flush=True)

    args = json.loads(Path(args_path).read_text(encoding="utf-8"))
    project = project_from_jsonable(args["project"])

    from sofaopt.video import generate_summary_video

    generate_summary_video(
        project,
        Path(args["summary_path"]),
        top_n=int(args.get("top_n", project.record_summary_top_n)),
        bottom_n=int(args.get("bottom_n", project.record_summary_bottom_n)),
        text_overlay=True,
    )
    print("[summary-gen] Done.", flush=True)
    Path(args_path).unlink(missing_ok=True)

    # Skip Python finalization: SOFA's bindings crash in Py_FinalizeEx on
    # Windows (same workaround as scene/runner.py).
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m sofaopt.video.summary_entry <args.json>")
    main(sys.argv[1])
