"""CLI for sofaopt video generation: ``python -m sofaopt.video --help``."""

from __future__ import annotations

import argparse
from pathlib import Path

from sofaopt.project import load_project_file
from sofaopt.video import (
    cleanup_trial_recordings,
    generate_selected_videos,
    generate_summary_video,
    generate_trial_video,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate videos for sofaopt trials.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  Single trial (default):
    python -m sofaopt.video --project examples/cube_drop/project.py \\
        --trial examples/cube_drop/runtime/trials/gen_0001/trial_01 \\
        --output trial_01.mp4

  Batch (top/bottom N trials):
    python -m sofaopt.video --project examples/cube_drop/project.py \\
        --batch --top 5 --bottom 3 --output-dir runtime/videos \\
        --crf 30 --frame-skip 2

  Summary reel (clips concatenated):
    python -m sofaopt.video --project examples/cube_drop/project.py \\
        --summary --top 3 --bottom 2 --clip-steps 100 \\
        --output runtime/summary.mp4
""",
    )
    parser.add_argument(
        "--project", required=True, type=Path,
        help="Path to project.py (must define PROJECT = SofaOptProject(...))",
    )
    # Mode selection
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--batch",   action="store_true", help="Batch mode: generate for top/bottom N trials")
    mode.add_argument("--summary", action="store_true", help="Summary mode: one highlight-reel MP4")
    mode.add_argument("--cleanup", action="store_true", help="Delete cached trial.mp4s, keeping top/bottom N")

    # Single-trial args
    parser.add_argument("--trial",  type=Path, default=None, help="Trial directory (single-trial mode)")
    parser.add_argument("--output", type=Path, default=None, help="Output MP4 path")

    # Batch/summary args
    parser.add_argument("--output-dir",  type=Path, default=None, help="Output directory (batch mode)")
    parser.add_argument("--top",         type=int,  default=5,    help="Number of best trials (batch/summary)")
    parser.add_argument("--bottom",      type=int,  default=3,    help="Number of worst trials (batch/summary)")
    parser.add_argument("--score-above", type=float, default=None, help="Include trials scoring above this (batch)")
    parser.add_argument("--score-below", type=float, default=None, help="Include trials scoring below this (batch)")
    parser.add_argument("--clip-steps",  type=int,  default=150,  help="Max sim steps per re-rendered clip (summary)")

    # Common render args
    parser.add_argument("--test",       default=None, dest="test_name", help="Test name to render")
    parser.add_argument("--width",      type=int,   default=800)
    parser.add_argument("--height",     type=int,   default=600)
    parser.add_argument("--fps",        type=int,   default=30)
    parser.add_argument("--max-steps",  type=int,   default=None, help="Stop after N sim steps (single trial)")
    parser.add_argument("--crf",        type=int,   default=28,   help="ffmpeg CRF quality (lower=better, default 28)")
    parser.add_argument("--preset",     default="fast",           help="ffmpeg preset (default fast)")
    parser.add_argument("--frame-skip", type=int,   default=16,   help="Capture every Nth frame (default 16)")
    parser.add_argument("--text-overlay", action="store_true",    help="Burn trial ID, score and params into frames")

    args = parser.parse_args()
    project = load_project_file(args.project)

    common = dict(
        test_name=args.test_name,
        width=args.width,
        height=args.height,
        fps=args.fps,
        crf=args.crf,
        preset=args.preset,
        frame_skip=args.frame_skip,
        text_overlay=args.text_overlay,
    )

    if args.cleanup:
        cleanup_trial_recordings(project, keep_top_n=args.top, keep_bottom_n=args.bottom)
    elif args.batch:
        if args.output_dir is None:
            parser.error("--batch requires --output-dir")
        generate_selected_videos(
            project,
            args.output_dir,
            top_n=args.top,
            bottom_n=args.bottom,
            score_above=args.score_above,
            score_below=args.score_below,
            **common,
        )
    elif args.summary:
        if args.output is None:
            parser.error("--summary requires --output")
        generate_summary_video(
            project,
            args.output,
            top_n=args.top,
            bottom_n=args.bottom,
            clip_steps=args.clip_steps,
            **common,
        )
    else:
        if args.trial is None or args.output is None:
            parser.error("Single-trial mode requires --trial and --output")
        generate_trial_video(
            project,
            args.trial,
            args.output,
            max_steps=args.max_steps,
            **common,
        )


if __name__ == "__main__":
    main()
