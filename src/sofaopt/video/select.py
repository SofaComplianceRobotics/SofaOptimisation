"""Trial selection and multi-clip outputs: batch videos, summary reel, pruning.

Trials are ranked by the *recorded* study score via
:mod:`sofaopt.core.results` — never a recomputed one (§11).
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from sofaopt.core.results import load_trial_records, rank_completed
from sofaopt.video.ffmpeg import apply_text_overlay, concat_videos
from sofaopt.video.overlay import format_param_line
from sofaopt.video.single import generate_trial_video


def _rank_trials(project) -> list[dict]:
    """All completed trial records, best→worst by the recorded final score."""
    return rank_completed(load_trial_records(project.trials_dir))


def _select_top_bottom(ranked: list[dict], top_n: int, bottom_n: int) -> list[dict]:
    selected: list[dict] = []
    seen: set[tuple] = set()

    def _add(r: dict) -> None:
        key = (r["gen_name"], r["trial_name"])
        if key not in seen:
            seen.add(key)
            selected.append(r)

    for r in ranked[:top_n]:
        _add(r)
    for r in ranked[-bottom_n:] if bottom_n else []:
        _add(r)
    return selected


def generate_selected_videos(
    project,
    output_dir: Path | str,
    *,
    top_n: int = 5,
    bottom_n: int = 3,
    score_above: float | None = None,
    score_below: float | None = None,
    test_name: str | None = None,
    **video_kwargs,
) -> list[Path]:
    """Generate videos for the top/bottom N trials (or by score threshold).

    Args:
        project:     The SofaOptProject whose runtime data to scan.
        output_dir:  Directory for output MP4 files.
        top_n:       Generate for this many highest-scoring trials.
        bottom_n:    Generate for this many lowest-scoring trials.
        score_above: Also include all trials with score > this value.
        score_below: Also include all trials with score < this value.
        **video_kwargs: Forwarded to ``generate_trial_video()`` (crf, preset,
                        frame_skip, text_overlay, width, height, fps, ...).

    Returns:
        List of paths to generated MP4 files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ranked = _rank_trials(project)
    if not ranked:
        print("[video] No completed trials found.")
        return []

    selected = _select_top_bottom(ranked, top_n, bottom_n)
    seen = {(r["gen_name"], r["trial_name"]) for r in selected}
    for r in ranked:
        score = r["final_score"]
        key = (r["gen_name"], r["trial_name"])
        if key in seen:
            continue
        if (score_above is not None and score > score_above) or (
            score_below is not None and score < score_below
        ):
            seen.add(key)
            selected.append(r)

    outputs: list[Path] = []
    for i, rec in enumerate(selected, 1):
        score = rec["final_score"]
        fname = f"{rec['gen_name']}_{rec['trial_name']}_score{score:.1f}.mp4"
        out = output_dir / fname
        trial_dir = project.trials_dir / rec["gen_name"] / rec["trial_name"]
        print(
            f"[video] [{i}/{len(selected)}] "
            f"{rec['gen_name']}/{rec['trial_name']} score={score:.1f}"
        )
        cached = trial_dir / "trial.mp4"
        if cached.exists():
            print("[video]   -> using cached recording")
            shutil.copy2(cached, out)
            outputs.append(out)
            continue
        video_kwargs.setdefault("text_overlay", True)
        try:
            generate_trial_video(
                project,
                trial_dir,
                out,
                test_name=test_name,
                _overlay_meta={
                    "gen_name": rec["gen_name"],
                    "trial_name": rec["trial_name"],
                    "score": score,
                },
                **video_kwargs,
            )
            outputs.append(out)
        except Exception as exc:
            print(f"[video]   -> failed: {exc}")

    print(f"[video] Generated {len(outputs)}/{len(selected)} videos in {output_dir}")
    return outputs


def generate_summary_video(
    project,
    output_path: Path | str,
    *,
    top_n: int = 3,
    bottom_n: int = 2,
    test_name: str | None = None,
    clip_steps: int | None = None,
    **video_kwargs,
) -> None:
    """Generate a single highlight-reel MP4 from the best and worst trials.

    When cached ``trial.mp4`` recordings exist (produced during the run with
    ``record_frames=True``), uses those directly — fast, no SOFA re-run needed.
    Otherwise re-runs each trial's scene to render the video.

    Text overlay (gen / trial / score / params) is burned in via ffmpeg
    drawtext so the summary clip is self-explanatory without the dashboard.

    Args:
        project:     The SofaOptProject whose runtime data to scan.
        output_path: Output MP4 path.
        top_n:       Number of highest-scoring trials to include.
        bottom_n:    Number of lowest-scoring trials to include.
        clip_steps:  Max sim steps per re-rendered clip (ignored for cached
                     recordings).
        **video_kwargs: Forwarded to ``generate_trial_video()`` when
                     re-rendering (crf, preset, frame_skip, width, ...).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ranked = _rank_trials(project)
    if not ranked:
        print("[video] No completed trials found.")
        return

    clips = _select_top_bottom(ranked, top_n, bottom_n)

    video_kwargs.setdefault("text_overlay", True)
    text_overlay = video_kwargs.pop("text_overlay", True)
    crf = video_kwargs.pop("crf", 28)
    preset = video_kwargs.pop("preset", "fast")

    def _overlay_line1(rec: dict) -> str:
        label = "best" if clips.index(rec) < top_n else "worst"
        return (
            f"{rec['gen_name']} / {rec['trial_name']}  [{label}]  "
            f"score: {rec['final_score']:.1f}"
        )

    def _overlay_line2(rec: dict) -> str:
        return format_param_line(rec.get("params") or {})

    # Fast path: all selected trials have cached in-run recordings.
    cached_clips = [
        project.trials_dir / r["gen_name"] / r["trial_name"] / "trial.mp4"
        for r in clips
    ]
    if all(p.exists() for p in cached_clips):
        print(f"[video] Using {len(cached_clips)} cached recordings -> {output_path}")
        if text_overlay:
            tmp_dir = Path(tempfile.mkdtemp(prefix="sofaopt_sumovl_"))
            try:
                overlaid: list[Path] = []
                for i, (clip_path, rec) in enumerate(zip(cached_clips, clips)):
                    ovl = tmp_dir / f"clip_{i:02d}.mp4"
                    try:
                        apply_text_overlay(
                            clip_path, ovl, _overlay_line1(rec), _overlay_line2(rec),
                            crf=crf, preset=preset,
                        )
                        overlaid.append(ovl)
                    except Exception as exc:
                        print(
                            f"[video]   overlay failed for clip {i}: {exc} "
                            "-- using raw clip"
                        )
                        overlaid.append(clip_path)
                concat_videos(overlaid, output_path, crf=crf, preset=preset)
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)
        else:
            concat_videos(cached_clips, output_path, crf=crf, preset=preset)
        print(f"[video] Summary saved: {output_path}")
        return

    # Slow path: re-render each trial that has no cached recording.
    tmp_dir = Path(tempfile.mkdtemp(prefix="sofaopt_summary_"))
    clip_paths: list[Path] = []
    try:
        for i, rec in enumerate(clips, 1):
            score = rec["final_score"]
            label = "best" if i <= top_n else "worst"
            trial_dir = project.trials_dir / rec["gen_name"] / rec["trial_name"]
            cached = trial_dir / "trial.mp4"
            print(
                f"[video] Clip {i}/{len(clips)} ({label}): "
                f"{rec['gen_name']}/{rec['trial_name']} score={score:.1f}"
            )
            clip_out = tmp_dir / f"clip_{i:02d}_{label}.mp4"
            if cached.exists():
                print("[video]   -> using cached recording")
                if text_overlay:
                    try:
                        apply_text_overlay(
                            cached, clip_out, _overlay_line1(rec), _overlay_line2(rec),
                            crf=crf, preset=preset,
                        )
                    except Exception:
                        shutil.copy2(cached, clip_out)
                else:
                    shutil.copy2(cached, clip_out)
                clip_paths.append(clip_out)
                continue
            try:
                generate_trial_video(
                    project,
                    trial_dir,
                    clip_out,
                    test_name=test_name,
                    max_steps=clip_steps,
                    crf=crf,
                    preset=preset,
                    text_overlay=text_overlay,
                    _overlay_meta={
                        "gen_name": rec["gen_name"],
                        "trial_name": rec["trial_name"],
                        "score": score,
                    },
                    **video_kwargs,
                )
                clip_paths.append(clip_out)
            except Exception as exc:
                print(f"[video]   -> failed: {exc}")

        if not clip_paths:
            raise RuntimeError("No clips were generated; cannot create summary.")

        print(f"[video] Concatenating {len(clip_paths)} clips -> {output_path}")
        concat_videos(clip_paths, output_path, crf=crf, preset=preset)
        print(f"[video] Summary saved: {output_path}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def cleanup_trial_recordings(
    project,
    *,
    keep_top_n: int = 5,
    keep_bottom_n: int = 3,
) -> tuple[int, int]:
    """Delete cached ``trial.mp4`` files, keeping only top/bottom N trials.

    After a long optimization run with ``record_frames=True`` every trial
    produces a ``trial.mp4``. This deletes the ones neither in the top
    ``keep_top_n`` nor the bottom ``keep_bottom_n`` by score, freeing disk
    space while preserving the most informative recordings.

    Returns:
        ``(kept, deleted)`` counts.
    """
    ranked = _rank_trials(project)
    if not ranked:
        print("[video] No completed trials found.")
        return 0, 0

    keep_keys = {
        (r["gen_name"], r["trial_name"])
        for r in _select_top_bottom(ranked, keep_top_n, keep_bottom_n)
    }

    kept = 0
    deleted = 0
    for rec in ranked:
        mp4 = project.trials_dir / rec["gen_name"] / rec["trial_name"] / "trial.mp4"
        if not mp4.exists():
            continue
        if (rec["gen_name"], rec["trial_name"]) in keep_keys:
            kept += 1
        else:
            mp4.unlink()
            deleted += 1

    print(f"[video] Recordings: kept {kept}, deleted {deleted}.")
    return kept, deleted


def apply_generation_overlays(project, gen: int) -> None:
    """Burn gen/trial/score/params text into each trial.mp4 of a generation.

    Called by the orchestrator after run_generation() finishes. FrameRecorder
    streams raw frames at capture time and has no score info, so the overlay
    must be applied here, once trial_state.json has been written.
    Silently skips trials with no trial.mp4 or missing metadata.
    """
    gen_dir = project.trials_dir / f"gen_{gen:04d}"
    if not gen_dir.is_dir():
        return
    for trial_dir in sorted(gen_dir.iterdir()):
        if not trial_dir.is_dir():
            continue
        mp4 = trial_dir / "trial.mp4"
        if not mp4.exists():
            continue
        try:
            params: dict = {}
            params_path = trial_dir / "params.json"
            if params_path.exists():
                params = json.loads(params_path.read_text(encoding="utf-8"))
            score: float | None = None
            state_path = trial_dir / "trial_state.json"
            if state_path.exists():
                state_data = json.loads(state_path.read_text(encoding="utf-8"))
                score = state_data.get("final_score")
            line1 = f"gen_{gen:04d} / {trial_dir.name}"
            if score is not None:
                line1 += f"  score: {score:.1f}"
            line2 = format_param_line(params)
            tmp = mp4.with_name(f"_{mp4.stem}_overlay_tmp.mp4")
            apply_text_overlay(mp4, tmp, line1, line2, crf=32, preset="ultrafast")
            tmp.replace(mp4)
        except Exception as exc:
            print(f"[video] Overlay failed for {trial_dir.name}: {exc}")
