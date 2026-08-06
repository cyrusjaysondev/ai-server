#!/usr/bin/env python3
"""Run resumable, direct-pod approval renders for every Viral Dance template.

This intentionally accepts only a persistent pod base URL. It does not know
about or call the RunPod serverless API.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


PROFILE_SETTING_DEFAULTS: dict[str, float | int | bool] = {
    "duration_seconds": 4.0,
    "motion_strength": 1.0,
    "seed": -1,
    "reference_start_seconds": 0.0,
    "auto_select_motion_window": True,
}


def resolve_profile_settings(
    profile: dict[str, Any],
    *,
    duration_seconds: float | None = None,
    motion_strength: float | None = None,
    seed: int | None = None,
    reference_start_seconds: float | None = None,
    auto_select_motion_window: bool | None = None,
) -> dict[str, float | int | bool]:
    """Resolve production profile values, applying only explicit CLI overrides."""

    def profile_value(key: str, fallback: float | int | bool) -> Any:
        value = profile.get(key)
        return fallback if value is None else value

    return {
        "duration_seconds": float(
            profile_value("durationSeconds", PROFILE_SETTING_DEFAULTS["duration_seconds"])
            if duration_seconds is None
            else duration_seconds
        ),
        "motion_strength": float(
            profile_value("motionStrength", PROFILE_SETTING_DEFAULTS["motion_strength"])
            if motion_strength is None
            else motion_strength
        ),
        "seed": int(
            profile_value("seed", PROFILE_SETTING_DEFAULTS["seed"])
            if seed is None
            else seed
        ),
        "reference_start_seconds": float(
            profile_value(
                "referenceStartSeconds",
                PROFILE_SETTING_DEFAULTS["reference_start_seconds"],
            )
            if reference_start_seconds is None
            else reference_start_seconds
        ),
        "auto_select_motion_window": bool(
            profile_value(
                "autoSelectMotionWindow",
                PROFILE_SETTING_DEFAULTS["auto_select_motion_window"],
            )
            if auto_select_motion_window is None
            else auto_select_motion_window
        ),
    }


def selected_templates_approved(report: dict[str, Any], selected_keys: list[str]) -> bool:
    """Return true only when every selected template has an automatic pass."""

    return bool(selected_keys) and all(
        report.get(template_key, {}).get("automatic_approval") is True
        for template_key in selected_keys
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--source-image", type=Path, required=True)
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument("--motion-strength", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--reference-start-seconds", type=float)
    motion_window = parser.add_mutually_exclusive_group()
    motion_window.add_argument(
        "--auto-select-motion-window",
        dest="auto_select_motion_window",
        action="store_true",
    )
    motion_window.add_argument(
        "--no-auto-select-motion-window",
        dest="auto_select_motion_window",
        action="store_false",
    )
    parser.set_defaults(auto_select_motion_window=None)
    parser.add_argument("--force", action="store_true")
    return parser


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def fetch_json(url: str, timeout: float = 30.0) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "Pandie-Viral-Dance-QA/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Pandie-Viral-Dance-QA/1.0"})
    with urllib.request.urlopen(request, timeout=180.0) as response:
        destination.write_bytes(response.read())


def wait_for_video_capacity(base_url: str) -> None:
    last_message = ""
    while True:
        try:
            health = fetch_json(f"{base_url}/health", timeout=15.0)
            if health.get("status") == "ready" and health.get("video_capacity_available"):
                return
            message = (
                f"pod busy: active_video_jobs={health.get('active_video_jobs')} "
                f"ready={health.get('status') == 'ready'}"
            )
        except Exception as error:  # live harness must survive a pod restart
            message = f"waiting for pod health: {error}"
        if message != last_message:
            print(message, flush=True)
            last_message = message
        time.sleep(10)


def submit_job(
    base_url: str,
    source_image: Path,
    reference_video: Path,
    prompt: str,
    *,
    duration_seconds: float,
    motion_strength: float,
    seed: int,
    reference_start_seconds: float,
    auto_select_motion_window: bool,
) -> dict[str, Any]:
    command = [
        "curl", "-fsS", "--max-time", "180", "-X", "POST", f"{base_url}/ltx/motion",
        "-F", f"image=@{source_image}",
        "-F", f"reference_video=@{reference_video}",
        "-F", f"prompt={prompt}",
        "-F", "preset=fast",
        "-F", "aspect_ratio=9:16",
        "-F", "width=544",
        "-F", "height=960",
        "-F", "length=121",
        "-F", "fps=24",
        "-F", "match_reference_duration=false",
        "-F", f"max_duration_seconds={duration_seconds}",
        "-F", f"reference_start_seconds={reference_start_seconds}",
        "-F", f"auto_select_motion_window={str(auto_select_motion_window).lower()}",
        "-F", "audio=false",
        "-F", "enhance_prompt=true",
        "-F", "inplace_strength=1",
        "-F", f"motion_strength={motion_strength}",
        "-F", f"seed={seed}",
        "-F", "require_full_body=true",
        "-F", "require_detectable_face=false",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def poll_job(poll_url: str) -> dict[str, Any]:
    last_marker: tuple[Any, Any, Any] | None = None
    transient_errors = 0
    while True:
        try:
            status = fetch_json(poll_url, timeout=45.0)
            transient_errors = 0
        except (TimeoutError, urllib.error.URLError) as error:
            transient_errors += 1
            if transient_errors > 12:
                raise
            print(f"  transient poll error ({transient_errors}/12): {error}", flush=True)
            time.sleep(8)
            continue
        marker = (status.get("stage"), status.get("progress"), status.get("segment"))
        if marker != last_marker:
            print(
                f"  {status.get('stage', status.get('status'))}: "
                f"{status.get('progress', 0)}% "
                f"{status.get('progress_message', '')}",
                flush=True,
            )
            last_marker = marker
        if status.get("status") in {"completed", "failed", "cancelled"}:
            return status
        time.sleep(6)


def measure_motion(video: Path) -> dict[str, float | int]:
    width, height = 144, 240
    command = [
        "ffmpeg", "-v", "error", "-i", str(video),
        "-vf", f"fps=5,scale={width}:{height},format=gray",
        "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ]
    raw = subprocess.run(command, check=True, capture_output=True).stdout
    frame_size = width * height
    frame_count = len(raw) // frame_size
    if frame_count < 2:
        return {"sampled_frames": frame_count, "mean_frame_delta": 0.0, "moving_frame_fraction": 0.0}
    frames = [memoryview(raw)[index * frame_size : (index + 1) * frame_size] for index in range(frame_count)]
    deltas = [
        sum(abs(current - previous) for current, previous in zip(frames[index], frames[index - 1])) / frame_size
        for index in range(1, frame_count)
    ]
    return {
        "sampled_frames": frame_count,
        "mean_frame_delta": round(sum(deltas) / len(deltas), 4),
        "moving_frame_fraction": round(sum(delta >= 0.75 for delta in deltas) / len(deltas), 4),
    }


def create_contact_sheet(video: Path, output: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y", "-i", str(video),
            "-vf",
            "select='eq(n,0)+eq(n,37)+eq(n,74)+eq(n,111)',"
            "scale=240:-2:force_original_aspect_ratio=decrease,"
            "pad=240:426:(ow-iw)/2:(oh-ih)/2:black,tile=4x1:padding=2:margin=2",
            "-frames:v", "1", str(output),
        ],
        check=True,
    )


def approval_result(
    final: dict[str, Any],
    motion: dict[str, float | int],
    expected_duration_seconds: float,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    identity = final.get("identity_quality") or {}
    if not identity.get("passed"):
        reasons.append(f"identity gate: {identity.get('reason') or 'failed'}")
    if float(motion.get("mean_frame_delta", 0.0)) < 0.75:
        reasons.append("motion is too weak or nearly static")
    if float(motion.get("moving_frame_fraction", 0.0)) < 0.7:
        reasons.append("motion is not sustained across the clip")
    minimum_duration = max(1.0, expected_duration_seconds - 0.25)
    if float(final.get("media_duration_seconds") or 0.0) < minimum_duration:
        reasons.append("output is shorter than the approved action window")
    return not reasons, reasons


def main() -> int:
    args = build_argument_parser().parse_args()

    base_url = args.base_url.rstrip("/")
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    references_dir = output_dir / "references"
    videos_dir = output_dir / "videos"
    sheets_dir = output_dir / "contact-sheets"
    for directory in (references_dir, videos_dir, sheets_dir):
        directory.mkdir(exist_ok=True)

    profiles = read_json(args.profiles)
    canonical_templates = {item["template_key"]: item for item in read_json(args.templates)}
    report_path = output_dir / "report.json"
    report = read_json(report_path) if report_path.exists() else {}

    missing = sorted(set(profiles) - set(canonical_templates))
    if missing:
        raise SystemExit(f"canonical template export is missing: {', '.join(missing)}")

    requested = {
        item.strip()
        for value in args.only
        for item in value.split(",")
        if item.strip()
    }
    selected_profiles = [
        (template_key, profile)
        for template_key, profile in profiles.items()
        if not requested or template_key in requested
    ]
    unknown = sorted(requested - set(profiles))
    if unknown:
        raise SystemExit(f"unknown profile key(s): {', '.join(unknown)}")

    selected_keys = [template_key for template_key, _ in selected_profiles]
    total = len(selected_profiles)
    for index, (template_key, profile) in enumerate(selected_profiles, start=1):
        existing = report.get(template_key)
        if existing and not args.force:
            print(f"[{index:02d}/{total:02d}] skip existing {profile['name']}", flush=True)
            continue

        template = canonical_templates[template_key]
        effective_settings = resolve_profile_settings(
            profile,
            duration_seconds=args.duration_seconds,
            motion_strength=args.motion_strength,
            seed=args.seed,
            reference_start_seconds=args.reference_start_seconds,
            auto_select_motion_window=args.auto_select_motion_window,
        )
        print(f"[{index:02d}/{total:02d}] {profile['name']} ({template_key})", flush=True)
        reference_path = references_dir / f"{template_key}.mp4"
        if not reference_path.exists():
            download(template["sample_video_url"], reference_path)

        try:
            wait_for_video_capacity(base_url)
            submission = submit_job(
                base_url,
                args.source_image,
                reference_path,
                profile["motionPrompt"],
                duration_seconds=float(effective_settings["duration_seconds"]),
                motion_strength=float(effective_settings["motion_strength"]),
                seed=int(effective_settings["seed"]),
                reference_start_seconds=float(effective_settings["reference_start_seconds"]),
                auto_select_motion_window=bool(effective_settings["auto_select_motion_window"]),
            )
            print(
                f"  submitted {submission.get('job_id')} "
                f"window={submission.get('reference_start_seconds')}s "
                f"frames={submission.get('target_frames')}",
                flush=True,
            )
            poll_url = submission["poll_url"]
            if poll_url.startswith("/"):
                poll_url = f"{base_url}{poll_url}"
            final = poll_job(poll_url)
            item_report: dict[str, Any] = {
                "template_key": template_key,
                "name": profile["name"],
                "job_id": submission.get("job_id"),
                "submission": submission,
                "final": final,
                "status": final.get("status"),
                "effective_settings": effective_settings,
            }
            if final.get("status") == "completed":
                video_path = videos_dir / f"{template_key}.mp4"
                download(final["url"], video_path)
                motion = measure_motion(video_path)
                create_contact_sheet(video_path, sheets_dir / f"{template_key}.jpg")
                approved, reasons = approval_result(
                    final,
                    motion,
                    float(effective_settings["duration_seconds"]),
                )
                item_report.update({"motion_quality": motion, "automatic_approval": approved, "review_reasons": reasons})
                print(
                    f"  {'AUTO-PASS' if approved else 'REVIEW'} "
                    f"identity={final.get('identity_quality', {}).get('median_similarity')} "
                    f"motion={motion['mean_frame_delta']} coverage={motion['moving_frame_fraction']}",
                    flush=True,
                )
            else:
                item_report.update({"automatic_approval": False, "review_reasons": [final.get("error") or "generation failed"]})
                print(f"  FAILED {item_report['review_reasons'][0]}", flush=True)
            report[template_key] = item_report
        except Exception as error:
            report[template_key] = {
                "template_key": template_key,
                "name": profile["name"],
                "status": "harness_error",
                "automatic_approval": False,
                "review_reasons": [str(error)],
                "effective_settings": effective_settings,
            }
            print(f"  HARNESS ERROR {error}", flush=True)
        write_json(report_path, report)

    passed = sum(
        report.get(template_key, {}).get("automatic_approval") is True
        for template_key in selected_keys
    )
    print(f"COMPLETE automatic_pass={passed}/{len(selected_keys)} report={report_path}", flush=True)
    return 0 if selected_templates_approved(report, selected_keys) else 1


if __name__ == "__main__":
    raise SystemExit(main())
