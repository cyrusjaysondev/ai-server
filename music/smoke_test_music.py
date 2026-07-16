#!/usr/bin/env python3
"""Small ACE-Step music worker smoke test.

Submits one short text-to-music job, polls until completion, prints the lyrics
fields, and optionally downloads the generated audio file.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def request_json(base_url: str, path: str, payload: dict[str, Any] | None, api_key: str | None) -> dict[str, Any]:
    url = base_url.rstrip("/") + path
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "ai-gen-api-v2-music-smoke-test/1.0",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="GET" if payload is None else "POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{exc.code} {exc.reason}: {body}") from exc


def download_file(base_url: str, relative_url: str, output: Path, api_key: str | None) -> None:
    file_url = relative_url if relative_url.startswith("http") else base_url.rstrip("/") + relative_url
    headers = {"User-Agent": "ai-gen-api-v2-music-smoke-test/1.0"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(file_url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=300) as response:
        output.write_bytes(response.read())


def parse_result_item(result_value: Any) -> dict[str, Any]:
    if isinstance(result_value, str):
        parsed = json.loads(result_value)
    else:
        parsed = result_value
    if not isinstance(parsed, list) or not parsed:
        raise RuntimeError(f"Unexpected result payload: {parsed!r}")
    if not isinstance(parsed[0], dict):
        raise RuntimeError(f"Unexpected result item: {parsed[0]!r}")
    return parsed[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test the RunPod ACE-Step music API")
    parser.add_argument("--base-url", required=True, help="Example: https://POD_ID-8001.proxy.runpod.net")
    parser.add_argument("--api-key", default=None, help="Bearer token if ACESTEP_API_KEY is enabled")
    parser.add_argument("--prompt", default="upbeat pop jingle, bright synths, clean vocal, radio ready")
    parser.add_argument(
        "--lyrics",
        default="[Verse]\nWe build the spark\nWe light the way\n[Chorus]\nCreate the sound\nAnd press play",
    )
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--poll-interval", type=float, default=3.0)
    parser.add_argument("--output", default="music-smoke-test.mp3")
    args = parser.parse_args()

    start = time.monotonic()
    health = request_json(args.base_url, "/health", None, args.api_key)
    print(json.dumps({"health": health}, indent=2, ensure_ascii=False))

    payload = {
        "prompt": args.prompt,
        "lyrics": args.lyrics,
        "audio_duration": args.duration,
        "inference_steps": args.steps,
        "batch_size": 1,
        "audio_format": "mp3",
        "thinking": False,
        "use_random_seed": True,
    }
    released = request_json(args.base_url, "/release_task", payload, args.api_key)
    task_id = (released.get("data") or {}).get("task_id")
    if not task_id:
        raise RuntimeError(f"No task_id returned: {released}")
    print(json.dumps({"task_id": task_id}, indent=2))

    deadline = time.monotonic() + args.timeout
    last_status: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        try:
            queried = request_json(args.base_url, "/query_result", {"task_id_list": [task_id]}, args.api_key)
        except (TimeoutError, socket.timeout) as exc:
            print(json.dumps({"status": "poll_timeout", "message": str(exc)}, ensure_ascii=False))
            time.sleep(args.poll_interval)
            continue
        rows = queried.get("data") or []
        if rows:
            last_status = rows[0]
            status = last_status.get("status")
            print(json.dumps({"status": status, "progress_text": last_status.get("progress_text")}, ensure_ascii=False))
            if status == 1:
                item = parse_result_item(last_status.get("result"))
                file_path = item.get("file")
                output = Path(args.output).resolve()
                if file_path:
                    download_file(args.base_url, file_path, output, args.api_key)
                summary = {
                    "task_id": task_id,
                    "elapsed_seconds": round(time.monotonic() - start, 2),
                    "audio_url": file_path,
                    "downloaded_to": str(output) if file_path else None,
                    "lyrics": item.get("lyrics"),
                    "original_lyrics": (item.get("metas") or {}).get("lyrics"),
                    "metas": item.get("metas"),
                }
                print(json.dumps(summary, indent=2, ensure_ascii=False))
                return 0
            if status == 2:
                print(json.dumps(last_status, indent=2, ensure_ascii=False), file=sys.stderr)
                return 2
        time.sleep(args.poll_interval)

    print(json.dumps({"error": "Timed out waiting for task", "last_status": last_status}, indent=2, ensure_ascii=False), file=sys.stderr)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
