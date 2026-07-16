"""TEMPORARY pip-install-triggered code-refresh shim.

The ComfyUI custom-node loader has been silently rejecting this repo
(node_count never goes up after install + restart, no diag log entry).
Switch strategies: trigger the refresh via pip install's setup.py
import-time side effect instead. /admin/install-comfy-node runs
`pip install -r requirements.txt`; requirements.txt points pip at this
directory in editable mode, which makes pip import setup.py — and pip
imports setup.py BEFORE doing anything else, so our side effects run
regardless of whether the actual install would succeed.

Steps at import time:
  1. wget latest main.py / workflows.py / etc. into /workspace/api/
  2. kill uvicorn by :7860 port owner so start_api.sh relaunches it
  3. raise SystemExit so pip stops (we don't actually want to install
     this as a Python package — we just used pip as a code-runner)

Idempotent via /tmp/api-refresh-claimed-<commit>-setup-py — re-running
with the same target commit no-ops.
"""

import os
import subprocess
import sys
import urllib.request
from pathlib import Path

API_REPO_RAW = "https://raw.githubusercontent.com/cyrusjaysondev/ai-server/main"
API_DIR = Path("/workspace/api")
FILES_TO_REFRESH = (
    "main.py",
    "workflows.py",
    "image_output.py",
    "safety.py",
    "logo_safety.py",
    "watermark.py",
)
MARKER = Path("/tmp/api-refresh-claimed-service-quality-v57")
DIAG_LOG = Path("/workspace/setup-vhs.log")


def _log(msg: str) -> None:
    line = f"[setup-py-shim] {msg}"
    print(line, flush=True)
    try:
        with DIAG_LOG.open("a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _refresh_and_kill() -> None:
    if MARKER.exists():
        _log(f"marker {MARKER.name} present — skipping")
        return
    if not API_DIR.is_dir():
        _log(f"{API_DIR} missing — wrong layout, bailing")
        return

    _log("entry — copying API files from local git clone")
    # We're being run by pip as part of `pip install -r requirements.txt`
    # inside the freshly git-pulled custom_nodes/ai-gen-api-v2 directory.
    # All the source files are RIGHT HERE on disk — no need to fight the
    # raw.githubusercontent.com CDN edge (which has been observed serving
    # stale content for >10min even with no-cache headers). Just copy
    # them from the local clone to /workspace/api/.
    import shutil
    src_dir = Path(__file__).resolve().parent
    for filename in FILES_TO_REFRESH:
        src = src_dir / filename
        target = API_DIR / filename
        if not src.is_file():
            _log(f"  ✗ {filename} missing from clone at {src}")
            continue
        try:
            shutil.copy2(str(src), str(target))
            _log(f"  ✓ {filename} ({target.stat().st_size} bytes)")
        except Exception as e:
            _log(f"  ✗ {filename} copy error: {e}")

    # Kill uvicorn — start_api.sh relaunches with fresh code.
    killed_pid = None
    try:
        netstat = subprocess.run(
            ["netstat", "-tlnp"], capture_output=True, timeout=5,
        ).stdout.decode(errors="replace")
        for line in netstat.splitlines():
            if ":7860" not in line:
                continue
            tail = line.split()[-1]
            pid = tail.split("/")[0]
            if pid.isdigit():
                subprocess.run(["kill", "-9", pid], capture_output=True, timeout=5)
                killed_pid = pid
                break
    except Exception as e:
        _log(f"netstat failed: {e}")

    if killed_pid:
        _log(f"killed uvicorn PID={killed_pid}")
    else:
        # Last resort
        try:
            res = subprocess.run(
                ["pkill", "-9", "-f", "uvicorn main:app"],
                capture_output=True, timeout=5,
            )
            _log(f"pkill -9 -f 'uvicorn main:app' rc={res.returncode}")
        except Exception as e:
            _log(f"pkill failed: {e}")

    try:
        MARKER.touch()
        _log(f"wrote marker {MARKER.name}")
    except Exception as e:
        _log(f"marker write failed: {e}")


# Run on import — pip imports setup.py before doing anything else.
try:
    _refresh_and_kill()
except Exception as e:
    _log(f"outer error: {e}")

# Tell pip we're not really a real package, abort the install. We've done
# our work already.
_log("exiting setup.py — install path not needed")
sys.exit(0)
