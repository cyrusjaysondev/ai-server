"""
TEMPORARY ComfyUI custom-node shim — code-refresh trampoline.

Purpose:
   The pod's start_api.sh only fetched main.py / workflows.py ONCE at
   supervisor startup, so killing uvicorn (e.g. via /admin/install-comfy-node
   pkill) restarted with cached files. Commit 290700f moves the fetch
   into the supervisor loop — but that fix needs a container restart to
   take effect, and container restarts have been failing on GPU exhaustion.

   This shim breaks the chicken-and-egg by running INSIDE ComfyUI's
   custom_nodes loader. ComfyUI imports any directory's __init__.py at
   startup, so when this repo gets cloned into custom_nodes/ via
   /admin/install-comfy-node, ComfyUI's next launch executes this code.
   It then wgets the latest workflows.py + main.py into /workspace/api/
   and SIGKILLs uvicorn so the existing supervisor's while-loop restarts
   it with the fresh files.

   Idempotent via /tmp/api-refresh-claimed-<git-sha> marker — running a
   second time with the same target commit is a no-op. Once the in-pod
   start_api.sh has been refreshed to the wget-in-loop pattern (commit
   290700f), this whole file becomes dead weight and should be deleted
   from the repo.
"""

import os
import subprocess
import urllib.request
from pathlib import Path

API_REPO_RAW = "https://raw.githubusercontent.com/cyrusjaysondev/ai-server/main"
API_DIR = Path("/workspace/api")
FILES_TO_REFRESH = ("main.py", "workflows.py", "safety.py", "logo_safety.py", "watermark.py")
# Bump this suffix to force the refresh to re-run after a subsequent push.
# We use a versioned marker so legit ComfyUI restarts after the work is
# done don't trigger another uvicorn cycle.
MARKER = Path("/tmp/api-refresh-claimed-290700f-v2")
DIAG_LOG = Path("/workspace/setup-vhs.log")  # piggyback on the log surfaced by /admin/comfy-status


def _diag(line: str) -> None:
    """Write to the install log so /admin/comfy-status surfaces it."""
    try:
        with DIAG_LOG.open("a") as f:
            f.write(f"[refresh-shim] {line}\n")
    except Exception:
        pass


def _refresh_api_files() -> None:
    """Fetch latest API .py files from main + kill uvicorn so it reloads."""
    _diag("import-time entry — shim is being loaded by ComfyUI")
    if MARKER.exists():
        _diag(f"marker {MARKER.name} present — skipping (already refreshed)")
        return
    if not API_DIR.is_dir():
        _diag(f"{API_DIR} missing — wrong pod layout, bailing")
        return

    for filename in FILES_TO_REFRESH:
        url = f"{API_REPO_RAW}/{filename}"
        tmp = API_DIR / f"{filename}.refresh-shim"
        target = API_DIR / filename
        try:
            urllib.request.urlretrieve(url, str(tmp))
            if tmp.stat().st_size > 0:
                os.replace(str(tmp), str(target))
                _diag(f"refreshed {filename} ({target.stat().st_size} bytes)")
            else:
                tmp.unlink(missing_ok=True)
                _diag(f"empty download for {filename} — kept existing")
        except Exception as e:
            _diag(f"failed to fetch {filename}: {e}")
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    # Kill uvicorn — start_api.sh's supervisor relaunches it within ~5s.
    # Target by port owner (mirrors start_api.sh's own stale-PID logic)
    # so we don't accidentally match the caller bash with "uvicorn" in
    # its env vars. Fall back to a pkill -f if netstat fails.
    killed = False
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
                _diag(f"killed uvicorn PID={pid} (port :7860 owner)")
                killed = True
                break
    except Exception as e:
        print(f"[refresh-api-shim] netstat kill failed: {e}")

    if not killed:
        # Last resort — pattern match. Less precise but works when netstat
        # isn't available.
        try:
            subprocess.run(
                ["pkill", "-9", "-f", "uvicorn main:app"],
                capture_output=True, timeout=5,
            )
            print("[refresh-api-shim] killed uvicorn via pkill -f")
        except Exception as e:
            print(f"[refresh-api-shim] pkill also failed: {e}")

    try:
        MARKER.touch()
    except Exception as e:
        print(f"[refresh-api-shim] couldn't write marker: {e}")


# Run once at import time. ComfyUI imports __init__.py during custom-node
# discovery on startup; any side effects we want happen here.
try:
    _refresh_api_files()
except Exception as e:
    print(f"[refresh-api-shim] outer error: {e}")


# Required by ComfyUI's custom-node loader. We register a single no-op
# sentinel node — exposing zero nodes was indistinguishable from "ComfyUI
# never loaded this module," but a registered node shows up in
# /admin/comfy-status's loaded_node_count + (if added to the watchlist)
# key_nodes_loaded. Confirms the import ran.
class _RefreshShimSentinel:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}
    RETURN_TYPES = ()
    FUNCTION = "noop"
    CATEGORY = "_internal/RefreshShim"
    def noop(self):
        return ()


NODE_CLASS_MAPPINGS: dict = {"_RefreshShimSentinel_290700f_v2": _RefreshShimSentinel}
NODE_DISPLAY_NAME_MAPPINGS: dict = {
    "_RefreshShimSentinel_290700f_v2": "Refresh Shim Sentinel (delete me)",
}
