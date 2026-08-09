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
import shutil
import subprocess
import urllib.request
from pathlib import Path

API_REPO_RAW = "https://raw.githubusercontent.com/cyrusjaysondev/ai-server/main"
API_DIR = Path("/workspace/api")
PINNED_API_RELEASE = API_DIR / "releases" / "0b88fac"
BACKUP_DIR = API_DIR / "backups" / "pre-shirt-keyframe-v1"
FILES_TO_REFRESH = ("main.py", "workflows.py")
# Bump this suffix to force the refresh to re-run after a subsequent push.
# We use a versioned marker so legit ComfyUI restarts after the work is
# done don't trigger another uvicorn cycle.
MARKER = Path("/tmp/api-refresh-claimed-shirt-keyframe-v1")
DIAG_LOG = Path("/workspace/setup-vhs.log")  # piggyback on the log surfaced by /admin/comfy-status


def _diag(line: str) -> None:
    """Write to the install log so /admin/comfy-status surfaces it."""
    try:
        with DIAG_LOG.open("a") as f:
            f.write(f"[refresh-shim] {line}\n")
    except Exception:
        pass


def _refresh_api_files() -> None:
    """Atomically install the tested API pair, then restart uvicorn.

    This pod's supervisor still restores ``main.py`` from release 0b88fac
    on every restart. Update that pinned copy as well as the active API
    file, while retaining one-command rollback copies of everything that
    is replaced. Download and compile both Python files before touching
    either live target so a partial or stale GitHub response cannot leave
    the service in a mixed release state.
    """
    _diag("import-time entry — shim is being loaded by ComfyUI")
    if MARKER.exists():
        _diag(f"marker {MARKER.name} present — skipping (already refreshed)")
        return
    if not API_DIR.is_dir():
        _diag(f"{API_DIR} missing — wrong pod layout, bailing")
        return

    if not PINNED_API_RELEASE.is_dir():
        _diag(f"pinned release {PINNED_API_RELEASE} missing — refusing partial deploy")
        return

    downloaded: dict[str, Path] = {}
    for filename in FILES_TO_REFRESH:
        url = f"{API_REPO_RAW}/{filename}?cb=shirt-keyframe-v1"
        tmp = API_DIR / f"{filename}.refresh-shim"
        try:
            urllib.request.urlretrieve(url, str(tmp))
            if not tmp.is_file() or tmp.stat().st_size == 0:
                raise RuntimeError("empty download")
            compile(tmp.read_text(), filename, "exec")
            downloaded[filename] = tmp
        except Exception as e:
            _diag(f"failed to fetch {filename}: {e}")
            for staged in downloaded.values():
                staged.unlink(missing_ok=True)
            tmp.unlink(missing_ok=True)
            return

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for filename, tmp in downloaded.items():
        target = API_DIR / filename
        backup = BACKUP_DIR / filename
        if target.is_file() and not backup.exists():
            shutil.copy2(target, backup)

        if filename == "main.py":
            pinned_target = PINNED_API_RELEASE / filename
            pinned_backup = BACKUP_DIR / "pinned-main.py"
            if pinned_target.is_file() and not pinned_backup.exists():
                shutil.copy2(pinned_target, pinned_backup)
            pinned_tmp = PINNED_API_RELEASE / f"{filename}.shirt-keyframe-v1"
            shutil.copy2(tmp, pinned_tmp)
            os.replace(str(pinned_tmp), str(pinned_target))

        os.replace(str(tmp), str(target))
        _diag(f"refreshed {filename} ({target.stat().st_size} bytes)")

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


NODE_CLASS_MAPPINGS: dict = {"_RefreshShimSentinel_shirt_keyframe_v1": _RefreshShimSentinel}
NODE_DISPLAY_NAME_MAPPINGS: dict = {
    "_RefreshShimSentinel_shirt_keyframe_v1": "Refresh Shim Sentinel (shirt keyframe v1)",
}
