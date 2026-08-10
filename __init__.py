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

import ctypes
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path

SOURCE_SHA = "544c6a5925178095a60fb5184c77291ccc6a90bd"
API_REPO_RAW = f"https://raw.githubusercontent.com/cyrusjaysondev/ai-server/{SOURCE_SHA}"
API_DIR = Path("/workspace/api")
PINNED_API_RELEASE = API_DIR / "releases" / "6a098a5"
BACKUP_DIR = API_DIR / "backups" / "pre-shirt-final-only-v3"
FILES_TO_REFRESH = ("main.py", "workflows.py")
# Bump this suffix to force the refresh to re-run after a subsequent push.
# We use a versioned marker so legit ComfyUI restarts after the work is
# done don't trigger another uvicorn cycle.
MARKER = Path("/tmp/api-refresh-claimed-shirt-final-only-v3")
DIAG_LOG = Path("/workspace/setup-vhs.log")  # piggyback on the log surfaced by /admin/comfy-status


def _diag(line: str) -> None:
    """Write to the install log so /admin/comfy-status surfaces it."""
    try:
        with DIAG_LOG.open("a") as f:
            f.write(f"[refresh-shim] {line}\n")
    except Exception:
        pass


def _exchange_directories(left: Path, right: Path) -> None:
    """Atomically exchange two directories on the Linux pod.

    Replacing ``main.py`` and ``workflows.py`` one at a time would leave a
    window where the supervisor could restore a mixed release. Linux
    ``renameat2(RENAME_EXCHANGE)`` swaps the fully staged and current release
    directories in one filesystem operation. Fail closed when the primitive is
    unavailable instead of falling back to a non-atomic pair update.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("renameat2 is unavailable; refusing non-atomic release update")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_exchange = 2
    result = renameat2(
        at_fdcwd,
        os.fsencode(left),
        at_fdcwd,
        os.fsencode(right),
        rename_exchange,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))


def _refresh_api_files() -> None:
    """Atomically install the tested API pair, then restart uvicorn.

    This pod's supervisor still restores API modules from release 6a098a5 on
    every restart. Stage and atomically swap a complete pinned release holding
    both ``main.py`` and ``workflows.py``, then update the active copies while
    retaining one-command rollback backups. Download and compile both Python
    files before touching either live target so a partial or stale GitHub
    response cannot leave the service in a mixed release state.
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
        url = f"{API_REPO_RAW}/{filename}?cb=shirt-final-only-v3"
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

    staged_release = PINNED_API_RELEASE.with_name(
        f".{PINNED_API_RELEASE.name}.shirt-final-only-v3"
    )
    release_backup = BACKUP_DIR / f"release-{PINNED_API_RELEASE.name}"
    active_backups: dict[str, Path | None] = {}
    pinned_swapped = False
    replaced_active: list[str] = []
    try:
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        if staged_release.exists():
            shutil.rmtree(staged_release)
        shutil.copytree(PINNED_API_RELEASE, staged_release)

        # Prepare backups and the complete replacement release before changing
        # either the pinned or active modules.
        for filename, tmp in downloaded.items():
            active_target = API_DIR / filename
            active_backup = BACKUP_DIR / f"active-{filename}"
            if active_target.is_file():
                if not active_backup.exists():
                    shutil.copy2(active_target, active_backup)
                active_backups[filename] = active_backup
            else:
                active_backups[filename] = None

            pinned_target = PINNED_API_RELEASE / filename
            pinned_backup = BACKUP_DIR / f"pinned-{filename}"
            if pinned_target.is_file() and not pinned_backup.exists():
                shutil.copy2(pinned_target, pinned_backup)

            staged_target = staged_release / filename
            staged_tmp = staged_release / f".{filename}.shirt-final-only-v3"
            shutil.copy2(tmp, staged_tmp)
            os.replace(str(staged_tmp), str(staged_target))

        # The supervisor can observe either the complete old release or the
        # complete SHA-pinned replacement, never a main/workflows mixture.
        _exchange_directories(PINNED_API_RELEASE, staged_release)
        pinned_swapped = True

        for filename, tmp in downloaded.items():
            active_target = API_DIR / filename
            os.replace(str(tmp), str(active_target))
            replaced_active.append(filename)
            _diag(f"refreshed {filename} ({active_target.stat().st_size} bytes)")

        # After the exchange, staged_release contains the complete old pinned
        # directory. Retain it as an additional directory-level rollback.
        if not release_backup.exists():
            os.replace(str(staged_release), str(release_backup))
        else:
            shutil.rmtree(staged_release)
        _diag(f"pinned release atomically refreshed from {SOURCE_SHA}")
    except Exception as e:
        _diag(f"atomic API refresh failed: {e}")
        for filename in reversed(replaced_active):
            try:
                active_target = API_DIR / filename
                active_backup = active_backups[filename]
                if active_backup is None:
                    active_target.unlink(missing_ok=True)
                    continue
                rollback_tmp = API_DIR / f".{filename}.shirt-final-only-v3-rollback"
                shutil.copy2(active_backup, rollback_tmp)
                os.replace(str(rollback_tmp), str(active_target))
            except Exception as rollback_error:
                _diag(f"active {filename} rollback failed: {rollback_error}")
        if pinned_swapped and staged_release.exists():
            try:
                _exchange_directories(PINNED_API_RELEASE, staged_release)
                pinned_swapped = False
            except Exception as rollback_error:
                _diag(f"CRITICAL: pinned release rollback failed: {rollback_error}")
        if not pinned_swapped and staged_release.exists():
            shutil.rmtree(staged_release)
        elif pinned_swapped:
            _diag(f"preserving original pinned release at {staged_release}")
        for tmp in downloaded.values():
            tmp.unlink(missing_ok=True)
        return

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


NODE_CLASS_MAPPINGS: dict = {"_RefreshShimSentinel_shirt_final_only_v3": _RefreshShimSentinel}
NODE_DISPLAY_NAME_MAPPINGS: dict = {
    "_RefreshShimSentinel_shirt_final_only_v3": "Refresh Shim Sentinel (shirt final-only v3)",
}
