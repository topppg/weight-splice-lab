import contextlib
import fcntl
import importlib
import json
import os
import time
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import psutil

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "runs" / ".gpu.lock"
WATCHDOG_PID = ROOT / "runs" / ".memwatch.pid"  # written by memwatch; no longer accepted alone
HEARTBEAT = ROOT / "runs" / ".memwatch.heartbeat"
HEARTBEAT_MAX_AGE_S = 10.0
WATCHDOG_MODULE = "weight_splice_lab.memwatch"
MAX_GPU_GB = 10.0
RESERVE_GB = 4.0


def available_gb() -> float:
    return psutil.virtual_memory().available / 1e9


def gpu_budget_gb() -> float:
    return max(0.0, min(MAX_GPU_GB, available_gb() - RESERVE_GB))


def is_watchdog(pid: int, create_time: float) -> bool:
    try:
        p = psutil.Process(pid)
        if abs(p.create_time() - create_time) > 1e-3:
            return False
        cmd = p.cmdline()
    except (psutil.Error, SystemError, OSError):
        return False
    return WATCHDOG_MODULE in cmd or any(Path(c).name == "memwatch.py" for c in cmd)


def watchdog_healthy(heartbeat: Path = HEARTBEAT, max_age_s: float = HEARTBEAT_MAX_AGE_S) -> bool:
    # A fresh heartbeat from the same memwatch process (PID and creation time) it names; no fallback.
    try:
        hb = json.loads(heartbeat.read_text())
        pid, created, ts = int(hb["pid"]), float(hb["create_time"]), float(hb["ts"])
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return -1.0 < time.time() - ts < max_age_s and is_watchdog(pid, created)


watchdog_running = watchdog_healthy


def _keep_wired_limit() -> None:
    return None


def patch_mlx_wired_limit() -> ModuleType:
    # mlx_lm/__init__ exports a function named generate, so `import mlx_lm.generate as g` binds the
    # function; the helper that raises the wired limit lives in the submodule.
    mod = importlib.import_module("mlx_lm.generate")
    mod.maybe_set_recommended_wired_limit = _keep_wired_limit  # type: ignore[attr-defined]
    return mod


def _apply_limits(budget_gb: float) -> None:
    limit = int(budget_gb * 1e9)
    import mlx.core as mx

    mx.set_memory_limit(limit)
    mx.set_wired_limit(min(limit, mx.get_memory_limit()))
    mx.set_cache_limit(limit // 4)
    # mlx-lm generation raises the wired limit to the device maximum, which bypasses the cap above
    patch_mlx_wired_limit()
    import torch

    if torch.backends.mps.is_available():
        torch.mps.set_per_process_memory_fraction(
            min(1.0, limit / torch.mps.recommended_max_memory())
        )


@contextlib.contextmanager
def gpu_session(min_gb: float = 2.0) -> Iterator[float]:
    if not watchdog_running():
        raise SystemExit(
            "memwatch is not running: start `uv run python -m weight_splice_lab.memwatch`"
        )
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK, "a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"another GPU job holds {LOCK}; run jobs one at a time")
        budget = gpu_budget_gb()
        if budget < min_gb:
            raise SystemExit(
                f"only {available_gb():.1f} GB available; need {min_gb + RESERVE_GB:.1f} GB"
            )
        f.seek(0)
        f.truncate()
        f.write(str(os.getpid()))
        f.flush()
        _apply_limits(budget)
        print(f"[gpu] budget {budget:.1f} GB (available {available_gb():.1f} GB)", flush=True)
        yield budget
