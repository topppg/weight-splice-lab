import argparse
import ctypes
import ctypes.util
import json
import os
import signal
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

import psutil

from weight_splice_lab.gpu import HEARTBEAT, ROOT, WATCHDOG_PID, available_gb

VENV = str(ROOT / ".venv")
RUN_ROOTS = (str(ROOT), str(Path.home() / "wsl-runs"))
LOG = ROOT / "runs" / "memwatch.log"
PRESSURE_CRITICAL = 4

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)


def pressure_level() -> int:
    # no subprocess: fork/exec can fail or stall under the very pressure being measured
    val = ctypes.c_int(0)
    size = ctypes.c_size_t(ctypes.sizeof(val))
    name = b"kern.memorystatus_vm_pressure_level"
    if _libc.sysctlbyname(name, ctypes.byref(val), ctypes.byref(size), None, ctypes.c_size_t(0)):
        raise OSError(ctypes.get_errno(), "sysctlbyname failed")
    return val.value


def _is_repo_root(p: psutil.Process) -> bool:
    try:
        cmd = p.cmdline()
        if (cmd and cmd[0].startswith(VENV)) or p.exe().startswith(VENV):
            return True
        if "python" in p.name().lower():
            cwd = p.cwd()
            return any(cwd == r or cwd.startswith(r + os.sep) for r in RUN_ROOTS)
    # psutil on macOS can raise SystemError for processes exiting mid-read
    except (psutil.Error, SystemError, OSError):
        pass
    return False


def repo_processes(exclude: frozenset[int] | None = None) -> list[psutil.Process]:
    skip = exclude if exclude is not None else frozenset({os.getpid()})
    found: dict[int, psutil.Process] = {}
    for p in psutil.process_iter():
        if p.pid in skip or p.pid in found or not _is_repo_root(p):
            continue
        found[p.pid] = p
        try:
            for c in p.children(recursive=True):
                found.setdefault(c.pid, c)
        except (psutil.Error, SystemError, OSError):
            continue
    return [p for pid, p in found.items() if pid not in skip]


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def kill_all(reason: str) -> None:
    # The GPU lock file is never unlinked: killed holders release it, and a new inode at the same
    # path would let two jobs hold "exclusive" locks at once.
    procs = repo_processes()
    log(f"KILL {reason}: {[p.pid for p in procs]}")
    for p in procs:
        try:
            p.send_signal(signal.SIGKILL)
        except (psutil.Error, SystemError, OSError):
            pass


def write_heartbeat(path: Path, created: float) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"pid": os.getpid(), "create_time": created, "ts": time.time()}))
    os.replace(tmp, path)


class SwapGrowth:
    def __init__(self, window_s: float) -> None:
        self.window_s = window_s
        self.hist: deque[tuple[float, float]] = deque()

    def __call__(self) -> float:
        now, swap = time.monotonic(), psutil.swap_memory().used / 1e9
        self.hist.append((now, swap))
        while self.hist[0][0] < now - self.window_s:
            self.hist.popleft()
        return swap - min(s for _, s in self.hist)


def poll(checks: list[tuple[str, Callable[[], str | None]]]) -> str | None:
    # each check runs on its own, so one failing reading cannot skip the others
    for name, check in checks:
        try:
            reason = check()
        except Exception as e:  # noqa: BLE001 - the watchdog must outlive any single bad reading
            log(f"poll error in {name}: {e!r}")
            continue
        if reason is not None:
            return reason
    return None


def swap_reason(growth: float, avail: float, growth_gb: float, below_gb: float) -> str | None:
    # swap grows when macOS pages out idle apps too; only act on it when RAM is actually short
    if growth > growth_gb and avail < below_gb:
        return f"swap grew {growth:.2f} GB with {avail:.2f} GB available"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kill-below-gb", type=float, default=2.5)
    # off by default: the editor kernel is CPU-only and GPU-denied, and macOS swap growth from other
    # apps killed healthy runs; RAM floor and critical pressure remain the hard limits
    ap.add_argument("--kill-swap-growth-gb", type=float, default=float("inf"))
    ap.add_argument("--swap-window-s", type=float, default=60.0)
    ap.add_argument("--swap-only-below-gb", type=float, default=4.0)
    ap.add_argument("--interval", type=float, default=0.5)
    a = ap.parse_args()
    created = psutil.Process().create_time()
    WATCHDOG_PID.parent.mkdir(parents=True, exist_ok=True)
    WATCHDOG_PID.write_text(str(os.getpid()))
    write_heartbeat(HEARTBEAT, created)
    log(
        f"start: kill repo processes when available < {a.kill_below_gb} GB, pressure critical,"
        f" or swap grows > {a.kill_swap_growth_gb} GB in {a.swap_window_s:.0f}s"
        f" while available < {a.swap_only_below_gb} GB"
    )
    swap_growth = SwapGrowth(a.swap_window_s)

    def low_memory() -> str | None:
        avail = available_gb()
        return f"available {avail:.2f} GB" if avail < a.kill_below_gb else None

    def pressure() -> str | None:
        level = pressure_level()
        return f"memory pressure critical (level {level})" if level >= PRESSURE_CRITICAL else None

    def swap() -> str | None:
        return swap_reason(
            swap_growth(), available_gb(), a.kill_swap_growth_gb, a.swap_only_below_gb
        )

    checks = [("available", low_memory), ("pressure", pressure), ("swap", swap)]
    try:
        while True:
            reason = poll(checks)
            if reason is not None:
                try:
                    if repo_processes():
                        kill_all(reason)
                except Exception as e:  # noqa: BLE001
                    log(f"kill error: {e!r}")
            try:
                write_heartbeat(HEARTBEAT, created)
            except OSError as e:
                log(f"heartbeat error: {e!r}")
            time.sleep(a.interval)
    finally:
        for f in (HEARTBEAT, WATCHDOG_PID):
            try:
                if f == WATCHDOG_PID and f.read_text().strip() != str(os.getpid()):
                    continue
                if f == HEARTBEAT and json.loads(f.read_text()).get("pid") != os.getpid():
                    continue
                f.unlink()
            except (OSError, ValueError):
                pass


if __name__ == "__main__":
    main()
