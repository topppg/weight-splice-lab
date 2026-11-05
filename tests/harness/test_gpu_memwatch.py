import json
import subprocess
import sys
import time
import types
from collections.abc import Iterator
from pathlib import Path

import psutil
import pytest

from weight_splice_lab import gpu, memwatch

SLEEPER = "import time; time.sleep(60)"


@pytest.fixture
def procs() -> Iterator[list[subprocess.Popen[bytes]]]:
    started: list[subprocess.Popen[bytes]] = []
    yield started
    for p in started:
        for c in psutil.Process(p.pid).children(recursive=True) if p.poll() is None else []:
            c.kill()
        p.kill()
        p.wait()


def spawn(procs: list[subprocess.Popen[bytes]], *args: str) -> subprocess.Popen[bytes]:
    p = subprocess.Popen([sys.executable, "-c", *args])
    procs.append(p)
    time.sleep(0.3)
    return p


def test_wired_limit_patch_targets_the_submodule(monkeypatch: pytest.MonkeyPatch) -> None:
    def original() -> int:
        return 1

    def generate() -> None:
        return None

    pkg = types.ModuleType("mlx_lm")
    sub = types.ModuleType("mlx_lm.generate")
    sub.maybe_set_recommended_wired_limit = original  # type: ignore[attr-defined]
    pkg.generate = generate  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_lm", pkg)
    monkeypatch.setitem(sys.modules, "mlx_lm.generate", sub)
    assert gpu.patch_mlx_wired_limit() is sub
    assert sub.maybe_set_recommended_wired_limit is not original  # type: ignore[attr-defined]
    assert sub.maybe_set_recommended_wired_limit() is None  # type: ignore[attr-defined]
    assert not hasattr(pkg.generate, "maybe_set_recommended_wired_limit")  # type: ignore[attr-defined]


def _heartbeat(path: Path, pid: int, created: float, ts: float) -> None:
    path.write_text(json.dumps({"pid": pid, "create_time": created, "ts": ts}))


def test_heartbeat_validates_identity_and_freshness(
    tmp_path: Path, procs: list[subprocess.Popen[bytes]]
) -> None:
    w = spawn(procs, SLEEPER, gpu.WATCHDOG_MODULE)
    created = psutil.Process(w.pid).create_time()
    hb = tmp_path / "hb"
    _heartbeat(hb, w.pid, created, time.time())
    assert gpu.watchdog_healthy(hb)
    _heartbeat(hb, w.pid, created, time.time() - gpu.HEARTBEAT_MAX_AGE_S - 1)
    assert not gpu.watchdog_healthy(hb), "a stalled watchdog must not count"
    _heartbeat(hb, w.pid, created - 100, time.time())
    assert not gpu.watchdog_healthy(hb), "a reused PID must not count"
    me = psutil.Process()
    _heartbeat(hb, me.pid, me.create_time(), time.time())
    assert not gpu.watchdog_healthy(hb), "a process that is not memwatch must not count"
    hb.write_text("{not json")
    assert not gpu.watchdog_healthy(hb)
    hb.write_text('{"pid": 1}')
    assert not gpu.watchdog_healthy(hb)


def test_a_pid_file_alone_is_not_accepted(
    tmp_path: Path, procs: list[subprocess.Popen[bytes]], monkeypatch: pytest.MonkeyPatch
) -> None:
    w = spawn(procs, SLEEPER, gpu.WATCHDOG_MODULE)
    pid_file = tmp_path / "pid"
    pid_file.write_text(str(w.pid))
    monkeypatch.setattr(gpu, "WATCHDOG_PID", pid_file)
    assert not gpu.watchdog_healthy(tmp_path / "missing"), (
        "a stalled legacy watchdog must not count"
    )


def test_a_heartbeat_from_the_future_does_not_count(
    tmp_path: Path, procs: list[subprocess.Popen[bytes]]
) -> None:
    w = spawn(procs, SLEEPER, gpu.WATCHDOG_MODULE)
    hb = tmp_path / "hb"
    _heartbeat(hb, w.pid, psutil.Process(w.pid).create_time(), time.time() + 3600)
    assert not gpu.watchdog_healthy(hb)


@pytest.mark.skipif(not gpu.HEARTBEAT.exists(), reason="no watchdog running")
def test_the_running_watchdog_is_healthy() -> None:
    assert gpu.watchdog_healthy()
    assert gpu.watchdog_running is gpu.watchdog_healthy


def test_write_heartbeat_roundtrip(tmp_path: Path, procs: list[subprocess.Popen[bytes]]) -> None:
    me = psutil.Process()
    hb = tmp_path / ".memwatch.heartbeat"
    memwatch.write_heartbeat(hb, me.create_time())
    data = json.loads(hb.read_text())
    assert data["pid"] == me.pid and abs(data["ts"] - time.time()) < 5
    assert not (tmp_path / ".memwatch.heartbeat.tmp").exists()


def test_pressure_level_needs_no_subprocess() -> None:
    level = memwatch.pressure_level()
    out = subprocess.run(
        ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert level in (0, 1, 2, 4)
    assert level == int(out.stdout.strip())


def test_one_failing_check_does_not_skip_the_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(memwatch, "LOG", tmp_path / "memwatch.log")

    def broken() -> str | None:
        raise OSError("fork failed")

    assert memwatch.poll([("pressure", broken), ("available", lambda: "low")]) == "low"
    assert memwatch.poll([("pressure", broken), ("swap", lambda: None)]) is None
    assert "poll error in pressure" in (tmp_path / "memwatch.log").read_text()


def test_repo_processes_use_identity_and_tree_not_substrings(
    procs: list[subprocess.Popen[bytes]],
) -> None:
    named = spawn(procs, SLEEPER, "--run-id", "memwatch-test")
    parent = spawn(
        procs, "import subprocess, time; subprocess.Popen(['/bin/sleep', '60']); time.sleep(60)"
    )
    kids = [c.pid for c in psutil.Process(parent.pid).children()]
    assert kids, "the helper child did not start"
    found = {p.pid for p in memwatch.repo_processes()}
    assert named.pid in found, "a job whose command contains 'memwatch' must still be selected"
    assert parent.pid in found and set(kids) <= found, "descendants must be selected"
    assert psutil.Process().pid not in found
    found = {p.pid for p in memwatch.repo_processes(frozenset({named.pid}))}
    assert named.pid not in found and parent.pid in found


def test_kill_all_keeps_the_lock_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memwatch, "LOG", tmp_path / "memwatch.log")
    monkeypatch.setattr(memwatch, "repo_processes", list)
    existed = gpu.LOCK.exists()
    memwatch.kill_all("test")
    assert gpu.LOCK.exists() == existed
    assert not hasattr(memwatch, "LOCK")


def test_swap_growth_only_kills_when_memory_is_short() -> None:
    from weight_splice_lab.memwatch import swap_reason

    assert swap_reason(1.67, 6.7, 1.5, 4.0) is None
    assert swap_reason(1.67, 3.2, 1.5, 4.0) is not None
    assert swap_reason(0.5, 3.2, 1.5, 4.0) is None
