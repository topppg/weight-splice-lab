import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import psutil

KERNEL_SOURCE = (Path(__file__).parent / "kernel_server.py").read_text()


@dataclass
class KernelLimits:
    timeout_s: float = 900.0
    max_rss_gb: float = 2.0
    threads: int = 4
    startup_timeout_s: float = 180.0


def sandbox_profile(sandbox: Path, tmp: Path) -> str:
    home = Path.home().resolve()
    venv = Path(sys.prefix).resolve()
    pyhome = Path(sys.base_prefix).resolve()
    site = next((venv / "lib").glob("python3*/site-packages")).resolve()
    sb, tp = sandbox.resolve(), tmp.resolve()
    # The venv python is a symlink into the uv interpreter install; exec resolves through it.
    interp = (venv / "bin" / "python").resolve()
    # The editable install's metadata embeds the repo README; its .pth points into the repo.
    leaks = [*site.glob("weight_splice_lab*.dist-info"), *site.glob("_editable_impl_*.pth")]
    deny_leaks = " ".join(f'(subpath "{p}")' for p in leaks)
    return f"""(version 1)
(allow default)
(deny network*)
(deny file-read* (subpath "{home}") (subpath "/private/tmp") (subpath "/private/var/folders")
  (subpath "/Volumes"))
(allow file-read* (subpath "{sb}") (subpath "{tp}") (subpath "{venv}")
  (subpath "{pyhome}"))
{f"(deny file-read* {deny_leaks})" if leaks else ""}
(deny file-write*)
(allow file-write* (subpath "{sb / "work"}") (subpath "{tp}") (literal "/dev/null"))
(allow file-read* file-write* (regex #"^/private/tmp/__KMP_REGISTERED_LIB"))
; No new symlinks or hard links: the unsandboxed driver reads and writes under work/ by path, so a
; link there would redirect it outside the sandbox.
(deny file-write-create (vnode-type SYMLINK))
(deny file-link)
; Only the kernel interpreter may run; this blocks subprocess, os.system, open(1), osascript and any
; detached helper that would escape the process group and the memory accounting.
(deny process-exec*)
(allow process-exec* (literal "{venv / "bin" / "python"}") (literal "{interp}"))
(deny process-fork)
; No GPU: MPS and MLX need the Metal device user client and compiler, so both fail to allocate.
(deny iokit-open
  (iokit-user-client-class "AGXDeviceUserClient")
  (iokit-user-client-class "IOAccelDevice")
  (iokit-user-client-class "IOAccelContext")
  (iokit-user-client-class "IOAccelSharedUserClient")
  (iokit-user-client-class "IOAccelCommandQueue")
  (iokit-user-client-class "IOSurfaceRootUserClient")
  (iokit-user-client-class "IOSurfaceSendRight"))
(deny mach-lookup
  (global-name "com.apple.MTLCompilerService")
  (global-name "com.apple.gpumemd.source"))
; No brokers that act outside the sandbox on the kernel's behalf: LaunchServices (open apps/URLs) and
; Apple events (script other apps), even when called directly through native APIs.
(deny lsopen)
(deny appleevent-send)
(deny mach-lookup
  (global-name "com.apple.coreservices.launchservicesd")
  (global-name "com.apple.CoreServices.coreservicesd")
  (global-name "com.apple.lsd.open")
  (global-name "com.apple.lsd.mapdb")
  (global-name "com.apple.lsd.modifydb")
  (global-name "com.apple.coreservices.appleevents")
  (global-name "com.apple.coreservices.quarantine-resolver"))
"""


Ident = tuple[int, float]  # (pid, create_time): a PID alone may be reused after the process dies


def _create_time(pid: int) -> float | None:
    try:
        return psutil.Process(pid).create_time()
    except (psutil.Error, SystemError, OSError):
        return None


def _kill_if_same(ident: Ident) -> None:
    pid, created = ident
    try:
        p = psutil.Process(pid)
        if p.create_time() == created:
            p.send_signal(signal.SIGKILL)
    except (psutil.Error, SystemError, OSError):
        pass


class Kernel:
    def __init__(self, sandbox: Path, private: Path, limits: KernelLimits) -> None:
        self.sandbox = sandbox
        self.tmp = private / "kernel_tmp"
        self.profile = private / "kernel.sb"
        self.limits = limits
        self.proc: subprocess.Popen[str] | None = None
        self.lines: queue.Queue[str | None] = queue.Queue()
        self.restarts = 0
        self.peak_rss_gb = 0.0
        self.last_integrity: list[dict[str, str]] = []
        self._nonce = 0
        self._pgid: int | None = None
        self._ident: Ident | None = None
        self._seen: set[Ident] = set()
        self._lock = threading.Lock()
        self._breach: str | None = None
        self._monitor: threading.Thread | None = None
        self._monitor_stop = threading.Event()

    def start(self) -> str:
        self._stop_monitor()
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.profile.write_text(sandbox_profile(self.sandbox, self.tmp))
        python = str(Path(sys.prefix) / "bin" / "python")
        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(self.tmp),
            "TMPDIR": str(self.tmp),
            "KERNEL_THREADS": str(self.limits.threads),
            "OMP_NUM_THREADS": str(self.limits.threads),
            "PYTHONHASHSEED": "0",
            "LANG": "en_US.UTF-8",
        }
        self._breach = None
        proc = subprocess.Popen(
            ["sandbox-exec", "-f", str(self.profile), python, "-I", "-u", "-c", KERNEL_SOURCE],
            cwd=self.sandbox,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        )
        # start_new_session makes the kernel its own group leader: the group ID is fixed at spawn and
        # never re-read from a PID that may since have been reused.
        created = _create_time(proc.pid)
        with self._lock:
            self.proc = proc
            self._pgid = proc.pid
            self._ident = (proc.pid, created) if created is not None else None
            self._seen = set()
        self.lines = queue.Queue()
        threading.Thread(target=self._reader, args=(proc, self.lines), daemon=True).start()
        self._start_monitor(proc)
        reply = self._wait(self.limits.startup_timeout_s, None)
        if reply is None or not reply.get("ready"):
            out = reply.get("output", "") if reply else ""
            self.kill()
            raise RuntimeError(f"kernel failed to start: {out[-3000:]}")
        return str(reply.get("output", ""))

    @staticmethod
    def _reader(proc: subprocess.Popen[str], lines: "queue.Queue[str | None]") -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    def _start_monitor(self, proc: subprocess.Popen[str]) -> None:
        stop = threading.Event()
        self._monitor_stop = stop
        self._monitor = threading.Thread(target=self._monitor_rss, args=(proc, stop), daemon=True)
        self._monitor.start()

    def _stop_monitor(self) -> None:
        # The old monitor must be gone before the shared state it reads is replaced for a new kernel.
        self._monitor_stop.set()
        mon = self._monitor
        if mon is not None and mon is not threading.current_thread():
            mon.join(timeout=5)
        self._monitor = None

    def _monitor_rss(self, proc: subprocess.Popen[str], stop: threading.Event) -> None:
        # Runs for the kernel's whole life, including between cells (during the Codex turn), so a cell
        # that allocates and returns before the next poll cannot keep the memory past the limit.
        while not stop.wait(0.25):
            if proc.returncode is not None or not psutil.pid_exists(proc.pid):
                return
            rss = self._rss_gb(proc)
            self.peak_rss_gb = max(self.peak_rss_gb, rss)
            if rss > self.limits.max_rss_gb:
                self._breach = f"{rss:.2f} GB > {self.limits.max_rss_gb} GB"
                self._kill_proc(proc)
                return

    def _rss_gb(self, proc: subprocess.Popen[str] | None = None) -> float:
        proc = proc or self.proc
        if proc is None or proc.returncode is not None:
            return 0.0
        try:
            root = psutil.Process(proc.pid)
            if self._ident is None or root.create_time() != self._ident[1]:
                return 0.0
            tree = [root, *root.children(recursive=True)]
        except psutil.Error:
            return 0.0
        total = 0.0
        kids: list[Ident] = []
        for q in tree:
            try:
                total += q.memory_info().rss
                if q.pid != proc.pid:
                    kids.append((q.pid, q.create_time()))
            except psutil.Error:
                continue  # a child that vanished mid-walk is not counted, not fatal
        if kids:
            with self._lock:
                self._seen.update(kids)
        return total / 1e9

    def _wait(self, timeout: float, nonce: int | None) -> dict[str, object] | None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                line = self.lines.get(timeout=0.25)
            except queue.Empty:
                if self._breach is not None:
                    raise MemoryError(self._breach)
                continue
            if line is None:
                if self._breach is not None:
                    raise MemoryError(self._breach)
                raise ChildProcessError("the kernel process exited")
            try:
                reply = json.loads(line)
            except json.JSONDecodeError:
                continue  # a stray print on the protocol stream: ignore, keep waiting
            if not isinstance(reply, dict) or reply.get("nonce") != nonce:
                continue  # a stale or mistagged reply
            rss = self._rss_gb()
            self.peak_rss_gb = max(self.peak_rss_gb, rss)
            if rss > self.limits.max_rss_gb:
                raise MemoryError(f"{rss:.2f} GB > {self.limits.max_rss_gb} GB")
            return reply
        raise TimeoutError(f"no result after {timeout:.0f} s")

    def run(self, code: str) -> tuple[str, bool, str | None]:
        self.last_integrity = []
        if self.proc is None or self.proc.poll() is not None:
            self.kill()
            self.start()
        assert self.proc is not None and self.proc.stdin is not None
        self._nonce += 1
        nonce = self._nonce
        try:
            self.proc.stdin.write(json.dumps({"code": code, "nonce": nonce}) + "\n")
            self.proc.stdin.flush()
            reply = self._wait(self.limits.timeout_s, nonce)
        except (TimeoutError, MemoryError, ChildProcessError, BrokenPipeError, OSError) as e:
            reason = {
                TimeoutError: f"time limit of {self.limits.timeout_s:.0f} s",
                MemoryError: f"memory limit ({e})",
                ChildProcessError: "the kernel process died",
            }.get(type(e), f"the kernel failed ({type(e).__name__})")
            self.kill()
            self.restarts += 1
            self.start()
            return "", False, reason
        assert reply is not None
        self.last_integrity = list(reply.get("integrity") or [])  # type: ignore[arg-type]
        return str(reply.get("output", "")), bool(reply.get("ok")), None

    def _kill_proc(self, proc: subprocess.Popen[str]) -> None:
        with self._lock:
            # While the kernel is unreaped its PID, and so its group ID, cannot be reused: signal the
            # group retained at spawn. Once reaped, the number may belong to someone else: skip it.
            unreaped = proc is self.proc and proc.returncode is None and self._pgid is not None
            same = self._ident is not None and _create_time(proc.pid) in (self._ident[1], None)
            if unreaped and same and self._pgid is not None:
                try:
                    os.killpg(self._pgid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
            seen = list(self._seen)
        # Descendants seen while monitoring that left the group, each only if it is still the same
        # process (PID and creation time).
        for ident in seen:
            _kill_if_same(ident)

    def kill(self) -> None:
        self._stop_monitor()
        proc = self.proc
        if proc is None:
            return
        self._kill_proc(proc)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        with self._lock:
            self.proc = None
            self._pgid = None
            self._ident = None
            self._seen = set()

    def close(self) -> None:
        self.kill()
