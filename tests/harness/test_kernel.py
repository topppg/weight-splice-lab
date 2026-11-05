import os
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import psutil
import pytest

from weight_splice_lab.harness.edit import read_staged, validate
from weight_splice_lab.harness.kernel import Kernel, KernelLimits
from weight_splice_lab.harness.rundir import ROOT, RunPaths
from weight_splice_lab.toy.model import load_model

pytestmark = pytest.mark.skipif(shutil.which("sandbox-exec") is None, reason="needs macOS sandbox")


@pytest.fixture(scope="module")
def kernel(run_paths: RunPaths) -> Iterator[Kernel]:
    k = Kernel(run_paths.sandbox, run_paths.private, KernelLimits(timeout_s=60, threads=1))
    k.start()
    yield k
    k.close()


def run(kernel: Kernel, code: str) -> str:
    out, _, restart = kernel.run(code)
    assert restart is None
    return out


def test_state_persists_and_last_expression_is_shown(kernel: Kernel) -> None:
    run(kernel, "x = 41")
    assert run(kernel, "x + 1").strip() == "42"
    assert run(kernel, "print('a'); y = 2").strip() == "a"


def test_errors_are_reported_and_kernel_survives(kernel: Kernel) -> None:
    out, ok, _ = kernel.run("1 / 0")
    assert not ok and "ZeroDivisionError" in out
    assert run(kernel, "raise SystemExit").startswith("SystemExit was caught")
    assert run(kernel, "x").strip() == "41"


@pytest.mark.parametrize(
    "code",
    [
        "torch.ones(2).sum().backward()",
        "torch.ones(2).requires_grad_()",
        "torch.autograd.grad(torch.ones(1), torch.ones(1))",
        "torch.set_grad_enabled(True)",
        "with torch.enable_grad(): pass",
        "torch.optim.SGD",
        "from torch.optim import Adam",
        "import torch.optim as o; o.AdamW",
        "torch.func.grad(lambda t: t.sum())",
    ],
)
def test_gradients_and_optimizers_are_blocked(kernel: Kernel, code: str) -> None:
    out, ok, _ = kernel.run(code)
    assert not ok and "HarnessRuleError" in out


def test_grad_mode_is_off_and_no_grad_still_works(kernel: Kernel) -> None:
    assert run(kernel, "torch.is_grad_enabled()").strip() == "False"
    assert (
        run(kernel, "with torch.no_grad(): z = torch.ones(2) * 2\nz.sum().item()").strip() == "4.0"
    )


@pytest.mark.parametrize(
    "code",
    [
        "m = toy_model.ToyLM(toy_model.ToyConfig(vocab_size=99)); m(torch.ones(1, 3, dtype=torch.long))",
        "toy_model.SwiGLU(8, 16)(torch.ones(1, 8))",
        "torch.nn.functional.scaled_dot_product_attention(*(torch.ones(1, 1, 2, 4),) * 3)",
    ],
)
def test_running_the_model_is_blocked(kernel: Kernel, code: str) -> None:
    out, ok, _ = kernel.run(code)
    assert not ok and "HarnessRuleError" in out


def test_reads_are_confined_to_the_run_folder(kernel: Kernel) -> None:
    assert "TASK" in run(kernel, "open('TASK.md').read(4)")
    home = Path.home()
    for path in [ROOT / "README.md", ROOT / "docs", ROOT.parent, home / ".codex-switch", home]:
        code = f"import os\ntry:\n    os.listdir(os.path.expanduser({str(path)!r})) if not str({str(path)!r}).endswith('.md') else open({str(path)!r}).read()\n    print('READ')\nexcept OSError as e:\n    print(type(e).__name__)"
        assert run(kernel, code).strip() == "PermissionError", path


def test_writes_are_confined_to_work(kernel: Kernel) -> None:
    assert run(kernel, "open('work/ok.txt', 'w').write('x')").strip() == "1"
    for path in ["model/x.txt", "TASK2.md", str(ROOT / "x.txt")]:
        code = f"try:\n    open({path!r}, 'w').write('x')\n    print('WROTE')\nexcept OSError as e:\n    print(type(e).__name__)"
        assert run(kernel, code).strip() == "PermissionError", path


def test_network_and_subprocess_reads_are_denied(kernel: Kernel) -> None:
    code = "import socket\ntry:\n    socket.create_connection(('1.1.1.1', 80), timeout=3)\n    print('OPEN')\nexcept OSError as e:\n    print('denied')"
    assert run(kernel, code).strip() == "denied"
    code = f"import subprocess\ntry:\n    subprocess.run(['/bin/ls', {str(ROOT)!r}], capture_output=True)\n    print('RAN')\nexcept OSError as e:\n    print(type(e).__name__)"
    assert run(kernel, code).strip() == "PermissionError"


def test_repo_package_is_not_importable(kernel: Kernel) -> None:
    out, ok, _ = kernel.run("import weight_splice_lab")
    assert not ok and "ModuleNotFoundError" in out


def test_staging_helpers_match_driver_validation(kernel: Kernel, run_paths: RunPaths) -> None:
    run(kernel, "reset_edit()")
    cell = """
new = add_neurons(1, 3)
w = load('blocks.1.mlp.up_proj.weight')
w[new] = 0.1
set_tensor('blocks.1.mlp.up_proj.weight', w)
d = load('blocks.0.mlp.down_proj.weight'); d[0, 0] += 1.0
set_tensor('blocks.0.mlp.down_proj.weight', d)
print(new, staged(), added_params(), shape('blocks.1.mlp.down_proj.weight'))
"""
    out = run(kernel, cell)
    assert out.strip() == (
        "[96, 97, 98] ['blocks.0.mlp.down_proj.weight', 'blocks.1.mlp.up_proj.weight'] 576 (64, 99)"
    )
    base = load_model(run_paths.sandbox / "model" / "base.pt")
    report = validate(read_staged(run_paths.stage, base), base, "add_capacity", 1_500_000)
    assert report["added_neurons"] == [0, 3]
    out, ok, _ = kernel.run("set_tensor('embed.weight', load('embed.weight'))")
    assert not ok and "may not be edited" in out
    run(kernel, "reset_edit()")
    assert run(kernel, "staged(), arch()['d_mlp']").strip() == "([], [96, 96])"


def test_timeout_restarts_the_kernel(run_paths: RunPaths) -> None:
    k = Kernel(run_paths.sandbox, run_paths.private, KernelLimits(timeout_s=2, threads=1))
    k.start()
    try:
        k.run("x = 5")
        out, ok, restart = k.run("import time; time.sleep(30)")
        assert not ok and restart is not None and "time limit" in restart
        out, ok, _ = k.run("x")
        assert not ok and "NameError" in out
    finally:
        k.close()


def test_creating_symlinks_and_hard_links_in_work_is_denied(kernel: Kernel) -> None:
    run(kernel, "open('work/real.txt', 'w').write('x')")
    link = (
        "import os\ntry:\n    os.symlink({t!r}, 'work/link')\n    print('MADE')\n"
        "except OSError as e:\n    print(type(e).__name__)"
    )
    assert run(kernel, link.format(t=str(ROOT / "README.md"))).strip() == "PermissionError"
    hard = (
        "import os\ntry:\n    os.link('work/real.txt', 'work/hard')\n    print('MADE')\n"
        "except OSError as e:\n    print(type(e).__name__)"
    )
    assert run(kernel, hard).strip() == "PermissionError"


def test_launching_other_executables_is_denied(kernel: Kernel) -> None:
    for argv in (
        "['/usr/bin/open', 'work']",
        "['/usr/bin/osascript', '-e', 'beep']",
        "['/bin/sh', '-c', 'echo hi']",
    ):
        code = (
            f"import subprocess\ntry:\n    subprocess.run({argv}, capture_output=True)\n    print('RAN')\n"
            "except OSError as e:\n    print(type(e).__name__)"
        )
        assert run(kernel, code).strip() == "PermissionError", argv


def test_mps_is_unavailable_and_a_tiny_mps_allocation_fails(kernel: Kernel) -> None:
    assert run(kernel, "torch.backends.mps.is_available()").strip() == "False"
    out, ok, _ = kernel.run("torch.ones(1, device='mps')")
    assert not ok and "MPS" in out


def test_a_tiny_mlx_op_fails(kernel: Kernel) -> None:
    out, ok, _ = kernel.run("import mlx.core as mx; mx.eval(mx.ones((1,)) + 1)")
    assert not ok and ("Metal" in out or "metal" in out)


BROKERS = (
    "com.apple.coreservices.launchservicesd",
    "com.apple.lsd.open",
    "com.apple.coreservices.appleevents",
)
LOOKUP = (
    "import ctypes\n"
    "libc = ctypes.CDLL('/usr/lib/libSystem.B.dylib')\n"
    "bp = ctypes.c_uint.in_dll(libc, 'bootstrap_port').value\n"
    "def lookup(name):\n"
    "    port = ctypes.c_uint(0)\n"
    "    return libc.bootstrap_look_up(bp, name.encode(), ctypes.byref(port))\n"
)


def test_broker_mach_services_cannot_be_looked_up(kernel: Kernel) -> None:
    # A lookup only obtains a send right; nothing is launched or messaged.
    ns: dict[str, object] = {}
    exec(LOOKUP, ns)  # noqa: S102 - trusted test source
    assert ns["lookup"](BROKERS[0]) == 0, "outside the sandbox the lookup must succeed"  # type: ignore[operator]
    out = run(kernel, LOOKUP + f"print([lookup(n) for n in {BROKERS!r}])")
    assert out.strip() == "[1100, 1100, 1100]"  # BOOTSTRAP_NOT_PRIVILEGED: denied by the profile


def test_cleanup_never_signals_a_reaped_kernel_group(
    run_paths: RunPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    k = Kernel(run_paths.sandbox, run_paths.private, KernelLimits())
    proc = subprocess.Popen(["/bin/sleep", "30"], start_new_session=True, text=True)
    k.proc, k._pgid = proc, proc.pid
    k._ident = (proc.pid, psutil.Process(proc.pid).create_time())
    other = subprocess.Popen(["/bin/sleep", "30"])
    k._seen = {(other.pid, psutil.Process(other.pid).create_time() - 100)}  # a "reused" PID
    proc.kill()
    proc.wait()  # reaped: its PID / group ID may now belong to anyone
    signalled: list[int] = []
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: signalled.append(pgid))
    try:
        k.kill()
        assert signalled == []
        assert other.poll() is None, "a process whose creation time differs must not be killed"
    finally:
        other.kill()
        other.wait()


def test_cleanup_kills_the_spawn_group_while_the_kernel_is_unreaped(run_paths: RunPaths) -> None:
    k = Kernel(run_paths.sandbox, run_paths.private, KernelLimits())
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import subprocess, time; subprocess.Popen(['/bin/sleep', '30']); time.sleep(30)",
        ],
        start_new_session=True,
        text=True,
    )
    time.sleep(0.5)
    kid = psutil.Process(proc.pid).children()[0]
    k.proc, k._pgid = proc, proc.pid
    k._ident = (proc.pid, psutil.Process(proc.pid).create_time())
    k.kill()
    assert proc.returncode is not None
    kid.wait(timeout=5)
    assert not kid.is_running()
