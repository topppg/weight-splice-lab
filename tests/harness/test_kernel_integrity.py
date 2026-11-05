import shutil
import time
from collections.abc import Iterator

import pytest

from weight_splice_lab.harness.kernel import Kernel, KernelLimits
from weight_splice_lab.harness.rundir import RunPaths

pytestmark = pytest.mark.skipif(shutil.which("sandbox-exec") is None, reason="needs macOS sandbox")


@pytest.fixture
def kernel(run_paths: RunPaths) -> Iterator[Kernel]:
    k = Kernel(run_paths.sandbox, run_paths.private, KernelLimits(timeout_s=60, threads=1))
    k.start()
    yield k
    k.close()


def checks(k: Kernel) -> set[str]:
    return {v["check"] for v in k.last_integrity}


def test_a_clean_cell_reports_no_integrity_violations(kernel: Kernel) -> None:
    kernel.run("w = load('blocks.0.mlp.down_proj.weight'); w[0, 0] += 0.1")
    assert kernel.last_integrity == []


def test_cpp_set_grad_enable_is_blocked_and_grad_stays_off(kernel: Kernel) -> None:
    out, ok, _ = kernel.run("torch._C._set_grad_enabled(True)")
    assert not ok and "HarnessRuleError" in out
    assert kernel.run("torch.is_grad_enabled()")[0].strip() == "False"
    assert checks(kernel) == set()


def test_setting_requires_grad_by_attribute_is_detected(kernel: Kernel) -> None:
    # the attribute assignment itself is not guarded, so a live grad tensor must be reported
    _, ok, _ = kernel.run("g = torch.ones(3); g.requires_grad = True")
    assert ok
    assert "live_grad_tensor" in checks(kernel)


def test_the_autograd_engine_cannot_build_a_graph_with_grad_off(kernel: Kernel) -> None:
    code = (
        "w = torch.zeros(3)\n"
        "try:\n"
        "    w.requires_grad = True\n"
        "except Exception:\n"
        "    pass\n"
        "y = (w - 1) ** 2\n"
        "print('grad_fn', y.grad_fn)"
    )
    out, ok, _ = kernel.run(code)
    assert ok and "grad_fn None" in out


def test_reimporting_the_model_module_is_detected(kernel: Kernel) -> None:
    kernel.run("import sys; del sys.modules['model']; import model as m2; m2.ToyLM")
    assert "model_module" in checks(kernel)


def test_restoring_a_patched_forward_is_detected(kernel: Kernel) -> None:
    kernel.run("import model as m; _ = m")
    kernel.run("toy_model.ToyLM.forward = lambda *a, **k: 0")
    assert "patched_forward" in checks(kernel)


def test_restoring_an_optim_module_is_detected(kernel: Kernel) -> None:
    kernel.run("import sys; del sys.modules['torch.optim']; import torch.optim")
    assert "optim" in checks(kernel)


def test_replacing_sdpa_is_detected(kernel: Kernel) -> None:
    kernel.run("torch.nn.functional.scaled_dot_product_attention = lambda *a, **k: 0")
    assert "patched_function" in checks(kernel)


def test_a_stray_background_print_does_not_corrupt_replies(kernel: Kernel) -> None:
    code = (
        "import threading, time\n"
        "threading.Thread(target=lambda: (time.sleep(0.6), print('progress 50%')), "
        "daemon=True).start()\n"
        "'started'"
    )
    out, ok, _ = kernel.run(code)
    assert ok and "started" in out
    time.sleep(1.0)
    assert kernel.run("1 + 1")[0].strip() == "2"


def test_a_cell_over_the_memory_limit_is_killed(run_paths: RunPaths) -> None:
    k = Kernel(run_paths.sandbox, run_paths.private, KernelLimits(timeout_s=30, max_rss_gb=0.5))
    k.start()
    try:
        _, ok, restart = k.run("big = torch.ones(int(2e8))")  # ~1.6 GB > 0.5 GB cap
        assert not ok and restart is not None and "memory" in restart
        assert k.run("1 + 1")[0].strip() == "2"
    finally:
        k.close()


def test_reloading_the_model_module_is_detected(kernel: Kernel) -> None:
    # a reload rebinds fresh, unguarded classes inside the same module object
    kernel.run("import importlib\nimportlib.reload(toy_model)")
    assert "model_module" in checks(kernel)


def test_a_large_module_in_the_namespace_is_detected(kernel: Kernel) -> None:
    kernel.run("big = torch.nn.Linear(400, 400)")
    assert "model_object" in checks(kernel)
    kernel.run("del big")
    assert "model_object" not in checks(kernel)
