import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
from conftest import make_run

from weight_splice_lab.harness.audit import scan_code
from weight_splice_lab.harness.kernel import Kernel, KernelLimits
from weight_splice_lab.harness.rundir import RunConfig, render_prompt

needs_sandbox = pytest.mark.skipif(
    shutil.which("sandbox-exec") is None, reason="needs macOS sandbox"
)


@pytest.fixture
def kf_kernel(tiny_ckpt: Path, tmp_path: Path) -> Iterator[Kernel]:
    paths = make_run(tmp_path, tiny_ckpt, RunConfig(run_id="kf", kernel_forward=True))
    k = Kernel(paths.sandbox, paths.private, KernelLimits(timeout_s=60, threads=1))
    k.start()
    yield k
    k.close()


@needs_sandbox
def test_the_staged_edit_runs_in_the_kernel_without_gradients(kf_kernel: Kernel) -> None:
    out, ok, _ = kf_kernel.run(
        "import torch\n"
        "ids = torch.tensor([tok.encode('1+2=')])\n"
        "base = build_model(False); same = build_model()\n"
        "d0 = (same(ids) - base(ids)).abs().max().item()\n"
        "new = add_neurons(1, 4); g = load('blocks.1.mlp.up_proj.weight'); g[new] = 1.0\n"
        "set_tensor('blocks.1.mlp.up_proj.weight', g)\n"
        "set_tensor('blocks.1.mlp.gate_proj.weight', g)\n"
        "dn = load('blocks.1.mlp.down_proj.weight'); dn[:, new] = 0.5\n"
        "set_tensor('blocks.1.mlp.down_proj.weight', dn)\n"
        "m = build_model(); d1 = (m(ids) - base(ids)).abs().max().item()\n"
        "print(d0 == 0.0, d1 > 0, m(ids).requires_grad, base.embed.weight is m.embed.weight)"
    )
    assert ok and out.strip() == "True True False True"
    assert kf_kernel.last_integrity == []
    out, ok, _ = kf_kernel.run("m(ids).sum().backward()")
    assert not ok and "not allowed" in out


def test_model_calls_are_flagged_only_without_kernel_forward() -> None:
    code = "m = build_model()\nh = m.hidden(ids)\ny = m.generate(ids, 4)"
    assert {r for r, _, _, _ in scan_code(code)} == {"model_execution"}
    assert scan_code(code, kernel_forward=True) == []


@pytest.mark.parametrize(
    "code",
    [
        "for s in (1, 2, 4):\n    set_tensor('w', base * s)\n    acc[s] = score(build_model())",
        "while k < 3:\n    m.blocks[0].mlp.down_proj.weight[:, j] = v * k\n    out = m.generate(x, 3)",
    ],
)
def test_a_loop_that_writes_weights_and_runs_the_model_is_flagged(code: str) -> None:
    assert ("search_loop", "flag") in {(r, s) for r, s, _, _ in scan_code(code, True)}


@pytest.mark.parametrize(
    "code",
    [
        # the review's misses: plain m(ids) calls, an alias, an in-place op, a helper, a comprehension
        "for s in vals:\n    m.blocks[0].mlp.down_proj.weight[:, j] = v * s\n    acc[s] = score(m(ids))",
        "w = load('a')\nfor s in vals:\n    w[:, j] = v * s\n    set_tensor('a', w)\n    r = m(ids)",
        "for s in vals:\n    p.mul_(s)\n    out = m(ids)",
        (
            "def trial(s):\n    set_tensor('a', base * s)\n    return score(build_model())\n"
            "for s in vals:\n    acc[s] = trial(s)"
        ),
        "accs = [trial(s) for s in vals]\ndef trial(s):\n    set_tensor('a', s)\n    return m(ids)",
    ],
)
def test_sweeps_the_first_heuristic_missed_are_flagged(code: str) -> None:
    assert ("search_loop", "flag") in {(r, s) for r, s, _, _ in scan_code(code, True)}


@pytest.mark.parametrize(
    "code",
    [
        "for t in texts:\n    outs.append(m.hidden(t))",
        "for t in texts:\n    res[t] = m(t).argmax(-1)",
        "w = load('a')\nfor j in cols:\n    w[:, j] = v[j]\nset_tensor('a', w)",
    ],
)
def test_loops_that_only_run_or_only_write_are_not_flagged(code: str) -> None:
    assert scan_code(code, kernel_forward=True) == []


@pytest.mark.parametrize(
    "code", ["import __main__", "with torch.inference_mode(False):\n    y = 1"]
)
def test_reaching_the_server_or_leaving_inference_mode_is_a_violation(code: str) -> None:
    assert "violation" in {s for _, s, _, _ in scan_code(code, True)}


@needs_sandbox
def test_the_server_module_and_inference_mode_false_are_out_of_reach(kf_kernel: Kernel) -> None:
    out, ok, _ = kf_kernel.run("import __main__\nprint(hasattr(__main__, '_check_integrity'))")
    assert ok and out.strip() == "False"
    out, ok, _ = kf_kernel.run("import torch\nwith torch.inference_mode(False):\n    pass")
    assert not ok and "not allowed" in out
    out, ok, _ = kf_kernel.run(
        "import torch\nids = torch.tensor([tok.encode('1+2=')])\n"
        "print(build_model().generate(ids, 2).shape, torch.is_grad_enabled())"
    )
    assert ok and out.strip() == "torch.Size([1, 7]) False"
    assert kf_kernel.last_integrity == []


def test_the_prompt_states_the_mode() -> None:
    mcfg = {"d_model": 640, "head_dim": 256}
    tmpl = Path(__file__).parents[2] / "prompts" / "gemma_reversed.md"
    off = render_prompt(RunConfig(run_id="a", family="gemma_rev"), mcfg, tmpl)
    on = render_prompt(RunConfig(run_id="a", family="gemma_rev", kernel_forward=True), mcfg, tmpl)
    assert "build_model" not in off and "do not implement or simulate" in off
    assert "MAY RUN IN THIS KERNEL" in on and "optimisation by trial" in on
