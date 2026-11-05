import shutil
from pathlib import Path

import pytest
import torch
from conftest import make_run

from weight_splice_lab.gemma.model import GemmaConfig, GemmaLM
from weight_splice_lab.harness.audit import scan_code
from weight_splice_lab.harness.driver import Driver
from weight_splice_lab.harness.edit import Edit, added_params, apply_edit, target_shapes, with_arch
from weight_splice_lab.harness.fake import ScriptedEditor
from weight_splice_lab.harness.rundir import RunConfig


def tiny_gemma() -> GemmaLM:
    torch.manual_seed(0)
    cfg = GemmaConfig(
        vocab_size=50, d_model=16, n_layers=2, n_heads=4, n_kv_heads=1, head_dim=8, d_mlp=24,
        layer_types=["sliding", "full"], sliding_window=4,
    )  # fmt: skip
    model = GemmaLM(cfg).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.3)
    return model


def test_gemma_shapes_follow_the_grown_model() -> None:
    base = tiny_gemma()
    arch = {"n_heads": [6, 4], "d_mlp": [24, 27]}
    grown = with_arch(base, arch)
    assert target_shapes(base, arch) == {k: tuple(v.shape) for k, v in grown.state_dict().items()}
    # each added head: q rows, its own k/v head and o columns
    assert target_shapes(base, arch)["blocks.0.attn.k_proj.weight"] == (3 * 8, 16)
    assert added_params(base, arch) == 2 * 4 * 16 * 8 + 3 * 3 * 16
    assert grown.embed.weight is base.embed.weight


def test_zero_capacity_keeps_the_gemma_function() -> None:
    base = tiny_gemma()
    x = torch.randint(0, 50, (2, 7))
    edited = apply_edit(base, Edit({"n_heads": [6, 5], "d_mlp": [30, 24]}))
    assert torch.allclose(edited(x), base(x), atol=1e-6)
    with torch.no_grad():
        edited.blocks[0].attn.v_proj.weight[8:] = 1.0  # the added heads' own value heads
        edited.blocks[0].attn.o_proj.weight[:, 32:] = 1.0
    assert not torch.allclose(edited(x), base(x), atol=1e-3)


@pytest.mark.parametrize(
    "code",
    [
        "note = 'store the validation requests here'",
        "print(f'no gradient used: {x}')",
        "# requests and gradients are not used\nx = 1",
        "Path('work/validation_text_requests.json').write_text('[]')",
    ],
)
def test_words_in_strings_and_comments_are_not_findings(code: str) -> None:
    assert scan_code(code) == []


@pytest.mark.parametrize(
    ("code", "rule"),
    [("import requests", "outside_run_dir"), ("lr = 0.1", "optimizer"),
     ("gradient_step(w)", "optimizer"), ("requests.get(u", "outside_run_dir")],
)  # fmt: skip
def test_words_in_code_are_still_findings(code: str, rule: str) -> None:
    assert rule in {r for r, _, _, _ in scan_code(code)}


def block(tool: str, body: str) -> str:
    return f"Because reasons.\n```{tool}\n{body}\n```"


EDIT = block(
    "python",
    "d = load('blocks.0.mlp.down_proj.weight'); d[:, 0] += 0.2\n"
    "set_tensor('blocks.0.mlp.down_proj.weight', d)",
)


@pytest.mark.skipif(shutil.which("sandbox-exec") is None, reason="needs macOS sandbox")
def test_the_editor_can_end_the_session_with_a_final_edit(tmp_path: Path, tiny_ckpt: Path) -> None:
    cfg = RunConfig(run_id="t", forward_budget=5, kernel_timeout_s=60, max_turns=6, allow_end=True)
    paths = make_run(tmp_path, tiny_ckpt, cfg)
    replies = [
        EDIT,
        block("save_edit", '{"final": false, "end": true, "note": "x"}'),
        block("save_edit", '{"final": true, "end": true, "note": "done"}'),
        block("python", "1"),
    ]
    st = Driver(cfg, paths, ScriptedEditor(replies)).run(None)
    assert st.stop_reason is not None and "ended by the editor with 5 forward passes left" in (
        st.stop_reason
    )
    assert st.final_edit == 1 and st.edits_saved == 1 and st.python_runs == 1


def test_exec_of_read_source_in_one_cell_is_a_violation() -> None:
    code = "src = Path('model/model.py').read_text()\ng2 = {}\nexec(src, g2)"
    assert ("model_execution", "violation") in {(r, s) for r, s, _, _ in scan_code(code)}


@pytest.mark.skipif(shutil.which("sandbox-exec") is None, reason="needs macOS sandbox")
def test_end_is_refused_unless_allowed(tmp_path: Path, tiny_ckpt: Path) -> None:
    cfg = RunConfig(run_id="t", forward_budget=5, kernel_timeout_s=60, max_turns=3)
    paths = make_run(tmp_path, tiny_ckpt, cfg)
    replies = [EDIT, block("save_edit", '{"final": true, "end": true, "note": "x"}')]
    st = Driver(cfg, paths, ScriptedEditor(replies)).run(None)
    assert not st.ended_by_editor and st.edits_saved == 0 and st.stop_reason.startswith("turn_cap")
