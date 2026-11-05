import json
import shutil
from pathlib import Path

import pytest
from conftest import make_run

from weight_splice_lab.harness.driver import Driver
from weight_splice_lab.harness.fake import ScriptedEditor
from weight_splice_lab.harness.rundir import RunConfig

pytestmark = pytest.mark.skipif(shutil.which("sandbox-exec") is None, reason="needs macOS sandbox")


def block(tool: str, body: str) -> str:
    return f"Because reasons.\n```{tool}\n{body}\n```"


def test_loop_runs_tools_and_stops_when_done(tmp_path: Path, tiny_ckpt: Path) -> None:
    cfg = RunConfig(run_id="t", forward_budget=2, handoff_turns=3, kernel_timeout_s=60)
    paths = make_run(tmp_path, tiny_ckpt, cfg)
    gen = {"type": "generate", "model": "both", "ids": ["add_n01_000", "copy_n02_001"]}
    tf = {
        "type": "teacher_forced",
        "model": "both",
        "texts": ["abc", "A:1+2=30"],
        "return": {"topk": 2, "sites": ["resid_pre.1", "attn.0.pattern", "resid_final", "logits"]},
    }
    replies = [
        block(
            "python",
            "d = load('blocks.0.mlp.down_proj.weight'); d[:, 0] += 0.2\n"
            "set_tensor('blocks.0.mlp.down_proj.weight', d)",
        ),
        block("forward", json.dumps(gen)),
        block("save_edit", '{"final": false, "note": "first"}'),
        "no block here",
        block("forward", json.dumps(gen)),
        block("forward", json.dumps(tf)),
        block("forward", json.dumps(gen)),
        block("save_edit", '{"final": true, "note": "done"}'),
    ]
    editor = ScriptedEditor(replies, points_per_turn=0.5)
    st = Driver(cfg, paths, editor).run()
    assert st.stop_reason == "done: forward budget used and final edit submitted"
    assert (st.python_runs, st.passes, st.edits_saved, st.final_edit) == (1, 2, 2, 2)
    assert st.turn == 8 and st.handoffs == 2
    driver_total = sum(
        json.loads(x)["usage"]["input_tokens"]
        for x in (paths.private / "usage.jsonl").read_text().splitlines()
    )
    assert driver_total == sum(editor.per_turn_input)
    assert "Replies left:" in editor.prompts[5]
    assert "python run 1" in (paths.work / "cells.py").read_text()
    prompts = editor.prompts
    assert "no edit has been saved yet" in prompts[2]
    assert "stored as edit 1" in prompts[3]
    assert "was not run" in prompts[4] and "THIS IS A RESTARTED CONVERSATION" in prompts[3]
    assert "Forward passes left: 1." in prompts[5]
    assert "Forward pass 2 of 2 done (teacher_forced)" in prompts[6]
    assert "No forward passes left" in prompts[7]
    summary = json.loads(prompts[5].split("```\n")[1].split("\n```")[0])
    assert set(summary["paired"]) == {"rescues", "losses", "outputs_changed"}
    assert (paths.work / "fwd_2" / "edited" / "attn.0.pattern.npy").exists()
    calls = [json.loads(x) for x in (paths.private / "tool_calls.jsonl").read_text().splitlines()]
    assert [c["tool"] for c in calls] == [
        "python",
        "forward",
        "save_edit",
        "forward",
        "forward",
        "forward",
        "save_edit",
    ]
    usage = (paths.private / "usage.jsonl").read_text().splitlines()
    assert len(usage) == 8 and json.loads(usage[-1])["points_used"] == pytest.approx(3.5)
    manifest = json.loads((paths.edits / "edit_2" / "manifest.json").read_text())
    assert manifest["final"] and list(manifest["validation"]["tensors"]) == [
        "blocks.0.mlp.down_proj.weight"
    ]


def test_rate_limit_guard_stops_the_loop(tmp_path: Path, tiny_ckpt: Path) -> None:
    cfg = RunConfig(run_id="t", max_limit_points=2.0, kernel_timeout_s=60)
    paths = make_run(tmp_path, tiny_ckpt, cfg)
    editor = ScriptedEditor([block("python", "1")], points_per_turn=1.0)
    st = Driver(cfg, paths, editor).run()
    assert st.stop_reason is not None and st.stop_reason.startswith("rate_limit")
    assert st.turn == 3


def test_throttle_detection_and_backoff() -> None:
    from weight_splice_lab.harness.codex import classify_errors
    from weight_splice_lab.harness.driver import throttle_wait_s

    def is_throttled(errors: list[str]) -> bool:
        return classify_errors(errors).transient

    assert is_throttled(['{"type": "error", "message": "last status: 429 Too Many Requests"}'])
    assert is_throttled(['{"type": "error", "message": "Reconnecting... 2/5 (request timed out)"}'])
    assert is_throttled(
        ["ERROR codex_models_manager: failed to refresh available models: request timed out"]
    )
    assert not is_throttled(['{"type": "error", "message": "invalid model gpt-x"}'])
    waits = [throttle_wait_s(n) for n in range(1, 9)]
    assert waits[:7] == [120.0, 240.0, 480.0, 960.0, 1920.0, 3600.0, 3600.0]
    assert waits[7] is None and throttle_wait_s(0) is None
