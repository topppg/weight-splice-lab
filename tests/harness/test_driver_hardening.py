import importlib.util
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import make_run

from weight_splice_lab.harness import driver as driver_mod
from weight_splice_lab.harness import evaluate as toy_eval
from weight_splice_lab.harness.budget import RateReading, Usage, fresh_baseline
from weight_splice_lab.harness.codex import TurnResult, run_group
from weight_splice_lab.harness.driver import Driver, RunState, finalize, mirror
from weight_splice_lab.harness.fake import ScriptedEditor
from weight_splice_lab.harness.rundir import IntegrityError, RunConfig, RunPaths, copy_tree_regular

R = RateReading(10.0, 1_900_000_000, 10080, "x")
TF = {"type": "teacher_forced", "model": "original", "texts": ["abc"]}
GEN = {"type": "generate", "model": "original", "ids": ["add_n01_000"]}


class Alive:
    def poll(self) -> int | None:
        return None


class Dead:
    def poll(self) -> int | None:
        return -9


class StubKernel:
    def __init__(self) -> None:
        self.proc: Alive | Dead | None = None
        self.codes: list[str] = []

    def start(self) -> str:
        self.proc = Alive()
        return ""

    def run(self, code: str) -> tuple[str, bool, str | None]:
        if self.proc is None or self.proc.poll() is not None:
            self.start()
        self.codes.append(code)
        return "ran", True, None

    def close(self) -> None:
        self.proc = None


def block(tool: str, body: str) -> str:
    return f"Step.\n```{tool}\n{body}\n```"


def ok(text: str, tid: str, total_in: int) -> TurnResult:
    return TurnResult(
        True, text, tid, Usage(total_in, 0, 10, 0), R, [], {"agent_message": 1}, [], 0.0
    )


def fail(
    errors: list[str], tid: str | None = None, usage: Usage | None = None, **kw: bool
) -> TurnResult:
    return TurnResult(False, "", tid, usage, None, [], {}, errors, 0.0, **kw)  # type: ignore[arg-type]


@pytest.fixture
def setup(tmp_path: Path, tiny_ckpt: Path) -> tuple[RunPaths, Path]:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("CANARY-SECRET")
    return make_run(tmp_path / "runs", tiny_ckpt, RunConfig(run_id="r", forward_budget=5)), outside


def make_driver(paths: RunPaths, editor: object, **cfg: object) -> Driver:
    d = Driver(RunConfig(run_id="r", forward_budget=5, **cfg), paths, editor)  # type: ignore[arg-type]
    d.kernel = StubKernel()  # type: ignore[assignment]
    d.kernel.start()
    return d


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    waits: list[float] = []
    monkeypatch.setattr(driver_mod.time, "sleep", waits.append)
    return waits


def violations(paths: RunPaths) -> list[dict[str, object]]:
    f = paths.private / "integrity.jsonl"
    return [json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []


def test_symlinked_notes_are_not_read_at_handoff(setup: tuple[RunPaths, Path]) -> None:
    paths, outside = setup
    (paths.work / "notes.md").unlink()
    (paths.work / "notes.md").symlink_to(outside / "secret.txt")
    d = make_driver(paths, ScriptedEditor(["x"]))
    d.st.thread_id, d.st.since_handoff = "old", d.cfg.handoff_turns
    msg = d._maybe_handoff("last result")
    assert "CANARY" not in msg and "integrity check" in msg and "RESTARTED" in msg
    assert d.st.integrity_violations == 1 and violations(paths)[0]["where"] == "work/notes.md"


def test_symlinked_cells_file_is_never_appended(setup: tuple[RunPaths, Path]) -> None:
    paths, outside = setup
    (paths.work / "cells.py").symlink_to(outside / "appended.pth")
    d = make_driver(paths, ScriptedEditor(["x"]))
    reply = d._python("x = 1")
    assert not (outside / "appended.pth").exists()
    assert "OUTPUT of Python run 1" in reply and "integrity check" in reply
    assert violations(paths)[0]["where"] == "work/cells.py"


def test_forward_outputs_refuse_symlinks(setup: tuple[RunPaths, Path]) -> None:
    paths, outside = setup
    d = make_driver(paths, ScriptedEditor(["x"]))
    (paths.work / "fwd_1").symlink_to(outside, target_is_directory=True)
    reply = d._dispatch("forward", json.dumps(TF))
    # inference ran before the write was refused, so the pass is used (codex review v3 #4)
    assert "integrity check" in reply and "counted as pass 1" in reply and d.st.passes == 1
    (paths.work / "fwd_2.jsonl").symlink_to(outside / "secret.txt")
    reply = d._dispatch("forward", json.dumps(GEN))
    assert "integrity check" in reply and d.st.passes == 2
    assert sorted(p.name for p in outside.iterdir()) == ["secret.txt"]
    assert (outside / "secret.txt").read_text() == "CANARY-SECRET"
    assert len(violations(paths)) == 2


def test_forward_failure_on_a_file_where_its_dir_goes_is_counted(
    setup: tuple[RunPaths, Path],
) -> None:
    # harness audit K3: NotADirectoryError escaped and ended the run; v3 #4: it ran, so it counts
    paths, _ = setup
    d = make_driver(paths, ScriptedEditor(["x"]))
    (paths.work / "fwd_1").write_text("a file")
    reply = d._dispatch("forward", json.dumps(TF))
    assert "counted as pass 1" in reply and "NotADirectoryError" in reply
    assert d.st.passes == 1


def test_bad_forward_request_is_not_counted(setup: tuple[RunPaths, Path]) -> None:
    paths, _ = setup
    d = make_driver(paths, ScriptedEditor(["x"]))
    reply = d._dispatch("forward", json.dumps(TF | {"model": "edited"}))
    assert "failed and was not counted" in reply and d.st.passes == 0


def test_partial_forward_output_consumes_the_pass(setup: tuple[RunPaths, Path]) -> None:
    # codex review v3 #4: fwd_1_original.jsonl was delivered, fwd_1_edited.jsonl failed, passes 0
    paths, _ = setup
    d = make_driver(paths, ScriptedEditor(["x"]))
    d.fwd.edited = d.base
    (paths.work / "fwd_1_edited.jsonl").mkdir()
    reply = d._dispatch("forward", json.dumps(GEN | {"model": "both"}))
    assert (paths.work / "fwd_1_original.jsonl").is_file()
    assert "counted as pass 1" in reply and d.st.passes == 1
    row = json.loads((paths.private / "tool_calls.jsonl").read_text().splitlines()[-1])
    assert row["counted"] is True and row["index"] == 1 and not row["ok"]


def test_symlinked_staged_tensor_is_refused(setup: tuple[RunPaths, Path]) -> None:
    paths, outside = setup
    d = make_driver(paths, ScriptedEditor(["x"]))
    (paths.stage / "tensors").mkdir(parents=True, exist_ok=True)
    torch.save(torch.zeros(3), outside / "t.pt")
    (paths.stage / "tensors" / "blocks.0.mlp.down_proj.weight.pt").symlink_to(outside / "t.pt")
    reply = d._dispatch("save_edit", '{"final": false}')
    assert "integrity check" in reply and d.st.edits_saved == 0


@pytest.mark.parametrize(
    "content",
    [b"", b"not a checkpoint", "numpy"],
    ids=["empty", "garbage", "numpy"],
)
def test_bad_staged_files_are_refused_and_the_run_continues(
    setup: tuple[RunPaths, Path], content: bytes | str
) -> None:
    # harness audit t4: EOFError / UnpicklingError escaped save_edit and ended the run
    paths, _ = setup
    tdir = paths.stage / "tensors"
    tdir.mkdir(parents=True, exist_ok=True)
    target = tdir / "blocks.0.mlp.down_proj.weight.pt"
    if content == "numpy":
        torch.save(np.zeros((3, 3), dtype=np.float32), target)
    else:
        assert isinstance(content, bytes)
        target.write_bytes(content)
    editor = ScriptedEditor([block("save_edit", '{"final": false}'), "no block"])
    d = make_driver(paths, editor, max_turns=2)
    st = d.run()
    assert st.stop_reason == "turn_cap: 2 editor replies" and not st.crashed
    assert "save_edit refused the staged edit" in editor.prompts[1]


def test_failed_turn_usage_is_charged_once(setup: tuple[RunPaths, Path]) -> None:
    # codex review #8: totals 100, 150 (failed after emitting usage), 200 were counted as 250
    paths, _ = setup
    seq = [ok("no block", "A", 100), fail(["Error: exit status 1"], "A", Usage(150)),
           ok("no block", "A", 200)]  # fmt: skip
    d = make_driver(paths, ScriptedEditor(seq))  # type: ignore[arg-type]
    m = "go"
    for _ in range(3):
        m = d._turn(m)
    assert d.guard.usage.input_tokens == 200
    assert d.guard.replies == 2 and d.guard.attempts == 3


def test_refusal_right_after_handoff_resets_thread_accounting(setup: tuple[RunPaths, Path]) -> None:
    # harness audit t6 / m1: 35k counted vs 50k true, and a second handoff fired at once
    paths, _ = setup
    refused = TurnResult(False, "", "new", Usage(5_000), R, [], {},
                         ['{"message":"flagged: usage policy"}'], 0.0, refused=True)  # fmt: skip
    seq: list[str | TurnResult] = [refused]
    seq += [ok("no block", "new", 5_000 + 15_000 * i) for i in range(1, 4)]
    d = make_driver(paths, ScriptedEditor(seq))
    d.st.thread_id, d.st.since_handoff, d.st.last_input_tokens = "old", 40, 60_000
    d.st.thread_totals = {"old": Usage(3_000_000)}
    m = d._maybe_handoff("last result")
    for _ in range(4):
        m = d._maybe_handoff(d._turn(m))
    assert d.st.handoffs == 1
    assert d.guard.usage.input_tokens == 50_000
    assert d.st.last_input_tokens == 15_000


def test_throttled_turns_keep_the_reply_cap(
    setup: tuple[RunPaths, Path], no_sleep: list[float]
) -> None:
    paths, _ = setup
    e429 = '{"type": "error", "message": "stream error: 429 Too Many Requests: Rate limit reached"}'
    seq: list[str | TurnResult] = [fail([e429], transient=True)] * 3 + ["no block"] * 3
    d = make_driver(paths, ScriptedEditor(seq), max_turns=2)
    st = d.run()
    assert st.stop_reason == "turn_cap: 2 editor replies" and st.turn == 5
    assert no_sleep == [120.0, 240.0, 480.0]


def test_attempt_cap_bounds_endless_failures(setup: tuple[RunPaths, Path]) -> None:
    paths, _ = setup
    e = fail(
        ['{"type": "error", "message": "Reconnecting... (request timed out)"}'], transient=True
    )
    d = make_driver(paths, ScriptedEditor([e]), max_turns=10, max_attempts=4)
    st = d.run()
    assert st.stop_reason is not None and st.stop_reason.startswith("attempt_cap: 4")


def test_quota_wording_stops_and_rate_limit_wording_backs_off(setup: tuple[RunPaths, Path]) -> None:
    paths, _ = setup
    quota = fail(['{"type": "error", "message": "You\'ve hit your usage limit."}'], limit_hit=True)
    d = make_driver(paths, ScriptedEditor([quota]))
    st = d.run()
    assert st.stop_reason is not None and st.stop_reason.startswith("codex usage limit")


def test_unverified_budget_stops_the_run(setup: tuple[RunPaths, Path]) -> None:
    paths, _ = setup
    no_rate = TurnResult(True, "no block", None, Usage(10), None, [], {}, [], 0.0)
    d = make_driver(paths, ScriptedEditor([no_rate]), stale_rate_turns=2)
    st = d.run()
    assert st.stop_reason is not None and st.stop_reason.startswith("budget_unverified")
    assert st.turn == 2


def test_nudge_limit_matches_its_message(setup: tuple[RunPaths, Path]) -> None:
    paths, _ = setup
    d = make_driver(paths, ScriptedEditor(["a", "b", "c", "d"]))
    st = d.run()
    assert st.stop_reason == "3 replies in a row without a block" and st.turn == 3


def test_refusals_are_counted_in_a_row(setup: tuple[RunPaths, Path]) -> None:
    paths, _ = setup
    r = fail(['{"message": "flagged for usage policy"}'], refused=True)
    seq: list[str | TurnResult] = [r, block("python", "1"), r, block("python", "2"), r]
    seq += [block("python", "3")] * 2
    d = make_driver(paths, ScriptedEditor(seq), max_turns=4)
    st = d.run()
    assert st.stop_reason == "turn_cap: 4 editor replies" and st.refusals == 0


def test_kernel_exit_between_cells_is_reported(setup: tuple[RunPaths, Path]) -> None:
    paths, _ = setup
    d = make_driver(paths, ScriptedEditor(["x"]))
    d._python("y = 7")
    d.kernel.proc = Dead()  # type: ignore[attr-defined]
    reply = d._python("y")
    assert "kernel had exited since your previous Python run" in reply
    row = json.loads((paths.private / "tool_calls.jsonl").read_text().splitlines()[-1])
    assert row["restart"] == "the kernel had exited since the previous run"


def test_kernel_integrity_violations_are_recorded_per_turn(setup: tuple[RunPaths, Path]) -> None:
    paths, _ = setup
    d = make_driver(paths, ScriptedEditor(["x"]))
    d.st.turn = 4
    d.kernel.last_integrity = [{"check": "grad_mode", "detail": "grad enabled after the cell"}]  # type: ignore[attr-defined]
    reply = d._python("torch._C._set_grad_enabled(True)")
    assert "integrity check after this run: grad_mode: grad enabled after the cell" in reply
    rows = violations(paths)
    assert rows == [rows[0]] and rows[0]["source"] == "kernel" and rows[0]["turn"] == 4
    assert rows[0]["check"] == "grad_mode" and d.st.integrity_violations == 1
    call = json.loads((paths.private / "tool_calls.jsonl").read_text().splitlines()[-1])
    assert call["integrity"][0]["check"] == "grad_mode"
    state = json.loads((d._save_state(), (paths.private / "state.json").read_text())[1])
    assert state["state"]["integrity_violations"] == 1


def test_editor_launch_errors_are_failures_not_crashes(setup: tuple[RunPaths, Path]) -> None:
    paths, _ = setup

    class Broken:
        def turn(self, prompt: str, thread_id: str | None, turn_dir: Path) -> TurnResult:
            raise FileNotFoundError("codex-switch")

    d = make_driver(paths, Broken())
    st = d.run()
    assert st.stop_reason == "codex failed 3 times in a row" and not st.crashed


def stub_eval(monkeypatch: pytest.MonkeyPatch, hidden_fails: bool) -> None:
    def hidden(base_ckpt: Path, edit: Path | None, n: int) -> dict[str, object]:
        if hidden_fails:
            raise RuntimeError("eval exploded")
        return {"addition_mean_n1_8": 0.5, "edit": str(edit)}

    monkeypatch.setattr(toy_eval, "evaluate_edit", hidden)
    monkeypatch.setattr(toy_eval, "base_reference", lambda c, d, n: {"base": True})
    monkeypatch.setattr(toy_eval, "eval_key", lambda c, n: {"n_per": n})


def test_crash_is_recorded_and_finalize_still_writes_everything(
    setup: tuple[RunPaths, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, _ = setup
    d = make_driver(paths, ScriptedEditor([block("python", "1")]))

    def boom(message: str) -> str:
        raise RuntimeError("driver bug")

    monkeypatch.setattr(d, "_turn", boom)
    with pytest.raises(RuntimeError):
        d.run()
    state = json.loads((paths.private / "state.json").read_text())
    assert state["state"]["crashed"] and state["state"]["stop_reason"].startswith(
        "crash: RuntimeError"
    )
    assert state["harness_version"] == "v4" and len(state["provenance"]["git_sha"]) == 40
    stub_eval(monkeypatch, hidden_fails=True)
    res = finalize(
        d.cfg, paths, d.st, n_per=2, mirror_root=tmp_path / "mirror", base_cache_dir=tmp_path / "c"
    )
    assert res["hidden_eval"] is None and "eval exploded" in str(res["errors"])
    assert res["crashed"] is True and res["scored"] is False and res["base_eval"] == {"base": True}
    saved = json.loads((tmp_path / "mirror" / "r" / "result.json").read_text())
    assert saved["provenance"]["harness_version"] == "v4" and "dirty" in saved["provenance"]
    assert (tmp_path / "mirror" / "r" / "audit.json").exists()


def test_isolation_breach_is_void(
    setup: tuple[RunPaths, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, _ = setup
    tool = TurnResult(True, "x", "t", Usage(1), R, ["command_execution"], {}, [], 0.0)
    d = make_driver(paths, ScriptedEditor([tool]))
    st = d.run()
    assert st.stop_reason is not None and st.stop_reason.startswith("isolation:")
    stub_eval(monkeypatch, hidden_fails=False)
    res = finalize(
        d.cfg, paths, st, n_per=2, mirror_root=tmp_path / "m", base_cache_dir=tmp_path / "c"
    )
    assert res["void"] is True and res["scored"] is False and res["hidden_eval"] is not None


def test_mirror_skips_linked_notes(setup: tuple[RunPaths, Path], tmp_path: Path) -> None:
    paths, outside = setup
    (paths.work / "notes.md").unlink()
    (paths.work / "notes.md").symlink_to(outside / "secret.txt")
    (paths.private / "kernel_tmp").mkdir(exist_ok=True)
    (paths.private / "kernel_tmp" / "x.json").write_text("{}")
    problems = mirror(paths, tmp_path / "m")
    assert [w for w, _ in problems] == ["mirror notes.md"]
    assert not (tmp_path / "m" / "notes.md").exists() and (tmp_path / "m" / "TASK.md").exists()
    assert not (tmp_path / "m" / "kernel_tmp").exists()


def load_script(name: str) -> object:
    script = Path(__file__).parents[2] / "experiments" / "phase3" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_editor_refuses_a_dirty_tree_or_another_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = load_script("run_editor")
    state = {"dirty": True, "dirty_files": ["src/x.py"], "git_sha": "a" * 40}
    monkeypatch.setattr(mod, "git_state", lambda: state)
    with pytest.raises(SystemExit, match="dirty"):
        mod.check_code(False, None)  # type: ignore[attr-defined]
    mod.check_code(True, None)  # type: ignore[attr-defined]
    with pytest.raises(SystemExit, match="expected"):
        mod.check_code(True, "b" * 40)  # type: ignore[attr-defined]


def test_finalize_run_rebuilds_a_killed_run(
    setup: tuple[RunPaths, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, _ = setup
    d = make_driver(paths, ScriptedEditor([block("python", "1")]), max_turns=1)
    d.run()
    state = json.loads((paths.private / "state.json").read_text())
    state["state"]["stop_reason"] = None  # as if killed mid-run
    state["pid"] = 999_999_999
    (paths.private / "state.json").write_text(json.dumps(state))
    assert RunState.from_dict(state["state"]).python_runs == 1
    mod = load_script("finalize_run")
    stub_eval(monkeypatch, hidden_fails=False)
    res = mod.finalize_from_state(paths.root, 2, tmp_path / "m", tmp_path / "c")  # type: ignore[attr-defined]
    assert str(res["stop_reason"]).startswith("killed:") and res["crashed"] is True
    assert res["counters"]["python_runs"] == 1
    assert (tmp_path / "m" / "r" / "result.json").exists()


# ---- v3.1 (codex re-review v3) ----


def rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def test_cells_py_as_a_directory_keeps_the_code_in_the_private_log(
    setup: tuple[RunPaths, Path],
) -> None:
    # codex review v3 #3: IsADirectoryError dropped the executed code from tool_calls.jsonl
    paths, _ = setup
    (paths.work / "cells.py").mkdir()
    d = make_driver(paths, ScriptedEditor(["x"]))
    reply = d._python("secret_step = 42")
    assert "OUTPUT of Python run 1" in reply and "integrity check" in reply
    ex = rows(paths.private / "exec.jsonl")
    assert [r["event"] for r in ex] == ["start", "done"]
    assert ex[0]["code"] == "secret_step = 42" and len(str(ex[0]["code_sha256"])) == 64
    assert ex[1]["output"] == "ran" and ex[1]["ok"] is True
    call = rows(paths.private / "tool_calls.jsonl")[-1]
    assert call["code"] == "secret_step = 42"


def test_code_is_logged_before_execution_and_output_is_not_cut(
    setup: tuple[RunPaths, Path],
) -> None:
    paths, _ = setup

    class LongKernel(StubKernel):
        def run(self, code: str) -> tuple[str, bool, str | None]:
            assert rows(paths.private / "exec.jsonl")[-1]["code"] == code  # already on disk
            return "y" * 500, True, None

    d = make_driver(paths, ScriptedEditor(["x"]), output_chars=50)
    d.kernel = LongKernel()  # type: ignore[assignment]
    d.kernel.start()
    reply = d._python("print('y' * 500)")
    assert "output cut at 50" in reply
    assert rows(paths.private / "exec.jsonl")[-1]["output"] == "y" * 500
    assert rows(paths.private / "tool_calls.jsonl")[-1]["output"] == "y" * 500


def test_truncated_integrity_log_still_finalizes(
    setup: tuple[RunPaths, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # codex review v3 #7: a partial JSON line made finalize raise before result.json
    paths, _ = setup
    d = make_driver(paths, ScriptedEditor([block("python", "1")]), max_turns=1)
    st = d.run()
    with (paths.private / "integrity.jsonl").open("a") as f:
        f.write('{"turn": 1, "source": "driver"}\n{"turn": 2, "sou')
    stub_eval(monkeypatch, hidden_fails=False)
    res = finalize(d.cfg, paths, st, 2, tmp_path / "m", tmp_path / "c")
    assert res["compliance"] == "pending_review" and res["review"] is None
    assert res["integrity_count"] == 2  # one valid row, one corrupt line
    assert res["budget_verified"] is False  # no pre-run reading
    assert isinstance(res["audit_ok"], bool)
    assert json.loads((tmp_path / "m" / "r" / "result.json").read_text())["integrity_count"] == 2


def test_mirror_findings_are_in_the_published_result(
    setup: tuple[RunPaths, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, outside = setup
    d = make_driver(paths, ScriptedEditor([block("python", "1")]), max_turns=1)
    st = d.run()
    (paths.work / "notes.md").unlink()
    (paths.work / "notes.md").symlink_to(outside / "secret.txt")
    stub_eval(monkeypatch, hidden_fails=False)
    res = finalize(d.cfg, paths, st, 2, tmp_path / "m", tmp_path / "c")
    assert res["integrity_count"] == 1
    assert json.loads((tmp_path / "m" / "r" / "result.json").read_text())["integrity_count"] == 1


def test_finalize_survives_an_unreadable_state_file(
    setup: tuple[RunPaths, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, _ = setup
    d = make_driver(paths, ScriptedEditor([block("python", "1")]), max_turns=1)
    st = d.run()
    (paths.private / "state.json").write_text('{"cut')
    stub_eval(monkeypatch, hidden_fails=False)
    res = finalize(d.cfg, paths, st, 2, tmp_path / "m", tmp_path / "c")
    assert "state_load" in res["errors"] and res["budget_verified"] is False  # type: ignore[operator]
    assert (tmp_path / "m" / "r" / "result.json").exists()


def test_budget_needs_a_fresh_pre_run_baseline(setup: tuple[RunPaths, Path]) -> None:
    # codex review v3 #9: without a pre-run reading, early spending vanished from the budget
    now = time.time()
    fresh = RateReading(10.0, int(now) + 86400, 10080, "x", now - 60)
    assert fresh_baseline(fresh, now, 3600)
    assert not fresh_baseline(None, now, 3600)
    assert not fresh_baseline(
        RateReading(10.0, int(now) + 86400, 10080, "x", now - 7200), now, 3600
    )
    assert not fresh_baseline(RateReading(10.0, int(now) + 86400, 300, "x", now), now, 3600)
    assert not fresh_baseline(RateReading(10.0, int(now) - 1, 10080, "x", now), now, 3600)
    paths, _ = setup
    for pre, verified in ((None, False), (fresh, True)):
        d = make_driver(paths, ScriptedEditor(["no block"]), max_turns=1)
        d.run(pre)
        snap = json.loads((paths.private / "state.json").read_text())["budget"]
        assert snap["budget_verified"] is verified


def test_copy_bounds_the_bytes_read_not_just_the_reported_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # codex review v3 #6: the size was checked once, then copyfileobj read to EOF
    work = tmp_path / "work"
    (work / "edited" / "tensors").mkdir(parents=True)
    (work / "edited" / "tensors" / "a.pt").write_bytes(b"x" * 5 * 2**20)
    with pytest.raises(IntegrityError, match="size limit"):
        copy_tree_regular(work, "edited", tmp_path / "snap", 2**20, 4 * 2**20)
    assert (tmp_path / "snap" / "tensors" / "a.pt").stat().st_size <= 2**20


def _alive(tag: str) -> list[str]:
    return subprocess.run(
        ["pgrep", "-f", tag], check=False, capture_output=True, text=True
    ).stdout.split()


def test_group_member_that_ignores_term_is_killed_after_the_parent_exits(tmp_path: Path) -> None:
    # codex review v3 #10: parent reaped -> empty snapshot -> TERM only, the member survived
    tag = "sleep 63.1415"
    cmd = ["sh", "-c", f"(trap '' TERM; exec {tag}) >/dev/null 2>&1 & echo started"]
    out, _, rc = run_group(cmd, "", tmp_path, {"PATH": "/usr/bin:/bin"}, 30.0)
    assert rc == 0 and out.strip() == "started"
    time.sleep(0.2)
    assert _alive(tag) == []


def test_unhealthy_watchdog_stops_the_run(
    tmp_path: Path, tiny_ckpt: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import weight_splice_lab.harness.driver as drv

    monkeypatch.setattr(drv, "watchdog_healthy", lambda: False)
    cfg = RunConfig(run_id="wd", forward_budget=2, kernel_timeout_s=60, require_watchdog=True)
    paths = make_run(tmp_path, tiny_ckpt, cfg)
    st = Driver(cfg, paths, ScriptedEditor(["no block"])).run()
    assert st.stop_reason is not None and st.stop_reason.startswith("watchdog_unhealthy")
    assert st.turn == 0
