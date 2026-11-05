import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from weight_splice_lab.harness import codex
from weight_splice_lab.harness.codex import (
    CodexEditor,
    CodexSettings,
    ErrorKind,
    audit_events,
    classify_errors,
    run_group,
)


def test_error_classes_follow_message_text_only() -> None:
    cases = {
        # harness audit t1 / M6: these were "usage limit" and stopped the run
        '{"type": "error", "message": "stream error: 429 Too Many Requests: Rate limit reached for requests"}': ErrorKind(False, False, True),
        '{"type": "turn.failed", "error": {"message": "rate limit exceeded, retry in 20s"}}': ErrorKind(False, False, True),
        '{"type": "error", "message": "Reconnecting... 2/5 (request timed out)", "rate_limits": {}}': ErrorKind(False, False, True),
        '{"type": "error", "message": "exceeded retry limit, last status: 429 Too Many Requests"}': ErrorKind(False, False, True),
        "Error: thread/resume failed: thread 01a1 already has an active writer (code -32600)": ErrorKind(False, False, True),
        '{"type": "error", "message": "You\'ve hit your usage limit. Try again in 3 days."}': ErrorKind(False, True, False),
        '{"type": "error", "message": "invalid model gpt-x", "rate_limits": {"primary": {}}}': ErrorKind(False, False, False),
        '{"type": "error", "message": "your prompt was flagged as violating our usage policy"}': ErrorKind(True, False, False),
    }  # fmt: skip
    for err, want in cases.items():
        assert classify_errors([err]) == want, err


def test_all_agent_messages_are_parsed_in_order() -> None:
    ev = [
        {"type": "item.completed", "item": {"type": "agent_message", "text": "```python\nprint(1)\n```"}},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "Done."}},
        {"type": "turn.completed", "usage": {"input_tokens": 1}},
    ]  # fmt: skip
    assert audit_events(ev).text == "```python\nprint(1)\n```\n\nDone."


def _alive(tag: str) -> list[str]:
    return subprocess.run(
        ["pgrep", "-f", tag], check=False, capture_output=True, text=True
    ).stdout.split()


def test_timeout_kills_the_whole_process_group(tmp_path: Path) -> None:
    # harness audit M6: only the direct child was killed; the grandchild kept the thread's writer
    tag = "sleep 61.4142"
    cmd = ["sh", "-c", f"{tag} & {tag} & wait"]
    t0 = time.time()
    _, err, rc = run_group(cmd, "", tmp_path, {"PATH": "/usr/bin:/bin"}, 1.0)
    assert rc == -1 and "timed out after" in err and time.time() - t0 < 20
    assert _alive(tag) == []


def test_a_finished_call_leaves_no_group_members(tmp_path: Path) -> None:
    tag = "sleep 62.7182"
    out, _, rc = run_group(
        ["sh", "-c", f"({tag} >/dev/null 2>&1 &) ; echo hi"],
        "",
        tmp_path,
        {"PATH": "/usr/bin:/bin"},
        30.0,
    )
    assert rc == 0 and out.strip() == "hi"
    time.sleep(0.2)
    assert _alive(tag) == []


def _rollout(sessions: Path, tid: str, ts: datetime) -> None:
    day = sessions / "2026" / "11" / "06"
    day.mkdir(parents=True, exist_ok=True)
    row = {"timestamp": ts.isoformat().replace("+00:00", "Z"), "type": "event_msg",
           "payload": {"type": "token_count", "rate_limits": {
               "primary": {"used_percent": 12.0, "window_minutes": 10080, "resets_at": 1900000000}}}}  # fmt: skip
    (day / f"rollout-2026-11-06T00-00-00-{tid}.jsonl").write_text(json.dumps(row) + "\n")


@pytest.mark.parametrize("fresh", [True, False])
def test_rollout_reading_from_before_the_call_is_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fresh: bool
) -> None:
    sessions = tmp_path / "home" / "sessions"
    monkeypatch.setattr(codex, "codex_home", lambda profile: tmp_path / "home")
    events = [{"type": "thread.started", "thread_id": "tid9"},
              {"type": "item.completed", "item": {"type": "agent_message", "text": "hi"}},
              {"type": "turn.completed", "usage": {"input_tokens": 5}}]  # fmt: skip
    lines = "\n".join(json.dumps(e) for e in events)
    if not fresh:
        _rollout(sessions, "tid9", datetime(2026, 1, 1, tzinfo=UTC))

    def fake_cmd(s: CodexSettings, thread_id: str | None, last: Path) -> list[str]:
        if fresh:  # Codex writes the turn's token_count while the call runs
            _rollout(sessions, "tid9", datetime.now(UTC))
        return ["sh", "-c", f"cat >/dev/null; printf '%s\\n' '{lines}'"]

    monkeypatch.setattr(codex, "build_command", fake_cmd)
    ed = CodexEditor(CodexSettings(instructions_file=tmp_path / "i.md"), tmp_path)
    res = ed.turn("prompt", None, tmp_path / "t001")
    run = json.loads((tmp_path / "t001" / "run.json").read_text())
    assert res.ok and res.thread_id == "tid9"
    if fresh:
        assert res.rate is not None and res.rate.used_percent == 12.0 and run["stale_rate"] is None
    else:
        assert res.rate is None and run["stale_rate"]["used_percent"] == 12.0
