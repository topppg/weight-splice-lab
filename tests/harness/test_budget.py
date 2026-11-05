from pathlib import Path

from conftest import DATA

from weight_splice_lab.harness.budget import (
    BudgetGuard,
    Limits,
    RateReading,
    Usage,
    find_rollout,
    latest_reading,
    reading_from_events,
    reading_from_rollout,
)
from weight_splice_lab.harness.codex import (
    CodexSettings,
    ErrorKind,
    audit_events,
    build_command,
    classify_errors,
    parse_jsonl,
)


def events(name: str) -> list[dict[str, object]]:
    return parse_jsonl((DATA / name).read_text())


def test_audit_tool_free_turn() -> None:
    a = audit_events(events("exec_events.jsonl"))
    assert a.thread_id == "id-f4333c3a"
    assert a.tool_free and a.turn_completed
    assert a.item_types == {"error": 1, "reasoning": 1, "agent_message": 1}
    assert "```python" in a.text
    assert a.usage == Usage(5917, 0, 492, 187)


def test_audit_flags_tool_items() -> None:
    a = audit_events(events("exec_events_tool.jsonl"))
    assert not a.tool_free and a.tool_items == ["command_execution"]


def test_failed_turn_is_classified_as_refusal() -> None:
    a = audit_events(events("exec_events_failed.jsonl"))
    assert not a.turn_completed and len(a.errors) == 2
    assert classify_errors(a.errors) == ErrorKind(refused=True, quota=False, transient=False)
    assert classify_errors(["You've hit your usage limit."]) == ErrorKind(False, True, False)


def test_rate_limits_from_rollout_and_events() -> None:
    r = reading_from_rollout(DATA / "rollout.jsonl")
    assert r is not None and r.observed_at is not None
    assert r == RateReading(5.0, 1793602835, 10080, "rollout:rollout.jsonl", r.observed_at)
    assert reading_from_events(events("exec_events.jsonl")) is None
    weekly = {"used_percent": 7.0, "window_minutes": 10080}
    rows = [{"type": "event_msg", "payload": {"rate_limits": {"primary": weekly}}}]
    got = reading_from_events(rows)
    assert got is not None and got.used_percent == 7.0


def test_only_a_positively_weekly_window_is_read() -> None:
    # codex review #7: a 5-hour or unlabelled window used to be taken as the weekly one
    for primary in ({"used_percent": 40.0, "window_minutes": 300}, {"used_percent": 40.0}):
        assert reading_from_events([{"rate_limits": {"primary": primary}}]) is None


def test_weekly_window_is_preferred() -> None:
    rows = [{"rate_limits": {
        "primary": {"used_percent": 40.0, "window_minutes": 300, "resets_at": 1},
        "secondary": {"used_percent": 9.0, "window_minutes": 10080, "resets_at": 2},
    }}]  # fmt: skip
    got = reading_from_events(rows)
    assert got is not None and got.used_percent == 9.0 and got.resets_at == 2


def test_find_rollout_and_latest(tmp_path: Path) -> None:
    day = tmp_path / "2026" / "11" / "05"
    day.mkdir(parents=True)
    target = day / "rollout-2026-11-05T10-00-00-abc123.jsonl"
    target.write_text((DATA / "rollout.jsonl").read_text())
    assert find_rollout(tmp_path, "abc123") == target
    assert find_rollout(tmp_path, "missing") is None
    latest = latest_reading(tmp_path)
    assert latest is not None and latest.used_percent == 5.0


def reading(pct: float, resets: int = 100) -> RateReading:
    return RateReading(pct, resets, 10080)


def test_guard_rate_limit_points() -> None:
    g = BudgetGuard(Limits(max_limit_points=5.0))
    g.observe(reading(10.0))
    g.record_turn(Usage(1000), reading(12.0))
    assert g.points_used == 2.0 and g.stop_reason(10, False) is None
    g.record_turn(Usage(1000), reading(14.9))
    assert g.stop_reason(10, False) is None
    g.record_turn(Usage(1000), reading(15.0))
    reason = g.stop_reason(10, False)
    assert reason is not None and reason.startswith("rate_limit")


def test_guard_baseline_from_first_turn_without_pre_reading() -> None:
    g = BudgetGuard(Limits())
    g.record_turn(Usage(10), reading(30.0))
    g.record_turn(Usage(10), reading(31.0))
    assert g.points_used == 1.0


def test_guard_counts_across_weekly_reset() -> None:
    g = BudgetGuard(Limits(max_limit_points=5.0))
    week = 7 * 24 * 3600
    g.observe(reading(97.0, resets=1_794_390_356))
    g.record_turn(None, reading(99.0, resets=1_794_390_356))
    g.record_turn(None, reading(2.0, resets=1_794_390_356 + week))
    assert g.points_used == 4.0
    g.record_turn(None, reading(3.0, resets=1_794_390_356 + week))
    assert g.stop_reason(10, False) is not None


def test_guard_ignores_resets_at_jitter() -> None:
    # pilot1 was stopped at 8 "points" when resets_at moved by one second at 8.0% usage
    g = BudgetGuard(Limits(max_limit_points=5.0))
    g.observe(reading(8.0, resets=1_794_390_356))
    g.record_turn(None, reading(8.0, resets=1_794_390_356))
    g.record_turn(None, reading(8.0, resets=1_794_390_357))
    g.record_turn(None, reading(8.0, resets=1_794_390_356))
    assert g.points_used == 0.0
    assert g.stop_reason(10, False) is None


def test_guard_input_tokens_turns_and_done() -> None:
    g = BudgetGuard(Limits(max_input_tokens=5000, max_turns=3))
    g.record_turn(Usage(3000), None)
    assert g.stop_reason(1, False) is None
    g.record_turn(Usage(2500), None)
    reason = g.stop_reason(1, False)
    assert reason is not None and reason.startswith("input_tokens")
    g = BudgetGuard(Limits(max_turns=2))
    g.record_turn(None, None)
    assert g.stop_reason(0, False) is None
    assert g.stop_reason(0, True) == "done: forward budget used and final edit submitted"
    assert g.stop_reason(1, True) is None
    g.record_turn(None, None)
    reason = g.stop_reason(5, False)
    assert reason is not None and reason.startswith("turn_cap")


def test_command_shape(tmp_path: Path) -> None:
    s = CodexSettings(instructions_file=tmp_path / "i.md", effort="low")
    first = build_command(s, None, tmp_path / "a.md")
    assert first[:7] == ["codex-switch", "run", "personal", "--", "codex", "--no-daemon", "exec"]
    assert first[-1] == "-" and "resume" not in first
    assert 'model_reasoning_effort="low"' in first and "features.shell_tool=false" in first
    assert "--ignore-user-config" in first and "--strict-config" in first
    later = build_command(s, "tid-1", tmp_path / "a.md")
    assert later[7] == "resume" and later[-2:] == ["tid-1", "-"]


def test_failed_attempts_do_not_use_the_reply_cap() -> None:
    # harness audit m2: throttled and failed turns used to count toward the 200 replies
    g = BudgetGuard(Limits(max_turns=2, max_attempts=5))
    for _ in range(3):
        g.record_turn(None, None, replied=False)
    assert g.replies == 0 and g.stop_reason(5, False) is None
    g.record_turn(Usage(1), reading(1.0))
    g.record_turn(Usage(1), reading(1.0))
    reason = g.stop_reason(5, False)
    assert reason is not None and reason.startswith("turn_cap: 2 editor replies")
    g = BudgetGuard(Limits(max_turns=10, max_attempts=3))
    for _ in range(3):
        g.record_turn(None, None, replied=False)
    reason = g.stop_reason(5, False)
    assert reason is not None and reason.startswith("attempt_cap")


def test_missing_weekly_readings_stop_the_run_as_unverified() -> None:
    # codex review #7: turns used to continue while no reading arrived
    g = BudgetGuard(Limits(stale_rate_turns=3))
    g.observe(reading(10.0))
    g.record_turn(Usage(10), None)
    g.record_turn(Usage(10), None)
    assert g.stop_reason(5, False) is None
    g.record_turn(None, None, replied=False)  # an attempt that cost nothing does not count
    assert g.unverified_turns == 2
    g.record_turn(Usage(10), reading(10.0))
    assert g.unverified_turns == 0
    for _ in range(3):
        g.record_turn(Usage(10), None, replied=False)  # failed turns that spent tokens count
    reason = g.stop_reason(5, False)
    assert reason is not None and reason.startswith("budget_unverified")


def test_weekly_window_at_100_percent_stops() -> None:
    g = BudgetGuard(Limits(max_limit_points=500.0))
    g.observe(reading(99.0))
    g.record_turn(Usage(1), reading(100.0))
    reason = g.stop_reason(5, False)
    assert reason is not None and reason.startswith("codex usage limit")


def test_guard_snapshot_round_trip() -> None:
    g = BudgetGuard(Limits(max_turns=7))
    g.observe(reading(3.0))
    g.record_turn(Usage(5, 1, 2, 0), reading(4.0))
    g.record_turn(None, None, replied=False)
    h = BudgetGuard.restore(g.snapshot())
    assert h.snapshot() == g.snapshot()
