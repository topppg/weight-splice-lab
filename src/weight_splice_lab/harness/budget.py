import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

WEEK_MINUTES = 10080


@dataclass(frozen=True)
class RateReading:
    used_percent: float
    resets_at: int | None
    window_minutes: int | None
    source: str = ""
    observed_at: float | None = None


@dataclass
class Usage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0

    def since(self, prev: "Usage") -> "Usage":
        return Usage(
            max(0, self.input_tokens - prev.input_tokens),
            max(0, self.cached_input_tokens - prev.cached_input_tokens),
            max(0, self.output_tokens - prev.output_tokens),
            max(0, self.reasoning_output_tokens - prev.reasoning_output_tokens),
        )

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.cached_input_tokens += other.cached_input_tokens
        self.output_tokens += other.output_tokens
        self.reasoning_output_tokens += other.reasoning_output_tokens


def usage_from_event(event: dict[str, object]) -> Usage | None:
    if event.get("type") != "turn.completed":
        return None
    u = event.get("usage")
    if not isinstance(u, dict):
        return None
    return Usage(
        int(u.get("input_tokens", 0)),
        int(u.get("cached_input_tokens", 0)),
        int(u.get("output_tokens", 0)),
        int(u.get("reasoning_output_tokens", 0)),
    )


def _find_rate_limits(obj: object) -> dict[str, object] | None:
    if isinstance(obj, dict):
        rl = obj.get("rate_limits")
        if isinstance(rl, dict):
            return rl
        for v in obj.values():
            found = _find_rate_limits(v)
            if found is not None:
                return found
    return None


def reading_from_rate_limits(
    rl: dict[str, object], source: str, observed_at: float | None = None
) -> RateReading | None:
    # Only a window positively identified as weekly counts; a 5-hour window is never read as weekly.
    candidates = [rl.get("primary"), rl.get("secondary")]
    weekly = [
        w for w in candidates
        if isinstance(w, dict) and "used_percent" in w and w.get("window_minutes") == WEEK_MINUTES
    ]  # fmt: skip
    if not weekly:
        return None
    w = weekly[0]
    resets = w.get("resets_at")
    return RateReading(
        float(w["used_percent"]),
        int(resets) if isinstance(resets, (int, float)) else None,
        WEEK_MINUTES,
        source,
        observed_at,
    )


def reading_from_events(
    events: list[dict[str, object]], observed_at: float | None = None
) -> RateReading | None:
    reading = None
    for e in events:
        rl = _find_rate_limits(e)
        if rl is not None:
            reading = reading_from_rate_limits(rl, "events", observed_at) or reading
    return reading


def _timestamp(row: object) -> float | None:
    ts = row.get("timestamp") if isinstance(row, dict) else None
    if not isinstance(ts, str):
        return None
    try:
        return datetime.fromisoformat(ts).timestamp()
    except ValueError:
        return None


def reading_from_rollout(path: Path) -> RateReading | None:
    reading = None
    with path.open() as f:
        for line in f:
            if '"rate_limits"' not in line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rl = _find_rate_limits(row)
            if rl is not None:
                got = reading_from_rate_limits(rl, f"rollout:{path.name}", _timestamp(row))
                reading = got or reading
    return reading


def latest_reading(sessions: Path, n_files: int = 20) -> RateReading | None:
    files = sorted(sessions.glob("*/*/*/rollout-*.jsonl"), key=lambda p: p.stat().st_mtime)
    for path in reversed(files[-n_files:]):
        reading = reading_from_rollout(path)
        if reading is not None:
            return reading
    return None


def find_rollout(sessions: Path, thread_id: str) -> Path | None:
    hits = sorted(sessions.glob(f"*/*/*/rollout-*{thread_id}.jsonl"))
    return hits[-1] if hits else None


@dataclass
class Limits:
    max_limit_points: float = 5.0
    max_input_tokens: int = 20_000_000
    max_turns: int = 200
    max_attempts: int = 260
    stale_rate_turns: int = 3


NEW_WINDOW_MIN_SHIFT_S = 3600


def _new_window(prev: RateReading, reading: RateReading) -> bool:
    # resets_at jitters by a second between reports; only a forward jump marks a new window
    if prev.resets_at is None or reading.resets_at is None:
        return False
    return reading.resets_at - prev.resets_at > NEW_WINDOW_MIN_SHIFT_S


def fresh_baseline(reading: RateReading | None, now: float, max_age_s: float) -> bool:
    return (
        reading is not None
        and reading.window_minutes == WEEK_MINUTES
        and (reading.resets_at or 0) > now
        and reading.observed_at is not None
        and now - max_age_s <= reading.observed_at <= now + 60
    )


@dataclass
class BudgetGuard:
    limits: Limits
    baseline: RateReading | None = None
    last: RateReading | None = None
    points_before_reset: float = 0.0
    usage: Usage = field(default_factory=Usage)
    replies: int = 0
    attempts: int = 0
    unverified_turns: int = 0
    # a fresh weekly reading taken before the run spends anything; set once, at start
    baseline_verified: bool = False

    def record_turn(
        self, usage: Usage | None, reading: RateReading | None, replied: bool = True
    ) -> None:
        # replies count toward max_turns; every editor call counts toward max_attempts
        self.attempts += 1
        if replied:
            self.replies += 1
        if usage is not None:
            self.usage.add(usage)
        if reading is not None:
            self.observe(reading)
            self.unverified_turns = 0
        elif replied or usage is not None:
            self.unverified_turns += 1

    def observe(self, reading: RateReading) -> None:
        prev = self.last or self.baseline
        if self.baseline is None:
            self.baseline = reading
        elif prev is not None and _new_window(prev, reading):
            # New weekly window: bank what was used in the old one, count all of the new one.
            self.points_before_reset += max(0.0, prev.used_percent - self.baseline.used_percent)
            self.baseline = RateReading(0.0, reading.resets_at, reading.window_minutes, "reset")
        self.last = reading

    @property
    def points_used(self) -> float:
        if self.baseline is None or self.last is None:
            return self.points_before_reset
        return self.points_before_reset + max(
            0.0, self.last.used_percent - self.baseline.used_percent
        )

    def stop_reason(self, forward_left: int, final_submitted: bool) -> str | None:
        lim = self.limits
        if self.last is not None and self.last.used_percent >= 100.0:
            return f"codex usage limit: weekly window at {self.last.used_percent:g}%"
        if self.points_used >= lim.max_limit_points:
            return f"rate_limit: weekly usage rose {self.points_used:g} points (cap {lim.max_limit_points:g})"
        if self.unverified_turns >= lim.stale_rate_turns:
            return (
                f"budget_unverified: no fresh weekly rate-limit reading for "
                f"{self.unverified_turns} turns in a row"
            )
        if self.usage.input_tokens >= lim.max_input_tokens:
            return f"input_tokens: {self.usage.input_tokens:,} (cap {lim.max_input_tokens:,})"
        if self.replies >= lim.max_turns:
            return f"turn_cap: {self.replies} editor replies"
        if self.attempts >= lim.max_attempts:
            return f"attempt_cap: {self.attempts} editor calls ({self.replies} replies)"
        if forward_left <= 0 and final_submitted:
            return "done: forward budget used and final edit submitted"
        return None

    def snapshot(self) -> dict[str, object]:
        return {
            "replies": self.replies,
            "attempts": self.attempts,
            "unverified_turns": self.unverified_turns,
            "budget_verified": self.baseline_verified,
            "usage": asdict(self.usage),
            "points_used": self.points_used,
            "points_before_reset": self.points_before_reset,
            "baseline": asdict(self.baseline) if self.baseline else None,
            "last": asdict(self.last) if self.last else None,
            "limits": asdict(self.limits),
        }

    @classmethod
    def restore(cls, snap: dict[str, object]) -> "BudgetGuard":
        def rr(d: object) -> RateReading | None:
            return RateReading(**d) if isinstance(d, dict) else None

        lim = snap.get("limits")
        g = cls(Limits(**lim) if isinstance(lim, dict) else Limits())
        g.baseline, g.last = rr(snap.get("baseline")), rr(snap.get("last"))
        u = snap.get("usage")
        g.usage = Usage(**u) if isinstance(u, dict) else Usage()
        old_turns = snap.get("turns", 0)  # v2 snapshots had one counter
        for key in ("replies", "attempts", "unverified_turns"):
            v = snap.get(key, old_turns if key != "unverified_turns" else 0)
            setattr(g, key, int(v) if isinstance(v, (int, float)) else 0)
        g.baseline_verified = snap.get("budget_verified") is True
        pbr = snap.get("points_before_reset", 0.0)
        g.points_before_reset = float(pbr) if isinstance(pbr, (int, float)) else 0.0
        return g
