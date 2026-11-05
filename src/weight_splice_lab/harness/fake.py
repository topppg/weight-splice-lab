import json
import time
from dataclasses import replace
from pathlib import Path

from weight_splice_lab.harness.budget import RateReading, Usage
from weight_splice_lab.harness.codex import TurnResult


# Replays fixed replies in place of Codex, with synthetic usage and rate limits. Like Codex,
# usage is reported as the thread's running total. A TurnResult in the script is returned as is
# (thread id filled in when missing), to replay recorded failures.
class ScriptedEditor:
    def __init__(self, replies: list[str | TurnResult], points_per_turn: float = 0.0) -> None:
        self.replies = replies
        self.points_per_turn = points_per_turn
        self.calls = 0
        self.prompts: list[str] = []
        self.per_turn_input: list[int] = []
        self.totals: dict[str, Usage] = {}

    def turn(self, prompt: str, thread_id: str | None, turn_dir: Path) -> TurnResult:
        turn_dir.mkdir(parents=True, exist_ok=True)
        (turn_dir / "prompt.txt").write_text(prompt)
        self.prompts.append(prompt)
        item = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        if isinstance(item, TurnResult):
            (turn_dir / "run.json").write_text(json.dumps({"fake": True, "audit": {
                "item_types": item.item_types, "tool_items": item.tool_items,
                "tool_free": not item.tool_items}}))  # fmt: skip
            return replace(item, thread_id=item.thread_id or thread_id)
        text = item
        (turn_dir / "answer.md").write_text(text)
        rate = RateReading(10.0 + self.points_per_turn * self.calls, 1_900_000_000, 10080, "fake")
        tid = thread_id or f"fake-{time.time_ns()}"
        total = self.totals.setdefault(tid, Usage())
        total.add(Usage(len(prompt) // 4, 0, len(text) // 4, 0))
        self.per_turn_input.append(len(prompt) // 4)
        usage = Usage(**vars(total))
        audit = {"item_types": {"agent_message": 1}, "tool_items": [], "tool_free": True}
        (turn_dir / "run.json").write_text(json.dumps({"fake": True, "audit": audit}))
        return TurnResult(
            ok=True, text=text, thread_id=tid, usage=usage,
            rate=rate, tool_items=[], item_types={"agent_message": 1}, errors=[], wall_s=0.0,
        )  # fmt: skip
