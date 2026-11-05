import hashlib
import json
import os
import shutil
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Protocol

from weight_splice_lab.gpu import watchdog_healthy
from weight_splice_lab.harness.audit import audit_run
from weight_splice_lab.harness.budget import (
    BudgetGuard,
    Limits,
    RateReading,
    Usage,
    fresh_baseline,
)
from weight_splice_lab.harness.codex import TurnResult
from weight_splice_lab.harness.edit import (
    EditError,
    apply_edit,
    architecture,
    read_staged,
    save_edit,
    validate,
)
from weight_splice_lab.harness.family import TOY
from weight_splice_lab.harness.forward import (
    ForwardError,
    ForwardLimits,
    ForwardServer,
    load_practice,
)
from weight_splice_lab.harness.kernel import Kernel, KernelLimits
from weight_splice_lab.harness.protocol import cut, parse_json_body, parse_reply
from weight_splice_lab.harness.rundir import (
    HARNESS_VERSION,
    IntegrityError,
    RunConfig,
    RunPaths,
    append_regular,
    copy_tree_regular,
    provenance,
    read_regular,
)

# toy defaults, kept for the phase-3 scripts; finalize uses the run's family
REPO_RUNS = TOY.runs_dir
BASE_CACHE_DIR = TOY.base_cache_dir
MAX_NUDGES = 3
MAX_FAILURES = 3
MAX_REFUSALS = 3
THROTTLE_BACKOFF_MIN = (2, 4, 8, 16, 32, 60, 60)
NOTES_CHARS = 8000
STAGED_MAX_FILE = 64 * 2**20
STAGED_MAX_TOTAL = 512 * 2**20
BASELINE_MAX_AGE_S = 3600.0


def throttle_wait_s(n_throttles: int) -> float | None:
    if n_throttles < 1 or n_throttles > len(THROTTLE_BACKOFF_MIN):
        return None
    return THROTTLE_BACKOFF_MIN[n_throttles - 1] * 60.0


REFUSAL_NOTE = (
    "Your last turn could not be processed: the provider's content filter refused the "
    "conversation, so the result of your last block is withheld. Do not print raw vocabulary "
    "strings or long text excerpts; print ids, counts and short snippets instead. Continue."
)


class Editor(Protocol):
    def turn(self, prompt: str, thread_id: str | None, turn_dir: Path) -> TurnResult: ...


@dataclass
class RunState:
    turn: int = 0  # editor calls; successful replies are counted by the budget guard
    thread_id: str | None = None
    since_handoff: int = 0
    handoffs: int = 0
    python_runs: int = 0
    passes: int = 0
    edits_saved: int = 0
    latest_edit: int | None = None
    final_edit: int | None = None
    nudges: int = 0
    failures: int = 0
    throttles: int = 0
    refusals: int = 0
    last_input_tokens: int = 0
    thread_totals: dict[str, Usage] = field(default_factory=dict)
    ended_by_editor: bool = False
    integrity_violations: int = 0
    tool_errors: int = 0
    crashed: bool = False
    stop_reason: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> "RunState":
        names = {f.name for f in fields(cls)}
        st = cls(**{k: v for k, v in d.items() if k in names and k != "thread_totals"})  # type: ignore[arg-type]
        totals = d.get("thread_totals")
        if isinstance(totals, dict):
            st.thread_totals = {
                str(k): Usage(**v) for k, v in totals.items() if isinstance(v, dict)
            }
        return st


def failed_result(errors: list[str]) -> TurnResult:
    return TurnResult(False, "", None, None, None, [], {}, errors, 0.0)


class Driver:
    def __init__(self, cfg: RunConfig, paths: RunPaths, editor: Editor) -> None:
        self.cfg = cfg
        self.paths = paths
        self.editor = editor
        self.st = RunState()
        self.provenance = provenance()
        fam = cfg.fam
        self.base = fam.load_model(paths.sandbox / "model" / "base.pt")
        limits = ForwardLimits(max_items=fam.forward_items)
        if fam.name != "toy":
            limits.tf_batch_size = 1  # Gemma logits: 262,144 x 4 bytes per position
        self.fwd = ForwardServer(
            self.base, load_practice(paths.sandbox / "practice"), paths.work, limits, fam
        )
        self.kernel = Kernel(
            paths.sandbox, paths.private, KernelLimits(cfg.kernel_timeout_s, cfg.kernel_max_rss_gb)
        )
        self.guard = BudgetGuard(
            Limits(
                cfg.max_limit_points,
                cfg.max_input_tokens,
                cfg.max_turns,
                cfg.max_attempts,
                cfg.stale_rate_turns,
            )
        )
        self.task = (paths.sandbox / "TASK.md").read_text()

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with (self.paths.private / "driver.log").open("a") as f:
            f.write(line + "\n")

    def _append(self, name: str, row: dict[str, object]) -> None:
        with (self.paths.private / name).open("a") as f:
            f.write(json.dumps(row) + "\n")

    def _save_state(self) -> None:
        state = {
            "harness_version": HARNESS_VERSION,
            "provenance": self.provenance,
            "pid": os.getpid(),
            "saved_at": time.time(),
            "state": asdict(self.st),
            "budget": self.guard.snapshot(),
        }
        tmp = self.paths.private / "state.json.tmp"
        tmp.write_text(json.dumps(state, indent=1))
        os.replace(tmp, self.paths.private / "state.json")

    @property
    def passes_left(self) -> int:
        return self.cfg.forward_budget - self.st.passes

    @property
    def replies_left(self) -> int:
        return max(0, self.cfg.max_turns - self.guard.replies)

    def violation(self, where: str, detail: str, source: str = "driver") -> str:
        self.st.integrity_violations += 1
        self._append("integrity.jsonl", {"turn": self.st.turn, "time": time.time(),
                                         "source": source, "where": where, "detail": detail})  # fmt: skip
        self.log(f"integrity violation ({where}): {detail}")
        return (
            f"[harness integrity check: {detail}. The harness only reads and writes regular files "
            "inside work/ and never follows links; nothing was read or written there, and this "
            "was logged as an integrity violation.]"
        )

    def run(self, pre_run: RateReading | None = None) -> RunState:
        try:
            now = time.time()
            if pre_run is not None and (pre_run.resets_at or 0) > now:
                self.guard.observe(pre_run)
                self.log(f"baseline weekly usage {pre_run.used_percent}% ({pre_run.source})")
            # without a fresh pre-spending reading the run's budget is unverified for good
            self.guard.baseline_verified = fresh_baseline(pre_run, now, BASELINE_MAX_AGE_S)
            if not self.guard.baseline_verified:
                self.log(f"no fresh pre-run weekly reading ({pre_run}): budget_verified false")
            self._save_state()
            self.kernel.start()
            message = self.task
            while True:
                reason = self.guard.stop_reason(self.passes_left, self.st.final_edit is not None)
                if reason is None and self.st.ended_by_editor:
                    reason = (
                        "done: final edit submitted and the session ended by the editor "
                        f"with {self.passes_left} forward passes left"
                    )
                if reason is None and self.cfg.require_watchdog and not watchdog_healthy():
                    reason = "watchdog_unhealthy: memwatch heartbeat missing or stale"
                if reason is not None:
                    self.st.stop_reason = reason
                    break
                message = self._maybe_handoff(message)
                message = self._turn(message)
                self._save_state()
                if self.st.stop_reason is not None:
                    break
        except BaseException as e:
            self.st.crashed = True
            self.st.stop_reason = f"crash: {type(e).__name__}: {e}"[:500]
            self.log(f"crash:\n{traceback.format_exc()}")
            raise
        finally:
            try:
                self.kernel.close()
            finally:
                self._save_state()
                self.log(f"stopped: {self.st.stop_reason}")
        return self.st

    def _maybe_handoff(self, message: str) -> str:
        st = self.st
        if st.thread_id is None:
            return message
        too_long = st.last_input_tokens >= self.cfg.handoff_tokens
        if st.since_handoff < self.cfg.handoff_turns and not too_long:
            return message
        st.thread_id, st.since_handoff, st.last_input_tokens = None, 0, 0
        st.handoffs += 1
        self.log(f"handoff {st.handoffs} at turn {st.turn + 1}")
        return self._handoff_text(message)

    def _notes(self) -> tuple[str, str]:
        try:
            raw = read_regular(self.paths.work, "notes.md", NOTES_CHARS * 4, tail=True)
        except FileNotFoundError:
            return "", ""
        except IntegrityError as e:
            return "", self.violation("work/notes.md", str(e)) + "\n"
        return raw.decode("utf-8", errors="replace")[-NOTES_CHARS:], ""

    def _handoff_text(self, message: str) -> str:
        st = self.st
        notes, warning = self._notes()
        files = sorted(p.name for p in self.paths.work.iterdir())
        latest = "none"
        if st.latest_edit is not None:
            man = json.loads(
                (self.paths.edits / f"edit_{st.latest_edit}" / "manifest.json").read_text()
            )
            latest = (
                f"edit {st.latest_edit} (tensors {sorted(man['validation']['tensors'])}, "
                f"added heads {man['validation']['added_heads']}, "
                f"added neurons {man['validation']['added_neurons']})"
            )
        final = f"edit {st.final_edit}" if st.final_edit is not None else "none yet"
        return (
            f"{self.task}\n\nTHIS IS A RESTARTED CONVERSATION (to keep it short): "
            f"{self.guard.replies} replies happened before. Forward passes used: {st.passes} of "
            f"{self.cfg.forward_budget}; Python runs used: {st.python_runs} of "
            f"{self.cfg.python_runs}. Your kernel is still running with its variables. Latest "
            f"saved edit: {latest}. Submitted final edit: {final}. Replies left: "
            f"{self.replies_left}. Files in work/: {files} (work/cells.py holds the code of every "
            f"Python run so far).\n{warning}YOUR NOTEBOOK (work/notes.md, last 8,000 characters):\n"
            f"```\n{notes}\n```\nContinue from here.\n\nLast result:\n{message}"
        )

    def _account(self, res: TurnResult) -> Usage | None:
        # Codex reports each thread's running total: charge the rise since that thread's last report,
        # whether or not the turn succeeded.
        if res.usage is None:
            return None
        tid = res.thread_id
        prev = self.st.thread_totals.get(tid, Usage()) if tid is not None else Usage()
        if tid is not None:
            self.st.thread_totals[tid] = res.usage
        return res.usage.since(prev)

    def _adopt_thread(self, res: TurnResult, usage: Usage | None) -> None:
        st = self.st
        if res.thread_id is not None and res.thread_id != st.thread_id:
            st.thread_id = res.thread_id
            st.last_input_tokens = 0
        if usage is not None and res.thread_id == st.thread_id:
            st.last_input_tokens = usage.input_tokens

    def _call_editor(self, message: str) -> TurnResult:
        try:
            return self.editor.turn(
                message, self.st.thread_id, self.paths.turns / f"t{self.st.turn:03d}"
            )
        except Exception as e:  # noqa: BLE001 - tool boundary: report, never crash the run
            self.log(f"turn {self.st.turn}: editor call raised:\n{traceback.format_exc()}")
            return failed_result([f"editor launch error: {type(e).__name__}: {e}"])

    def _turn(self, message: str) -> str:
        st = self.st
        st.turn += 1
        new_thread = st.thread_id is None
        res = self._call_editor(message)
        usage = self._account(res)
        replied = res.ok and not res.tool_items
        self.guard.record_turn(usage, res.rate, replied=replied)
        row = {
            "turn": st.turn, "reply": self.guard.replies if replied else None, "time": time.time(),
            "thread": res.thread_id, "new_thread": new_thread, "ok": res.ok, "wall_s": res.wall_s,
            "usage": asdict(usage) if usage else None,
            "rate": asdict(res.rate) if res.rate else None, "points_used": self.guard.points_used,
            "input_tokens_total": self.guard.usage.input_tokens, "tool_items": res.tool_items,
            "item_types": res.item_types, "errors": res.errors[:3],
        }  # fmt: skip
        self._append("usage.jsonl", {k: row[k] for k in
                     ("turn", "time", "usage", "rate", "points_used", "input_tokens_total")})  # fmt: skip
        if res.tool_items:
            st.stop_reason = f"isolation: Codex reported tool items {res.tool_items}"
            self._append("turns.jsonl", row | {"tool": None})
            return message
        if not res.ok:
            self._append("turns.jsonl", row | {"tool": None})
            return self._failed_turn(res, usage, message)
        st.failures = st.throttles = st.refusals = 0
        self._adopt_thread(res, usage)
        st.since_handoff += 1
        parsed = parse_reply(res.text)
        tool = parsed.block.tool if parsed.block else None
        self._append("turns.jsonl", row | {"tool": tool, "blocks": parsed.n_blocks})
        if parsed.block is None:
            return self._no_block(parsed.unterminated)
        st.nudges = 0
        t0 = time.time()
        reply = self._dispatch(parsed.block.tool, parsed.block.body)
        self.log(
            f"turn {st.turn} (reply {self.guard.replies}): {tool} ({time.time() - t0:.1f} s); "
            f"runs {st.python_runs}, passes {st.passes}/{self.cfg.forward_budget}, "
            f"points {self.guard.points_used:g}, input {self.guard.usage.input_tokens:,}"
        )
        if parsed.n_blocks > 1:
            reply += "\n\n(Your reply had more than one block; only the first was run.)"
        return reply

    def _dispatch(self, tool: str, body: str) -> str:
        handler = {"python": self._python, "forward": self._forward, "save_edit": self._save_edit}
        try:
            return handler[tool](body)
        except IntegrityError as e:
            return self.violation(tool, str(e))
        except Exception as e:  # noqa: BLE001 - tool boundary: report, never crash the run
            self.st.tool_errors += 1
            self._append("tool_calls.jsonl", {"turn": self.st.turn, "tool": tool, "ok": False,
                                              "error": f"{type(e).__name__}: {e}", "internal": True,
                                              "traceback": traceback.format_exc()[-4000:]})  # fmt: skip
            self.log(f"turn {self.st.turn}: {tool} tool error {type(e).__name__}: {e}")
            return (
                f"The {tool} tool failed and was not counted: {type(e).__name__}: {e}\n"
                f"{self._budget_line()}"
            )

    def _failed_turn(self, res: TurnResult, usage: Usage | None, message: str) -> str:
        st = self.st
        if res.refused:
            st.refusals += 1
            self.log(f"turn {st.turn}: refused by the content filter ({st.refusals} in a row)")
            if st.refusals >= MAX_REFUSALS:
                st.stop_reason = f"content filter refused {st.refusals} turns in a row"
            self._adopt_thread(res, usage)
            return REFUSAL_NOTE if st.thread_id else f"{REFUSAL_NOTE}\n\n{message}"
        if res.limit_hit:
            st.stop_reason = f"codex usage limit: {res.errors[:1]}"
            return message
        if res.transient:
            st.throttles += 1
            wait = throttle_wait_s(st.throttles)
            if wait is None:
                st.stop_reason = f"codex throttled or offline {st.throttles} times in a row"
            else:
                self.log(f"turn {st.turn}: codex throttled or offline, waiting {wait / 60:.0f} min")
                time.sleep(wait)
            return message
        st.failures += 1
        self.log(f"turn {st.turn}: codex failure rc {res.rc}: {res.errors[:2]}")
        if st.failures >= MAX_FAILURES:
            st.stop_reason = f"codex failed {st.failures} times in a row"
        else:
            time.sleep(30 * st.failures)
        return message

    def _no_block(self, unterminated: bool) -> str:
        st = self.st
        st.nudges += 1
        if st.nudges >= MAX_NUDGES:
            st.stop_reason = f"{st.nudges} replies in a row without a block"
        why = "a block was opened but never closed" if unterminated else "it had no block"
        status = self._budget_line()
        if self.passes_left <= 0 and st.final_edit is None:
            status += (
                ' Submit your final edit with a ```save_edit block containing {"final": true}.'
            )
        return (
            f"Your reply was not run: {why}. Reply with exactly one ```python, ```forward or "
            f"```save_edit block. {status}"
        )

    def _budget_line(self) -> str:
        return f"Forward passes left: {self.passes_left}. Replies left: {self.replies_left}."

    def _kernel_integrity(self) -> list[dict[str, object]]:
        # structured rule-integrity violations the kernel reports with each reply (Kernel.last_integrity)
        got = getattr(self.kernel, "last_integrity", None)
        return [v for v in got if isinstance(v, dict)] if isinstance(got, list) else []

    def _python(self, code: str) -> str:
        st = self.st
        if st.python_runs >= self.cfg.python_runs:
            return f"No Python runs left (limit {self.cfg.python_runs}). Use forward or save_edit."
        k = st.python_runs + 1
        t0 = time.time()
        # driver-private record of the code before the kernel sees it, and its full output after
        self._append("exec.jsonl", {"event": "start", "turn": st.turn, "index": k, "time": t0,
                                    "code": code, "code_sha256": hashlib.sha256(code.encode()).hexdigest()})  # fmt: skip
        st.python_runs = k
        full, ok, restart, error = "", False, None, None
        try:
            proc = self.kernel.proc
            was_dead = proc is None or proc.poll() is not None
            full, ok, restart = self.kernel.run(code)
        except BaseException as e:
            error = f"{type(e).__name__}: {e}"
            raise
        finally:
            self._append("exec.jsonl", {"event": "done", "turn": st.turn, "index": k,
                                        "time": time.time(), "output": full, "ok": ok,
                                        "restart": restart, "error": error})  # fmt: skip
        out = cut(full, self.cfg.output_chars) if full else "(no output)"
        if restart is not None:
            out = (
                f"[the kernel was stopped ({restart}) and restarted: its variables are lost; "
                f"the predefined helpers are defined again; files in work/ remain]"
            )
        elif was_dead:
            restart = "the kernel had exited since the previous run"
            out = (
                "[the kernel had exited since your previous Python run and was restarted before "
                "this one: variables from earlier runs are lost; the predefined helpers are "
                f"defined again; files in work/ remain]\n{out}"
            )
        integrity = self._kernel_integrity()
        for v in integrity:
            self.st.integrity_violations += 1
            self._append("integrity.jsonl", {"turn": st.turn, "time": time.time(), "source": "kernel",
                                             "where": f"python run {k}", "check": v.get("check"),
                                             "detail": str(v.get("detail"))[:500]})  # fmt: skip
            self.log(f"turn {st.turn}: kernel integrity violation {v}")
        notes = []
        if integrity:
            checks = "; ".join(f"{v.get('check')}: {v.get('detail')}" for v in integrity)
            notes.append(
                f"[harness integrity check after this run: {checks}. This breaks the kernel rules "
                "and was logged as an integrity violation.]"
            )
        try:
            append_regular(
                self.paths.work,
                "cells.py",
                f"\n# ---- python run {k} (reply {self.guard.replies}) ----\n{code}\n",
            )
        except Exception as e:  # noqa: BLE001 - cells.py is a best-effort copy for the editor
            notes.append(self.violation("work/cells.py", f"{type(e).__name__}: {e}"))
        self._append("tool_calls.jsonl", {
            "turn": st.turn, "tool": "python", "index": k, "code": code,
            "output": full, "ok": ok, "restart": restart, "integrity": integrity,
            "s": round(time.time() - t0, 2),
        })  # fmt: skip
        extra = ("\n" + "\n".join(notes)) if notes else ""
        return (
            f"OUTPUT of Python run {k} of at most {self.cfg.python_runs}:\n```\n{out}\n```"
            f"{extra}\n{self._budget_line()}"
        )

    def _forward(self, body: str) -> str:
        st = self.st
        k = st.passes + 1
        t0 = time.time()

        def charge() -> None:  # inference starts: the pass is used whatever happens next
            st.passes = k

        try:
            if self.passes_left <= 0:
                raise ForwardError("no forward passes left")
            req = parse_json_body(body)
            summary = self.fwd.run(req, k, charge)
        except Exception as e:  # noqa: BLE001 - tool boundary: report, never crash the run
            counted = st.passes == k
            self._append("tool_calls.jsonl", {"turn": st.turn, "tool": "forward", "request": body,
                                              "index": k if counted else None, "counted": counted,
                                              "ok": False, "error": f"{type(e).__name__}: {e}"})  # fmt: skip
            note = self.violation("forward", str(e)) + "\n" if isinstance(e, IntegrityError) else ""
            if counted:
                return (
                    f"{note}The forward pass failed after inference started; it counted as pass "
                    f"{k} of {self.cfg.forward_budget}: {type(e).__name__}: {e}\n"
                    f"{self._budget_line()}"
                )
            if isinstance(e, IntegrityError):
                return note.strip()
            if self.passes_left <= 0:
                return (
                    'No forward passes left. Submit your final edit with save_edit ("final": true).'
                )
            return f"The forward pass failed and was not counted: {type(e).__name__}: {e}"
        summary["s"] = round(time.time() - t0, 1)
        self._append("tool_calls.jsonl", {"turn": st.turn, "tool": "forward", "index": k,
                                          "request": req, "ok": True, "summary": summary})  # fmt: skip
        text = json.dumps(summary, indent=1)
        reply = (
            f"Forward pass {k} of {self.cfg.forward_budget} done ({req['type']}). Summary:\n"
            f"```\n{cut(text, self.cfg.output_chars)}\n```\n{self._budget_line()}"
        )
        if self.passes_left == 0 and st.final_edit is None:
            reply += (
                ' That was the last pass: submit your final edit with save_edit ("final": true).'
            )
        return reply

    def _snapshot_stage(self) -> Path:
        snap = self.paths.private / "stage_snapshot"
        shutil.rmtree(snap, ignore_errors=True)
        try:
            copy_tree_regular(self.paths.work, "edited", snap, STAGED_MAX_FILE, STAGED_MAX_TOTAL)
        except FileNotFoundError:
            snap.mkdir(parents=True, exist_ok=True)
        return snap

    def _save_edit(self, body: str) -> str:
        st = self.st
        try:
            req = parse_json_body(body)
            unknown = set(req) - {"final", "note", "end"}
            if unknown:
                raise EditError(f"unknown keys {sorted(unknown)}")
            final = req.get("final", False)
            end = req.get("end", False)
            if not isinstance(final, bool) or not isinstance(end, bool):
                raise EditError('"final" and "end" must be true or false')
            if end and not self.cfg.allow_end:
                raise EditError('"end" is not available in this session')
            if end and not final:
                raise EditError('"end": true needs "final": true')
            # validate a driver-private copy, so the kernel cannot swap files mid-read
            edit = read_staged(self._snapshot_stage(), self.base)
            report = validate(edit, self.base, self.cfg.condition, self.cfg.max_added_params)
            changed = report["tensors"]
            assert isinstance(changed, dict)
            if not changed and edit.arch == architecture(self.base):
                raise EditError("the staged edit changes nothing")
        except IntegrityError:
            raise
        except Exception as e:  # noqa: BLE001 - tool boundary: report, never crash the run
            self._append("tool_calls.jsonl", {"turn": st.turn, "tool": "save_edit", "request": body,
                                              "ok": False, "error": f"{type(e).__name__}: {e}"})  # fmt: skip
            return f"save_edit refused the staged edit; nothing was stored: {type(e).__name__}: {e}"
        edit.tensors = {k: v for k, v in edit.tensors.items() if k in changed}
        k = st.edits_saved + 1
        edir = self.paths.edits / f"edit_{k}"
        save_edit(edit, edir / "edit.pt")
        manifest = {"edit": k, "turn": st.turn, "final": final, "note": str(req.get("note", "")),
                    "validation": report}  # fmt: skip
        (edir / "manifest.json").write_text(json.dumps(manifest, indent=1))
        self.fwd.edited = apply_edit(self.base, edit)
        st.edits_saved = st.latest_edit = k
        if final:
            st.final_edit = k
        if end:
            st.ended_by_editor = True
        self._append("tool_calls.jsonl", {"turn": st.turn, "tool": "save_edit", "index": k,
                                          "ok": True, "manifest": manifest})  # fmt: skip
        reply = (
            f"save_edit: stored as edit {k}"
            + (" and submitted as your FINAL edit" if final else "")
            + '. It is now the "edited" model for forward passes. Validation:\n'
            + f"```\n{cut(json.dumps(report, indent=1), self.cfg.output_chars)}\n```"
        )
        if end:
            reply += "\nThe session ends now, as you asked."
        elif final and self.passes_left > 0:
            reply += (
                f"\nYou still have {self.passes_left} forward passes: the session continues until "
                "they are used. You may submit again; the last submitted edit counts."
            )
        return reply


def _try[T](errors: dict[str, str], key: str, fn: Callable[[], T]) -> T | None:
    try:
        return fn()
    except Exception:  # noqa: BLE001 - finalize must still write result.json
        errors[key] = traceback.format_exc()[-4000:]
        return None


def _read_jsonl(path: Path) -> tuple[list[dict[str, object]], int]:
    # valid records are kept; a line cut short by a kill is skipped and counted
    rows: list[dict[str, object]] = []
    corrupt = 0
    if not path.exists():
        return rows, corrupt
    for line in path.read_bytes().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            corrupt += 1
            continue
        if isinstance(row, dict):
            rows.append(row)
        else:
            corrupt += 1
    return rows, corrupt


def finalize(
    cfg: RunConfig,
    paths: RunPaths,
    st: RunState,
    n_per: int = 200,
    mirror_root: Path | None = None,
    base_cache_dir: Path | None = None,
) -> dict[str, object]:
    # Every step is guarded on its own: a failing step still leaves result.json and the mirror.
    # compliance is always pending_review; only experiments/phase3/mark_reviewed.py changes it.
    errors: dict[str, str] = {}
    ev = cfg.fam.evaluator()
    mirror_root = mirror_root or cfg.fam.runs_dir
    cache_dir = base_cache_dir or cfg.fam.base_cache_dir
    chosen = st.final_edit if st.final_edit is not None else st.latest_edit
    edit_path = paths.edits / f"edit_{chosen}" / "edit.pt" if chosen is not None else None
    base_ckpt = paths.sandbox / "model" / "base.pt"
    hidden = _try(errors, "hidden_eval", lambda: ev.evaluate_edit(base_ckpt, edit_path, n_per))
    base = _try(errors, "base_eval", lambda: ev.base_reference(base_ckpt, cache_dir, n_per))
    key = _try(errors, "eval_key", lambda: ev.eval_key(base_ckpt, n_per))
    audit = _try(errors, "audit", lambda: audit_run(paths.private, cfg.kernel_forward))
    if audit is not None:
        _try(errors, "audit_write", lambda: (paths.private / "audit.json").write_text(
            json.dumps(audit, indent=1)))  # fmt: skip
    state_file = paths.private / "state.json"
    state = _try(errors, "state_load", lambda: json.loads(state_file.read_text())) or {}
    manifest = None
    if chosen is not None:
        manifest = _try(errors, "edit_manifest", lambda: json.loads(
            (paths.edits / f"edit_{chosen}" / "manifest.json").read_text()))  # fmt: skip
    void = (st.stop_reason or "").startswith("isolation:")
    read = _try(errors, "integrity_log", lambda: _read_jsonl(paths.private / "integrity.jsonl"))
    violations, corrupt = read or ([], 0)
    budget = state.get("budget") if isinstance(state, dict) else None
    result: dict[str, object] = {
        "run_id": cfg.run_id,
        "harness_version": HARNESS_VERSION,
        "compliance": "pending_review",
        "review": None,
        "provenance": state.get("provenance"),
        "finalize_provenance": _try(errors, "finalize_provenance", provenance),
        "finalized_at": time.time(),
        "config": asdict(cfg),
        "stop_reason": st.stop_reason,
        "crashed": st.crashed,
        "void": void,
        # an isolation breach or a crash is never a scored result
        "scored": not void and not st.crashed and hidden is not None,
        "audit_ok": audit is not None and audit.get("violations") == 0,
        "budget_verified": isinstance(budget, dict) and budget.get("budget_verified") is True,
        "integrity_count": len(violations) + corrupt,
        "evaluated_edit": chosen,
        "edit_was_final": st.final_edit is not None,
        "counters": asdict(st),
        "budget": budget,
        "edit_manifest": manifest,
        "hidden_eval": hidden,
        "base_eval": base,
        "eval_key": key,
        "integrity": {
            "violations": len(violations),
            "corrupt_lines": corrupt,
            "rows": violations[:200],
        },
        "audit": None
        if audit is None
        else {k: v for k, v in audit.items() if k != "findings"}
        | {"findings": audit["findings"][:200]},
        "errors": errors,
    }
    _try(errors, "result_write", lambda: (paths.private / "result.json").write_text(
        json.dumps(result, indent=1)))  # fmt: skip
    problems = _try(errors, "mirror", lambda: mirror(paths, mirror_root / cfg.run_id)) or []
    for where, detail in problems:
        _try(errors, "mirror_violation", lambda w=where, d=detail: _append_violation(paths, w, d))
    if problems:  # fold mirror-time findings into the published result
        result["integrity_count"] = int(str(result["integrity_count"])) + len(problems)
        data = json.dumps(result, indent=1)
        _try(errors, "result_rewrite", lambda: (paths.private / "result.json").write_text(data))
        _try(
            errors,
            "mirror_result",
            lambda: (mirror_root / cfg.run_id / "result.json").write_text(data),
        )
    return result


def _append_violation(paths: RunPaths, where: str, detail: str) -> None:
    with (paths.private / "integrity.jsonl").open("a") as f:
        f.write(json.dumps({"turn": None, "time": time.time(), "source": "mirror",
                            "where": where, "detail": detail}) + "\n")  # fmt: skip


MIRROR_SUFFIXES = {".json", ".jsonl", ".txt", ".md", ".log", ".sb"}


def mirror(paths: RunPaths, dest: Path) -> list[tuple[str, str]]:
    # private/ is driver-owned except kernel_tmp (skipped); editor files are read no-follow.
    dest.mkdir(parents=True, exist_ok=True)
    problems: list[tuple[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(paths.private, followlinks=False):
        here = Path(dirpath)
        rel_dir = here.relative_to(paths.private)
        if rel_dir == Path("."):
            dirnames[:] = [d for d in dirnames if d != "kernel_tmp"]
        for name in sorted(filenames):
            src = here / name
            if src.is_symlink() or src.suffix not in MIRROR_SUFFIXES:
                continue
            target = dest / rel_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target, follow_symlinks=False)
    for root, rel in ((paths.sandbox, "TASK.md"), (paths.work, "notes.md")):
        try:
            data = read_regular(root, rel, 16 * 2**20, tail=True)
        except FileNotFoundError:
            continue
        except IntegrityError as e:
            problems.append((f"mirror {rel}", str(e)))
            continue
        (dest / Path(rel).name).write_bytes(data)
    return problems
