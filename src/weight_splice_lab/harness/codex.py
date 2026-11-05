import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import psutil

from weight_splice_lab.harness.budget import (
    RateReading,
    Usage,
    find_rollout,
    reading_from_events,
    reading_from_rollout,
    usage_from_event,
)

SWITCH_CONFIG = Path.home() / ".codex-switch" / "config.json"

# Every Codex feature that can hand the model a tool, or inject profile context, is switched off.
DISABLED_FEATURES = (
    "apps", "plugins", "remote_plugin", "recommended_plugins", "multi_agent", "multi_agent_v2",
    "code_mode", "code_mode_host", "collaboration_modes", "goals", "hooks", "shell_tool",
    "unified_exec", "unified_exec_tty", "shell_snapshot", "view_image", "image_generation",
    "browser_use", "browser_use_external", "browser_use_full_cdp_access", "in_app_browser",
    "computer_use", "sleep_tool", "tool_suggest", "skill_search", "skill_mcp_dependency_install",
    "worktrees", "workspace_dependencies", "memories", "realtime_conversation",
    "request_permissions_tool", "standalone_web_search",
)  # fmt: skip

ISOLATION_OVERRIDES = (
    'sandbox_mode="read-only"',
    'approval_policy="never"',
    'web_search="disabled"',
    "project_doc_max_bytes=0",
    "include_permissions_instructions=false",
    "include_environment_context=false",
    "include_apps_instructions=false",
    "include_collaboration_mode_instructions=false",
    "skills.include_instructions=false",
    "features.skip_host_skill_discovery=true",
    "agents.enabled=false",
    "tools.update_plan.enabled=false",
    "tools.experimental_request_user_input.enabled=false",
    'history.persistence="none"',
)

# Items a tool-free turn may contain. "error" carries Codex's notice that code mode is disabled.
TOOL_FREE_ITEMS = frozenset({"agent_message", "reasoning", "error"})


@dataclass
class CodexSettings:
    instructions_file: Path
    profile: str = "personal"
    model: str = "gpt-6-astra"
    effort: str = "high"
    timeout_s: float = 3600.0
    extra_overrides: tuple[str, ...] = ()


def codex_home(profile: str) -> Path:
    data = json.loads(SWITCH_CONFIG.read_text())
    return Path(data["profiles"][profile]["codex_home"])


def codex_env() -> dict[str, str]:
    # Inside a cmux terminal a `codex` shim prepends cmux's hooks (--enable hooks,
    # --dangerously-bypass-hook-trust); opt out and keep the shims off PATH.
    env = dict(os.environ)
    env["CMUX_CODEX_HOOKS_DISABLED"] = "1"
    env["PATH"] = os.pathsep.join(
        p for p in env.get("PATH", "").split(os.pathsep) if "cmux-cli-shims" not in p
    )
    return env


def build_command(s: CodexSettings, thread_id: str | None, last_message: Path) -> list[str]:
    cmd = ["codex-switch", "run", s.profile, "--", "codex", "--no-daemon", "exec"]
    if thread_id is not None:
        cmd.append("resume")
    cmd += ["--strict-config", "--ignore-user-config", "--ignore-rules", "--skip-git-repo-check"]
    cmd += ["--json", "-m", s.model, "-o", str(last_message)]
    overrides = [
        f'model_reasoning_effort="{s.effort}"',
        f'model_instructions_file="{s.instructions_file}"',
        *ISOLATION_OVERRIDES,
        *(f"features.{f}=false" for f in DISABLED_FEATURES),
        *s.extra_overrides,
    ]
    for o in overrides:
        cmd += ["-c", o]
    if thread_id is not None:
        cmd.append(thread_id)
    cmd.append("-")
    return cmd


@dataclass
class EventAudit:
    thread_id: str | None = None
    text: str = ""
    usage: Usage | None = None
    item_types: dict[str, int] = field(default_factory=dict)
    tool_items: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    turn_completed: bool = False

    @property
    def tool_free(self) -> bool:
        return not self.tool_items


def audit_events(events: list[dict[str, object]]) -> EventAudit:
    a = EventAudit()
    messages: list[str] = []
    for e in events:
        kind = e.get("type")
        if kind == "thread.started":
            a.thread_id = str(e.get("thread_id"))
        elif kind == "turn.completed":
            a.turn_completed = True
            a.usage = usage_from_event(e)
        elif kind in ("turn.failed", "error"):
            a.errors.append(json.dumps(e)[:2000])
        elif kind in ("item.started", "item.updated", "item.completed"):
            item = e.get("item")
            if not isinstance(item, dict):
                continue
            itype = str(item.get("type"))
            if kind == "item.completed":
                a.item_types[itype] = a.item_types.get(itype, 0) + 1
            if itype not in TOOL_FREE_ITEMS and itype not in a.tool_items:
                a.tool_items.append(itype)
            if kind == "item.completed" and itype == "agent_message":
                messages.append(str(item.get("text", "")))
    # every agent message of the turn, in order: a block in an earlier message is not lost
    a.text = "\n\n".join(messages)
    return a


def parse_jsonl(text: str) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            events.append(obj)
    return events


@dataclass
class TurnResult:
    ok: bool
    text: str
    thread_id: str | None
    usage: Usage | None
    rate: RateReading | None
    tool_items: list[str]
    item_types: dict[str, int]
    errors: list[str]
    wall_s: float
    refused: bool = False
    limit_hit: bool = False
    rc: int = 0
    transient: bool = False


REFUSAL_MARKERS = ("usage policy", "flagged")
# only explicit exhausted-quota wording stops a run; a 429 or "rate limit reached" is retried
QUOTA_MARKERS = (
    "usage limit", "hit your usage limit", "quota exceeded", "exceeded your quota",
    "insufficient_quota", "purchase more credits",
)  # fmt: skip
TRANSIENT_MARKERS = (
    "429", "too many requests", "rate limit", "timed out", "timeout", "reconnecting",
    "error sending request", "dns error", "failed to lookup address", "connection refused",
    "connection reset", "stream disconnected", "active writer", "service unavailable",
    "502", "503", "504", "overloaded",
)  # fmt: skip


@dataclass(frozen=True)
class ErrorKind:
    refused: bool
    quota: bool
    transient: bool


def _messages(obj: object) -> list[str]:
    if isinstance(obj, dict):
        out = [v for k, v in obj.items() if k == "message" and isinstance(v, str)]
        for k, v in obj.items():
            if k != "message":
                out += _messages(v)
        return out
    if isinstance(obj, list):
        return [m for v in obj for m in _messages(v)]
    return []


def error_messages(errors: list[str]) -> list[str]:
    # JSON events are classified on their message strings only, never on key names
    out: list[str] = []
    for e in errors:
        try:
            obj = json.loads(e)
        except ValueError:
            out.append(e)
            continue
        out += _messages(obj) if isinstance(obj, (dict, list)) else [e]
    return out


def classify_errors(errors: list[str]) -> ErrorKind:
    text = " ".join(error_messages(errors)).lower()
    refused = any(m in text for m in REFUSAL_MARKERS)
    quota = not refused and any(m in text for m in QUOTA_MARKERS)
    transient = not refused and not quota and any(m in text for m in TRANSIENT_MARKERS)
    return ErrorKind(refused, quota, transient)


def group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def kill_group(proc: "subprocess.Popen[str]", grace_s: float = 5.0) -> None:
    # Kill the whole process group plus any descendant that left it, and wait until all are gone,
    # so no orphaned Codex keeps the thread's writer lock.
    try:
        procs = [psutil.Process(proc.pid), *psutil.Process(proc.pid).children(recursive=True)]
    except psutil.Error:
        procs = []
    for sig, wait in ((signal.SIGTERM, grace_s), (signal.SIGKILL, grace_s)):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass
        for q in procs:
            try:
                q.send_signal(sig)
            except psutil.Error:
                pass
        _, alive = psutil.wait_procs(procs, timeout=wait)
        procs = alive
        # the parent may already be reaped (empty snapshot): KILL still goes to a live group
        if not procs and not group_alive(proc.pid):
            break
    deadline = time.monotonic() + grace_s
    while group_alive(proc.pid) and time.monotonic() < deadline:
        time.sleep(0.05)  # until KILLed members are gone


def run_group(
    cmd: list[str], stdin: str, cwd: Path, env: dict[str, str], timeout_s: float
) -> tuple[str, str, int]:
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, cwd=cwd, env=env, start_new_session=True,
    )  # fmt: skip
    try:
        out, err = proc.communicate(stdin, timeout=timeout_s)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        kill_group(proc)
        try:
            out, err = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            out, err = "", ""
        return out or "", (err or "") + f"\ntimed out after {timeout_s} s", -1
    kill_group(proc, grace_s=1.0)  # leftovers of a finished call
    return out, err, rc


class CodexEditor:
    def __init__(self, settings: CodexSettings, cwd: Path) -> None:
        self.settings = settings
        self.cwd = cwd
        self.sessions = codex_home(settings.profile) / "sessions"

    def turn(self, prompt: str, thread_id: str | None, turn_dir: Path) -> TurnResult:
        turn_dir.mkdir(parents=True, exist_ok=True)
        (turn_dir / "prompt.txt").write_text(prompt)
        cmd = build_command(self.settings, thread_id, turn_dir / "answer.md")
        t0 = time.time()
        stdout, stderr, rc = run_group(cmd, prompt, self.cwd, codex_env(), self.settings.timeout_s)
        wall = time.time() - t0
        (turn_dir / "events.jsonl").write_text(stdout)
        (turn_dir / "stderr.log").write_text(stderr[-20000:])
        events = parse_jsonl(stdout)
        audit = audit_events(events)
        tid = audit.thread_id or thread_id
        rate = reading_from_events(events, time.time())
        stale = None
        if rate is None and tid is not None:
            rollout = find_rollout(self.sessions, tid)
            rate = reading_from_rollout(rollout) if rollout is not None else None
            # a rollout reading written before this call started belongs to an earlier turn
            if rate is not None and (rate.observed_at is None or rate.observed_at < t0 - 1.0):
                stale, rate = rate, None
        errors = audit.errors + ([stderr.strip()[-2000:]] if rc != 0 and stderr.strip() else [])
        kind = classify_errors(errors)
        result = TurnResult(
            ok=rc == 0 and audit.turn_completed,
            text=audit.text,
            thread_id=tid,
            usage=audit.usage,
            rate=rate,
            tool_items=audit.tool_items,
            item_types=audit.item_types,
            errors=errors,
            wall_s=round(wall, 2),
            refused=kind.refused,
            limit_hit=kind.quota,
            rc=rc,
            transient=kind.transient,
        )
        run_json = {
            "cmd": cmd,
            "rc": rc,
            "wall_s": result.wall_s,
            "thread_in": thread_id,
            "thread_id": tid,
            "audit": {
                "item_types": audit.item_types,
                "tool_items": audit.tool_items,
                "tool_free": audit.tool_free,
                "usage": audit.usage.__dict__ if audit.usage else None,
                "errors": errors,
                "turn_completed": audit.turn_completed,
            },
            "rate": rate.__dict__ if rate else None,
            "stale_rate": stale.__dict__ if stale else None,
        }
        (turn_dir / "run.json").write_text(json.dumps(run_json, indent=1))
        return result
