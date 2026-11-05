#!/bin/zsh
# Runs editor pilots back to back unattended, all on one pinned commit: waits for idle repo + free
# RAM, runs each pilot, publishes its validated result.json and logs from a separate git worktree
# (so the checkout the pilots run from never changes). It never creates a release: every result is
# pending_review, and experiments/phase3/mark_reviewed.py releases an edit after a review.
# Usage: overnight.sh [run_id:condition ...]   (default: pilot4:add_capacity pilot5:edit_only)
set -u
REPO=${OVERNIGHT_REPO:-"$(cd "$(dirname "$0")/../.." && pwd)"}
cd "$REPO" || exit 1
PY=${OVERNIGHT_PY:-.venv/bin/python}
RUNS=${OVERNIGHT_RUNS_ROOT:-$HOME/wsl-runs}
REMOTE=${OVERNIGHT_REMOTE:-origin}
BRANCH=${OVERNIGHT_BRANCH:-main}
PUB=${OVERNIGHT_PUBLISH_DIR:-"$(mktemp -d)/publish"}
FAMILY=${OVERNIGHT_FAMILY:-toy}
PHASE_DIR=runs/$([ "$FAMILY" = toy ] && echo phase3 || echo phase4)
TRAILER="Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
PIN=""

log() { echo "$(date +%H:%M:%S) $*"; }
die() { log "STOP: $*"; exit 1; }

code_dirty() { [ -n "$(git status --porcelain -- . ':(exclude)runs')" ]; }

pin() {
  code_dirty && die "the code tree is dirty; commit first"
  PIN=$(git rev-parse HEAD) || die "no HEAD"
  git fetch -q "$REMOTE" "$BRANCH" || die "git fetch failed"
  git worktree add -q --detach "$PUB" "$REMOTE/$BRANCH" || die "git worktree add failed"
  log "pinned $PIN for the whole chain; publishing from $PUB"
}

verify_pin() {
  [ "$(git rev-parse HEAD)" = "$PIN" ] || die "HEAD moved away from $PIN"
  code_dirty && die "the code tree became dirty"
  return 0
}

repo_busy() {
  ps -Ao command | grep "weight-splice-lab/.venv" | grep -v -E "memwatch|codex_usage|grep" | grep -q .
}

# Seconds since any session file of the editor's Codex profile was written (other use of that
# account counts against a run's weekly-point cap).
codex_idle_s() {
  $PY - "${OVERNIGHT_CODEX_SESSIONS:-$HOME/.codex-switch/profiles/personal/sessions}" <<'EOF2'
import sys, time
from pathlib import Path
m = [p.stat().st_mtime for p in Path(sys.argv[1]).glob("*/*/*/*.jsonl")]
print(int(time.time() - max(m)) if m else 10**9)
EOF2
}

wait_ready() {
  local ok=0 not_before=${OVERNIGHT_NOT_BEFORE:-} idle_min=${OVERNIGHT_CODEX_IDLE_S:-600}
  if [ -n "$not_before" ]; then
    log "waiting until $not_before"
    while [[ "$(date +%H:%M)" < "$not_before" ]]; do sleep 60; done
  fi
  for i in $(seq 1 1440); do
    local a=$($PY -c "import psutil;print(int(psutil.virtual_memory().available/1e9*10))")
    local idle=$(codex_idle_s)
    if ! repo_busy && [ "$a" -ge 65 ] && [ "$idle" -ge "$idle_min" ]; then ok=$((ok+1)); else ok=0; fi
    [ $ok -ge 6 ] && { log "ready (available ${a}e-1 GB, codex idle ${idle}s)"; return 0; }
    [ $((i % 30)) -eq 0 ] && log "waiting: busy=$(repo_busy && echo yes || echo no) available ${a}e-1 GB codex idle ${idle}s"
    sleep 10
  done
  log "gave up waiting after 4h"; return 1
}

# Succeeds if result.json is complete, pending review and was produced by $PIN.
validate() {
  $PY - "$RUNS/$1" "$PUB/$PHASE_DIR/$1" "$1" "$PIN" <<'EOF'
import json, sys
from pathlib import Path
run, mirror, run_id, pin = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
f = run / "harness" / "result.json"
if not f.exists() or not (mirror / "result.json").exists():
    sys.exit(f"{run_id}: result.json or its mirror is missing")
r = json.loads(f.read_text())
if json.loads((mirror / "result.json").read_text()) != r:
    sys.exit(f"{run_id}: mirrored result.json differs")
prov = r.get("provenance") or {}
problems = [k for k, ok in (
    ("run_id", r.get("run_id") == run_id), ("harness_version", r.get("harness_version") == "v4"),
    ("compliance", r.get("compliance") == "pending_review"),
    ("git_sha", prov.get("git_sha") == pin), ("dirty", prov.get("dirty") is False),
    ("finalized_at", isinstance(r.get("finalized_at"), (int, float))),
    ("scored", isinstance(r.get("scored"), bool)),
) if not ok]
if problems:
    sys.exit(f"{run_id}: result.json fails checks {problems}")
EOF
}

publish() {
  local id=$1 dir="$PHASE_DIR/$1" sha pushed=0
  validate "$id" || { log "$id: not published (invalid result)"; return 1; }
  git -C "$PUB" fetch -q "$REMOTE" "$BRANCH" || die "git fetch failed"
  git -C "$PUB" checkout -q --detach "$REMOTE/$BRANCH" || die "worktree checkout failed"
  git -C "$PUB" add -- "$dir" || die "git add failed"
  git -C "$PUB" commit -q -m "feat: add $id run logs and hidden eval results" -m "$TRAILER" -- "$dir" \
    || die "git commit failed"
  for t in 1 2 3; do
    if git -C "$PUB" push -q "$REMOTE" "HEAD:$BRANCH"; then pushed=1; break; fi
    sleep "${OVERNIGHT_PUSH_WAIT:-30}"
    git -C "$PUB" fetch -q "$REMOTE" "$BRANCH" || die "git fetch failed"
    git -C "$PUB" rebase -q "$REMOTE/$BRANCH" || { git -C "$PUB" rebase --abort; die "git rebase failed"; }
  done
  [ $pushed = 1 ] || die "git push failed for $id"
  sha=$(git -C "$PUB" rev-parse HEAD)
  log "pushed $id as $sha (run code $PIN); pending review, no release"
}

run_pilot() {
  local id=$1 cond=$2 rc
  verify_pin
  # an older run dir would let a failed launch publish a stale result
  [ -e "$RUNS/$id" ] && die "$RUNS/$id already exists"
  wait_ready || die "the machine never became ready for $id"
  log "launching $id ($FAMILY, $cond) on $PIN"
  uv run --no-sync python experiments/phase3/run_editor.py --run-id "$id" --family "$FAMILY" --condition "$cond" \
    --forward-budget 50 --effort high --max-limit-points "${OVERNIGHT_MAX_POINTS:-5}" --expect-sha "$PIN" \
    --runs-root "$RUNS" --mirror-root "$PUB/$PHASE_DIR"
  rc=$?
  log "$id exited rc=$rc"
  verify_pin
  publish "$id" || die "$id has no valid result.json"
  [ $rc = 0 ] || die "$id: the editor exited with rc=$rc; logs published, chain stopped"
}

[[ "${OVERNIGHT_LIB:-0}" = 1 ]] && return 0

pin
pilots=("$@")
[ ${#pilots[@]} -eq 0 ] && pilots=(pilot4:add_capacity pilot5:edit_only)
first=1
for p in "${pilots[@]}"; do
  [ $first = 1 ] || { log "cooldown 15 min"; sleep 900; }
  first=0
  run_pilot "${p%%:*}" "${p##*:}"
done
git worktree remove "$PUB" 2>/dev/null || log "kept the publish worktree $PUB"
log "overnight chain done on $PIN; the checkout is still at $PIN (pull when nothing runs)"
