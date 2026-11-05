# Phase 3: editor harness for the toy addition splice

An LLM editor (`gpt-6-astra` through the Codex CLI, personal ChatGPT subscription) is given the
Phase 2 toy model (`runs/phase2/base.pt`) and asked to add multi-digit addition by writing weights
directly, without gradients, while leaving text and the native digit tasks intact. The harness
mirrors Takehi et al.'s (spec in `docs/phase1-repro.md`, "Harness spec"), adapted to the toy.

Code: `src/weight_splice_lab/harness/`, scripts in `experiments/phase3/`, tests in `tests/harness/`,
task prompt in `prompts/toy_addition.md`, run artefacts in `runs/phase3/<run_id>/`.

| Module | What |
|---|---|
| `rundir.py` | builds the run directory, practice data, renders the task prompt; `RunConfig` |
| `codex.py` | the Codex command and isolation overrides, event audit, one turn (`CodexEditor`) |
| `kernel.py`, `kernel_server.py` | the sandboxed persistent Python kernel and its prelude (helpers, guards) |
| `forward.py` | forward-pass tool: greedy generation and teacher-forced reads, on CPU |
| `edit.py` | staged-edit reading, `save_edit` validation, applying an edit to the base |
| `budget.py` | usage and rate-limit parsing, `BudgetGuard` |
| `driver.py` | the turn loop, handoffs, tool dispatch, logs, `finalize` (eval, audit, mirror) |
| `evaluate.py` | hidden evaluation (reuses `toy/metrics.py`) |
| `audit.py` | rule-violation scan of the logs |
| `fake.py` | `ScriptedEditor`, a Codex stand-in for tests and the dry run |

## Run directory and isolation

Each run lives outside the repo, in `~/wsl-runs/<run_id>/`:

```
sandbox/                 what the editor can see (kernel cwd, Codex cwd)
  TASK.md                the rendered task prompt
  model/base.pt          copy of runs/phase2/base.pt
  model/model.py         copy of toy/model.py (incl. add_heads / add_neurons)
  model/tokenizer.py     CharTokenizer source only
  model/config.json      model cfg, condition, added-parameter cap
  practice/addition.jsonl  1,000 problems, 50 per N = 1..20 (seed 20261105)
  practice/native.jsonl    720 copy/reverse/max problems, lengths 1..12
  practice/text.txt        200k chars of wikitext-103 *train* (eval uses validation)
  work/                  the only writable folder; work/edited/ is the staged edit
harness/                 private: turns, logs, stored edits, kernel profile and temp
```

Nothing else from the repo is copied: no `splice.py`, no Phase 2 doc or results.

**Kernel isolation.** The kernel is the repo venv's python run under `sandbox-exec` with a
generated profile (`harness/kernel.sb`): network denied; reads denied under `$HOME`, `/private/tmp`,
`/private/var/folders` and `/Volumes`, except the run's `sandbox/`, its private temp dir, the venv and
the uv Python install; the editable install's `.pth` and `dist-info` (whose METADATA embeds the
README) are denied too; writes only to `sandbox/work/` and the temp dir. It is started with `-I`
and a clean environment. Children (e.g. `subprocess`) inherit the sandbox. Because it runs the venv
python, memwatch still sees it as a repo process.

**Kernel rules** (enforced in the prelude, then audited): grad mode off; `Tensor.backward`,
`torch.autograd.backward/grad`, `torch.enable_grad`, `set_grad_enabled(True)`, `requires_grad_`,
`torch.func.grad/vjp/jvp/jac*/hessian` and every `torch.optim*` module raise
`HarnessRuleError`; `ToyLM/Block/Attention/SwiGLU/RMSNorm.forward`, `ToyLM.generate` and
`F.scaled_dot_product_attention` raise (the model can only be run through the forward tool).
Output is cut at 10k chars; 900 s per call; the driver kills and restarts the kernel above 2 GB RSS
or on timeout and tells the editor. Hand-written numpy re-implementations of the network cannot be
blocked; they are a stated rule and an audit flag.

## The Codex command

Per turn (prompt on stdin; `resume <thread>` from the second turn of each conversation):

```
codex-switch run personal -- codex --no-daemon exec [resume] --strict-config --ignore-user-config \
  --ignore-rules --skip-git-repo-check --json -m gpt-6-astra -o <turn>/answer.md \
  -c model_reasoning_effort="high" -c model_instructions_file="<run>/harness/instructions.md" \
  -c sandbox_mode="read-only" -c approval_policy="never" -c web_search="disabled" \
  -c project_doc_max_bytes=0 -c include_permissions_instructions=false \
  -c include_environment_context=false -c include_apps_instructions=false \
  -c include_collaboration_mode_instructions=false -c skills.include_instructions=false \
  -c features.skip_host_skill_discovery=true -c agents.enabled=false \
  -c tools.update_plan.enabled=false -c tools.experimental_request_user_input.enabled=false \
  -c history.persistence="none" \
  -c features.{apps,plugins,remote_plugin,recommended_plugins,multi_agent,multi_agent_v2,code_mode,
     code_mode_host,collaboration_modes,goals,hooks,shell_tool,unified_exec,unified_exec_tty,
     shell_snapshot,view_image,image_generation,browser_use,browser_use_external,
     browser_use_full_cdp_access,in_app_browser,computer_use,sleep_tool,tool_suggest,skill_search,
     skill_mcp_dependency_install,worktrees,workspace_dependencies,memories,realtime_conversation,
     request_permissions_tool,standalone_web_search}=false  [<thread>] -
```

(the `features.*` list is expanded to one `-c` per flag; `harness/codex.py` holds the canonical list).

- `--ignore-user-config` skips the profile's `config.toml` (its MCP servers, projects, model,
  danger-full-access sandbox); auth still comes from `CODEX_HOME`. `auth.json` and the profile
  config are never copied or modified.
- `--strict-config` makes Codex reject any unknown `-c` key, so every override above is known to
  this Codex version (0.160.0). Checked offline with an invalid `model_provider` (fails before any
  model call): the only error is the provider.
- `instructions.md` is one line, "Answer the user's request.", as in the paper.
- Inside a cmux terminal, `codex` on PATH is a shim that prepends cmux's hooks (`--enable hooks
  --dangerously-bypass-hook-trust -c hooks.*`). The driver sets `CMUX_CODEX_HOOKS_DISABLED=1` (the
  wrapper's opt-out) and drops the shims from PATH (`codex_env()`); `features.hooks=false` also
  comes last. The smoke test below still ran through the shim (it shows the hook-trust notice);
  the fix was added after it.

### Isolation evidence (`runs/phase3/smoke/smoke-20261105-172110.json`)

1. **Model-visible input** (`codex debug prompt-input` with the same overrides; local, no model
   call). Default profile: 5 developer messages (skills list with roots under `~/.agents/skills`,
   permissions "danger-full-access", collaboration mode, recommended plugins, multi-agent role)
   plus the environment context with the cwd. With the overrides: **only the user message**.
2. **Rollout `world_state`** of the smoke thread: `agents_md: {}`, `skills.includeInstructions:
   false`, `host_skills.includeInstructions: false`, `plugins_instructions/apps_instructions/
   environments_instructions: false`, `multi_agent_mode: {}`; `session_meta.base_instructions` =
   "Answer the user's request." (custom); `turn_context`: sandbox read-only, approval never,
   `multi_agent_version: disabled`. The rollout holds only message items.
3. **Call 1** (`low` effort): asked to list its tools, run `ls -la` / `cat secret.txt`, search the
   web, report its cwd and quote its instructions. A random canary was in `secret.txt`. Answer:
   it cannot run shell commands, search the web or see the cwd or files; canary not leaked; event
   stream had no tool items (`agent_message` plus Codex notices only). 1,798 input tokens.
4. **Residual declared tools.** The model listed `functions.exec`, `functions.wait`,
   `functions.request_user_input_async` and nested `tools.apply_patch`, `tools.clock__curr_time`.
   Every model in this Codex catalog has `tool_mode: code_mode_only`, so the code-mode shell is
   always declared; with `code_mode_host=false` it fails closed (Codex's notice: "Code Mode is
   unavailable because code-mode host is disabled. Code mode will fail closed"). The paper's
   logs show the same notice on every turn: this matches their setup. To make an attempt unlikely,
   the task prompt says the blocks are the only way to act and listed built-ins are disabled; if
   any tool item ever appears in a turn's events, the driver stops the run (`isolation: ...`).
5. **Call 2** resumed the thread ("Reply with OK") and checked per-turn rate-limit parsing.

**Rate-limit cost of the two smoke calls: 0 points** (weekly `primary.used_percent` 7.0 before,
7.0 after both calls; the counter has 1-point resolution).

## Protocol and tools

One block per reply, fenced ```` ```python ````, ```` ```forward ```` (JSON) or
```` ```save_edit ```` (JSON). The first block runs; extra blocks are ignored with a note. A reply
without a block gets a nudge (the run stops after 3 in a row). Result messages follow the paper:
`OUTPUT of Python run k of at most 300:` / `Forward pass k of B done (type). Summary: ...
Forward passes left: n.` / `The forward pass failed and was not counted: ...`.

- **python**: the persistent kernel. Prelude: `NAMES`, `shape`, `load` (staged else original, padded
  to added capacity), `load_original`, `set_tensor`, `reset_tensor`, `staged`, `reset_edit`, `arch`,
  `add_heads`, `add_neurons`, `added_params`, `CFG`, `CONDITION`, `tok`, `toy_model`. Helpers write
  the staged edit to `work/edited/` (`arch.json` + `tensors/<name>.pt`). Limit 300 runs.
- **forward** (budget 10/25/50, default 50; failed requests free; any size costs 1), CPU:
  - `generate`: original | edited | both, ≤ 2,048 practice ids and/or custom prompts, greedy,
    `max_new_tokens` ≤ 64. Practice items graded with the hidden-eval rule (addition: the N+1
    answer chars; native: answer + newline). Summary: accuracy overall / by task / by N, 5 failures,
    with "both" rescues / losses / outputs changed. Records in `work/fwd_k*.jsonl`.
  - `teacher_forced`: ≤ 256 texts of ≤ 256 chars; token logprobs always, optional top-k ≤ 20 and
    sites `resid_pre.L`, `resid_mid.L`, `resid_final`, `attn.L.pattern`, `attn.L.head_out`,
    `mlp.L.hidden`, `logits` as `.npy` (≤ 768 MB, checked before running). Bits per char per model;
    with "both" the mean KL(original ‖ edited).
  - "edited" is the latest *saved* edit, not the staged state.
- **save_edit** `{"final": bool, "note": str}`: reads `work/edited/`, validates it, stores it as
  `harness/edits/edit_k/{edit.pt,manifest.json}` and makes it the "edited" model. Validation:
  arch has one value per layer, no shrinking; `edit_only` forbids any added capacity; added
  parameters ≤ cap (default 1.5M; head = 4·256·32, neuron = 3·256); tensor names exist for the arch;
  `embed.weight` (tied unembedding) and `final_norm.weight` refused; exact shapes; finite; an edit
  that only multiplies a whole tensor by a constant refused; an edit that changes nothing refused.
  `add_capacity` allows native edits too (it is a superset of `edit_only`).
- **Handoff**: a fresh conversation every 40 turns, or earlier once a turn's input passes 180k
  tokens: task text + "THIS IS A RESTARTED CONVERSATION" + counters, latest/final edit, `work/`
  listing, last 8,000 chars of `work/notes.md`, then "Last result:". The kernel keeps running.
- Content-filter refusals get a canned note (stop after 3); other Codex failures are retried after
  30 s / 60 s (stop after 3 in a row); a Codex usage-limit error stops the run.

## Budget guards

Checked before every turn; the first that fires ends the run:

1. **Weekly limit**: points used ≥ `--max-limit-points` (default 5). Points = rise of the weekly
   window's `used_percent` above its value at run start. The value comes from the turn's `--json`
   events if they carry `rate_limits` (they don't in 0.160.0) else from the thread's rollout
   (`$CODEX_HOME/sessions/**/rollout-*<thread>.jsonl`, last `token_count`). The start value is the
   latest reading in the newest rollouts before the run (so it can be a little old, which only
   makes the guard stricter), else the first turn's. If the window resets mid-run, the old window's
   rise is banked and the new window counts from 0. `>=` (not `>`) because a turn can overshoot.
   Other personal-account Codex use during a run counts against the run.
2. **Input tokens**: cumulative `input_tokens` (cached included) ≥ `--max-input-tokens` (20M).
3. **Turn cap**: `--max-turns` (200).
4. **Done**: forward budget used up *and* a final edit submitted (in either order; a final edit
   with passes left is recorded, the session continues, a later final replaces it).

Usage is logged per turn in `harness/usage.jsonl` (also in `turns.jsonl` with tool, items, errors).

## Hidden evaluation, audit, artefacts

`finalize` evaluates the final edit (else the latest saved one, else the base), on CPU with
4 threads, never shown to the editor: addition accuracy for N = 1..20 (200 problems per N,
seed 4321, `metrics.target_accuracy`), means for N = 1-8 and 9-20; held-out wikitext validation
bpc and KL(base ‖ edited) on 256 × 256-char chunks (seed 0, as Phase 2); copy/reverse/max
exact match (1,200 each). `meets_bar` = KL < 0.05 and every native task ≥ 0.98 (Phase 2 bar). A
base-model reference is cached in `runs/phase3/base_eval_n<k>.json`. Since 2026-11-05 the eval also
reports a pre-registered robustness block and `meets_strict_bar` (`docs/eval-hardened.md`).

The audit (`audit.py`, also `experiments/phase3/audit_run.py <run_id>`) scans every Python call:
code regexes for autograd, optimizers (incl. `scipy.optimize`), model-execution tricks
(`exec(open(...))`, `importlib`, touching the guards), and outside-run-dir access (absolute paths,
`~`, `os.environ`, `subprocess`, sockets, URLs); outputs for `HarnessRuleError` (a blocked attempt)
and sandbox denials; every turn's Codex events for tool items. Findings are `violation` or `flag`
(needs a human look, e.g. `lr =`, `.forward(`).

`result.json` (config, stop reason, counters, budget trace, edit manifest, hidden eval, base eval,
audit) and the logs are mirrored to `runs/phase3/<run_id>/` (json/jsonl/txt/md/log/sb only;
tensors and `.npy` stay in `~/wsl-runs/`).

## Launch

```bash
uv run python experiments/phase3/dry_run.py            # scripted fake editor, no Codex
uv run python experiments/phase3/run_editor.py --run-id pilot1 \
  --condition add_capacity --forward-budget 50 --effort high --max-limit-points 5
```

`run_editor.py` refuses to start unless memwatch is running, ≥ 6.5 GB RAM is available and kernel
memory pressure is not critical. A run peaks at ~3.5 GB (kernel ≤ 2 GB, CPU forward ~1.2 GB, Codex
~0.3 GB) and memwatch kills every repo process, the driver included, below 2.5 GB available — and a
killed run cannot be resumed.

## Tests and dry run

`uv run pytest` is 86 passed (the full repo suite; 52 of them in `tests/harness/`):
- `test_protocol.py`: block parsing (first block wins, other fences ignored, unterminated, inline
  backticks, JSON bodies, output cut).
- `test_budget.py`: recorded sample events (`tests/harness/data/`: a tool-free turn with the
  code-mode notice, a turn with a `command_execution` item, a content-filter failure, a rollout with
  two `token_count` events): usage, tool-item audit, error classification, rate-limit parsing
  (rollout, events, weekly window preferred), rollout lookup, guard semantics (points, baseline,
  weekly reset, input tokens, turn cap, done), command shape.
- `test_save_edit.py`: allowed/refused tensors, shapes, finite values, constant scaling,
  `edit_only`, the parameter cap, arch checks, an exact no-op for zero added capacity, the
  staged-file layout.
- `test_kernel.py` (real sandboxed kernel): persistence, the last expression shown, errors survived,
  9 gradient/optimizer attempts blocked, model execution blocked (3 ways), reads outside the run
  folder denied (repo README, docs, the repo's parent, `~/.codex-switch`, listing `$HOME`), writes
  outside `work/` denied, network denied, `subprocess` inherits the sandbox, the repo package is not
  importable, kernel staging matches driver validation, timeout restart.
- `test_driver.py`: the loop with a scripted editor (tools, a failed pass that isn't counted, a
  nudge, 2 handoffs, both stop conditions: done, and rate-limit points).

**Dry run** (`experiments/phase3/dry_run.py --run-id dryrun-1`, scripted editor, budget 3, eval
20 problems per N; `runs/phase3/dryrun-1/`): 9 turns (3 python, 3 forward, 2 save_edit, 1 nudge),
stop reason "done: forward budget used and final edit submitted". The final edit (2 zero neurons in
layer 3, +1e-3 on one down_proj column of layer 5) was evaluated: addition 0.15 at N=1 and 0
elsewhere (base level), text KL 4.9e-9, native 1.00/1.00/1.00, `meets_bar` true. The scripted
guard probe (requires_grad_, torch.optim, reading the repo README) was blocked in the kernel and
reported by the audit as 4 violations + 1 flag; all 9 turns tool-free. 43 s wall, peak RSS 1.16 GB.
Base reference at 200 per N (`runs/phase3/base_eval_n200.json`): addition mean N 1-8 0.013,
N 9-20 0.000, 1.703 bpc, native 1.00; 60 s on CPU, 0.7 GB RSS.

## Open issues

1. The code-mode functions stay declared (the catalog's `code_mode_only`); they fail closed, and
   the prompt says not to use them. A third real call would be needed to test a custom
   `model_catalog_json` that drops them; not done, so the 2-call smoke limit is kept.
2. The rate-limit counter has 1-point resolution and is read after each turn, so a run can
   overshoot the 5-point cap by the last turn's cost (a `max`-effort turn costs well under a point).
3. No resume: if the driver dies (e.g. killed by memwatch), the run cannot be continued; the
   Codex thread and `work/` survive, but kernel variables and driver state would need a resume path.
4. Iterative fitting written as plain numpy loops, or a hand-written re-implementation of the
   forward pass, can't be blocked; the audit only flags suspicious patterns, so logs need review.
5. `scipy` and `sklearn` aren't in the venv (the paper's kernel had them); torch/numpy linalg
   cover the closed-form toolkit.

## Deviations from the paper

- CPU-only toy; forward passes take seconds, so "a pass costs 1 whatever its size" matters less.
- Kernel isolation by `sandbox-exec` (the paper used bubblewrap); no RLIMIT_AS on macOS, so memory
  is enforced by RSS polling.
- `save_edit` is a separate tool block (the paper's was a kernel function writing `/work/edited`).
- "EXPLAIN EVERY STEP" is in the prompt from turn 1 and not enforced (paper: from turn 29, enforced).
- Default context window (272k) with an early handoff at 180k input tokens, instead of the
  paper's 872k window override.
- Practice addition problems cover N = 1..20 like the evaluation; for small N they can coincide
  with evaluation problems (N = 1 has only 100 distinct problems).

## Hardening (v3)

Fixes for the two code reviews (`docs/`-level audits) of the kernel, sandbox, staged-edit and
watchdog paths. All CPU-only; the only GPU touch is a tiny must-fail allocation in the sandbox test.

**Sandbox profile (`kernel.py:sandbox_profile`).** Kept `(allow default)` plus the existing read /
write / network rules, and added:
- `(deny file-write-create (vnode-type SYMLINK))` and `(deny file-link)` — the kernel can no longer
  create a symlink or hard link under `work/`, so it cannot redirect the unsandboxed driver's reads
  and writes out of the run dir.
- `(deny process-exec*)` with an allow-list for the venv interpreter only (plus `(deny
  process-fork)`) — no `subprocess`, `os.system`, `/usr/bin/open`, `osascript` or detached helpers.
  Confirmed torch/numpy and all kernel helpers still work.
- `(deny iokit-open …)` for the Metal/IOAccel/IOSurface user clients and `(deny mach-lookup …)` for
  `MTLCompilerService`/`gpumemd` — MPS reports unavailable and a `torch.ones(1, device="mps")` or an
  MLX op fails to get a Metal device. Tested with a 1-element allocation only.

**Kernel rule integrity (`kernel_server.py`, `kernel.py`).** After every cell the server runs
`_check_integrity` and returns a `integrity` list in the reply. Each entry is
`{"check": <name>, "detail": <str>}`; checks are `grad_mode`, `patched_function`, `patched_forward`,
`model_module`, `optim`, `live_grad_tensor`, `non_cpu_tensor`, `integrity_error`. The grad/optim
originals (`set_grad_enabled`, `torch._C._set_grad_enabled`, `Tensor.requires_grad_`) are now hidden
in closures instead of left in the cell namespace, and `torch._C._set_grad_enabled(True)` is blocked,
so the autograd-engine / `grad_fn` bypass cannot build a graph (grad stays globally off). Re-importing
`model`, restoring a patched `forward`/SDPA/optim module, a namespace tensor with `requires_grad` or on
a non-CPU device are all detected and reported. Replies use a dedicated channel: stdout/stderr are
pointed at the capture file for the kernel's whole life and each request carries a nonce, so a stray
background-thread print cannot corrupt a reply. RSS is sampled by a monitor thread every 0.25 s for the
kernel's whole life (including between cells), per process and tolerating vanished children; a breach
kills the kernel and is reported as a `memory limit` restart; `kill` also SIGKILLs any descendant seen
during monitoring.

**Driver interface note.** `Kernel.run` keeps its `(output, ok, restart)` return. The per-cell
integrity list is exposed as `Kernel.last_integrity` (reset at the start of each `run`). The driver
should write it into each `tool_calls.jsonl` python row under the key `integrity`; `audit_run` already
ingests `row["integrity"]` as `kernel_integrity/<check>` violations.

**Staged edits (`edit.py`).** `read_staged` now checks each file is a regular file (no sym/hard
links), bounds its on-disk size, loads with `weights_only=True, mmap=True`, and validates
dtype/shape/numel and that the tensor owns a contiguous storage of exactly its logical size before any
`.float()` conversion — expanded or oversized storages are rejected with `EditError`. Corrupt, empty
and non-tensor payloads raise `EditError`. Names and the refused set are checked before loading.
The whole-tensor scalar-multiple check runs on the native slice and uses a fraction threshold, so it
cannot be bypassed by perturbing a few entries; a zeroed native tensor is treated as a constant
multiple.

**Audit (`audit.py`).** Comments are stripped with `tokenize` (a `#` inside a string no longer
truncates the line), and scanning is per logical line so multi-line calls are caught. Added patterns:
`sys.modules` mutation, `runpy`/`run_path`, `functional_call`, `_C._set_grad`, `_engine_run_backward`/
`_execution_engine`, `grad_fn`, `exec`/`compile` of files, `base64`, `getattr` on torch internals,
`symlink`/`link`, and device `mps`/`cuda`/`mlx`. Removed the two false positives both prior audits hit:
`HarnessRuleError` in a printed variable listing (the output sign now needs the guard's "not allowed"
message) and `__code__` reads on the editor's own helpers (narrowed to writes and to reads of the
guard objects). Integrity violations from the kernel are ingested when present.

**GPU / watchdog (`gpu.py`, `mlx_gen.py`, `memwatch.py`).** The mlx-lm wired-limit patch now targets
`importlib.import_module("mlx_lm.generate")` instead of the `generate` function the package exports.
`watchdog_running` validates a heartbeat file (`runs/.memwatch.heartbeat`: pid, create-time, fresh
timestamp) and the process identity, not just `pid_exists`; a watchdog with only the legacy PID file is
still accepted by identity. memwatch reads the pressure level in-process via `sysctlbyname` (no
subprocess), evaluates each check independently, selects repo processes by venv exe/cwd and process
tree (excluding itself by PID, not the substring "memwatch"), writes the heartbeat each poll, and no
longer unlinks the GPU lock file on kill.

**Running watchdog.** The currently running watchdog predates these changes and writes only
`runs/.memwatch.pid`; the new `gpu.py` accepts it by identity until it is restarted. It needs a restart
to pick up the in-process pressure read, independent checks, process-tree selection and heartbeat — the
coordinator should restart it; this change does not do so.

**Remaining unenforceable.** Arbitrary Python with full PyTorch cannot be made a hard no-training
environment: a hand-written numpy/torch re-implementation of the forward pass, or model classes built
fresh by `runpy.run_path` of `model/model.py` (not via `sys.modules['model']`), are not caught at
runtime — they are reported only by audit flags and require log review.

## Driver hardening (v3)

Fixes to the driver side of the harness after the two 2026-11-06 code reviews
(`docs/reviews/2026-11-06-claude.md`, `docs/reviews/2026-11-06-codex-gpt-6.1-sol.md`). The kernel,
sandbox profile, `edit.py`, audit and memory-safety side was fixed separately. Runs from this
version on record `harness_version: "v3"`. v1/v2 runs (pilot1b-pilot3) are not directly comparable
on the turn cap and the practice grader (see below).

**Editor-controlled paths.** The unsandboxed driver no longer follows links under `sandbox/work/`.
All access goes through `rundir.py` helpers: directory-relative `O_NOFOLLOW` opens component by
component, regular single-link files only (symlinks, hard links, FIFOs and devices are refused, and
a FIFO cannot block the driver), and writes go to a temp file that is renamed into place, so they
never write through a link. This covers `work/notes.md` at handoff, `work/cells.py`, the forward
outputs (`fwd_k*.jsonl`, `fwd_k/<model>/*.npy`, `meta.json`, `kl_per_text.npy`), the staged edit
(copied into `harness/stage_snapshot/` with size caps before `read_staged` sees it, so the kernel
cannot swap files mid-read) and `mirror()` (`os.walk` without following links, `kernel_tmp/`
skipped, `notes.md` read no-follow). A refused path is an **integrity violation**: the editor gets
a `[harness integrity check: ...]` note, the row goes to `harness/integrity.jsonl`, and
`state.integrity_violations` / `result.integrity` count it. The kernel's own per-cell integrity
report (`Kernel.last_integrity`: grad mode on, patched functions restored, `sys.modules['model']`
replaced, optimizers restored, live grad or non-CPU tensors) is recorded the same way per turn
(source `kernel`, and under `integrity` in that cell's `tool_calls.jsonl` row, which
`audit_run` counts as `kernel_integrity/<check>` violations), and a non-empty list is shown to the
editor in that turn's reply.

**Never lose a run.**
- `Driver.run` records any escaping exception as `stop_reason = "crash: <type>: <msg>"`,
  `crashed: true`, closes the kernel and saves `state.json`. `run_editor.py` calls `finalize` in a
  `finally`.
- Every tool call is wrapped: editor-caused failures (empty, truncated or garbage `.pt`, numpy
  arrays saved with `torch.save`, `fwd_k` existing as a file, ...) come back as "failed and was not
  counted" or "save_edit refused"; anything else is reported as a tool error with the traceback in
  `tool_calls.jsonl`. Counters only move after success. An editor launch error (e.g.
  `codex-switch` missing) is a failed turn, not a crash.
- `finalize` guards each step (hidden eval, base eval, audit, manifest). `result.json` and the mirror
  are always written; failures are listed under `errors`.
- `experiments/phase3/finalize_run.py <run_id>` finalizes a killed run (e.g. by memwatch) from its
  `state.json` (stop reason `killed: ...`, `crashed: true`). It refuses while the driver's pid is
  still a running driver, or if `result.json` exists (without `--force`).
- `result.json` has `void` (stop reason `isolation: ...`), `crashed` and
  `scored = not void and not crashed and hidden eval succeeded`. A void or crashed run is never
  released by the overnight chain.

**Codex calls.**
- Errors are classified on the events' `message` strings (and plain stderr text) only, never on
  JSON key names (a `rate_limits` key no longer reads as "rate limit"). Precedence: content-filter
  refusal, then explicit exhausted quota ("usage limit", "hit your usage limit", "quota exceeded",
  `insufficient_quota`, ...), which stops the run, then transient (429, "too many requests",
  "rate limit reached/exceeded", timeouts, network errors, "active writer", 5xx), which backs off
  2, 4, ..., 60 min. A weekly window at ≥ 100% also stops the run.
- Each Codex call runs in its own session/process group (`codex.run_group`). On timeout the whole
  group, plus any descendant that left it, gets SIGTERM and then SIGKILL, and the driver waits until
  all are gone. After a normal exit, leftover group members are killed too. That fixes pilot3's
  "thread ... already has an active writer" after the 3600 s timeout.
- All `agent_message` items of a turn are parsed, in order (a block in an earlier message was lost).
- **Turn cap**: `max_turns` (200) now counts successful editor replies only. Throttled, failed,
  refused and launch-failed calls don't use it. A separate `max_attempts` (260) caps all editor
  calls. Messages show "Replies left" from the reply count.

**Usage and budget.**
- Usage is accounted per actual thread id (`state.thread_totals`). Each report is charged as the
  rise since that thread's last report, on failed turns too, and the snapshot always advances. The
  verified cases: totals 100 / 150 (failed after emitting usage) / 200 now count 200, not 250; a
  refusal on the first turn after a handoff counts 50k, not 35k, and no longer fires a second
  handoff. A thread change resets `last_input_tokens`.
- A rate-limit reading is used only if a window is positively weekly (`window_minutes == 10080`).
  A 5-hour or unlabelled window is ignored.
- A rollout reading whose `timestamp` predates the call is stale and dropped (kept in `run.json` as
  `stale_rate`).
- If `stale_rate_turns` (3) consecutive turns that spent tokens bring no fresh weekly reading, the
  run stops as `budget_unverified`.

**One grader.**
- `toy/metrics.py` has the single rule, `grade_ids`: generated token ids at fixed positions,
  special tokens included. Addition is graded on the N+1 answer tokens; native tasks on answer
  + `\n`. The hidden eval (`answer_match`, `exact_match`) and the practice grader
  (`forward.score`) both use it.
- The case where practice said True and the hidden eval said False (answer `30`, generation
  `PAD 3 0`) is now False in both.
- Practice outputs show special tokens as `<pad>`/`<bos>`/`<unk>`, so characters line up with
  positions. Records also carry `output_ids`.
- The termination probe (b) reads token id N+1 directly; decoding dropped specials and shifted the
  index.
- For the hidden eval the change is a no-op except in (b): a decoded string of k chars equals the
  answer only if no special token was generated, so the old answer and exact-match rules were
  already id-exact.
- Re-verified on `rescore_hardened.json` (CPU, 4 threads, 200/N, peak 1.05 GB RSS): **base and
  pilot1b edit 16 reproduce exactly**. Every primary and robustness value is bit-identical, so no
  previously reported number changes. Neither model emitted a special token in the (b)
  generations, so (b) `terminated` stays 0.00 for both under either indexing. The splice and LoRA
  rows were not rerun. For them, (b) can only rise, never fall: the old decoded index could never
  find a terminator after a special token.

**Forward tool padding.** Every returned array is 0 at padding positions (`t >= length`; token
logprobs also at `length - 1`; attention patterns for padded query rows and key columns), and
`meta.json` says so. Activations, logits and top-k used to hold computed PAD-token values.

**Provenance.**
- `state.json` and `result.json` record `provenance`: `harness_version` "v3", `git_sha`, `dirty`
  (tracked or untracked changes outside `runs/`), `dirty_files`, and `harness_source_sha256` over
  `harness/*.py`, `toy/*.py` and the task prompt. `result.json` also has `finalize_provenance` and
  `eval_key`.
- The base-eval cache is `runs/phase3/base_eval_cache/base_eval_n<k>_<digest>.json`, keyed on the
  base checkpoint's sha256, a hash of the eval code (`evaluate.py`, `edit.py`, `metrics.py`,
  `data.py`, `model.py`), the dataset snapshot, all seeds, thresholds and n. Any change recomputes
  it. The old `base_eval_n*.json` files are no longer read.
- `run_editor.py` refuses a dirty tree unless `--allow-dirty` is given. `--expect-sha` refuses any
  other HEAD.

**Protocol.**
- A fence line inside a triple-quoted string no longer ends a python block. A candidate close is
  skipped while the body so far is an unterminated multi-line string; a genuinely broken body still
  closes at the first fence.
- `py` is accepted as `python`, and a closing fence must be at least as long as its opener.
- Nudges stop the run on the 3rd reply in a row without a block, matching the message.
- Refusals count in a row (reset by a successful reply).
- A kernel that died between cells is now reported: "the kernel had exited since your previous
  Python run ...".

**overnight.sh.**
- The chain pins `HEAD` at start (it refuses a dirty tree) and checks before and after every pilot
  that `HEAD` is unchanged and the tree is clean. It passes `--expect-sha` to `run_editor.py` and
  runs `uv run --no-sync`.
- Results are mirrored into a separate detached worktree at `origin/main`. Publishing fetches and
  checks out the remote tip there, so the pilots' checkout never moves.
- The script validates `result.json` (run id, `harness_version` v3, `git_sha` == pin, not dirty,
  finalized, mirror identical) before publishing. It commits only `runs/phase3/<id>` (`git commit
  -- <dir>`), pushes with fetch + rebase retries and never uses `--autostash`. Any git failure stops
  the chain.
- `gh release create --target <pushed sha>` runs only after a successful push, and only for a
  scored, non-crashed run. The released file is `edits/edit_<evaluated_edit>/edit.pt` from
  `result.json`.
- Pilot ids and conditions are arguments (default `pilot4:add_capacity pilot5:edit_only`).

**Tests** (`uv run pytest`, all CPU, no Codex):
- `test_driver_hardening.py`: links at every editor path, bad staged files, `fwd_k` as a file,
  usage accounting cases, cap semantics, quota vs backoff, unverified budget, nudges, refusals,
  kernel restart notice, kernel integrity rows, launch errors, crash + finalize, void,
  `finalize_run.py`, the dirty-tree refusal.
- `test_rundir_safe.py`: the no-follow helpers and provenance.
- `test_forward_grading.py`: practice/hidden agreement, termination positions, padding masks.
- `test_codex.py`: classification, process-group kill, all agent messages, stale rollout readings.
- `test_budget.py`: weekly-only windows, reply/attempt caps, unverified, 100%.
- `test_evaluate.py`: the keyed cache.
- `test_overnight.py`: publish against a local bare remote with a stub `gh`.
- `test_protocol.py`: fences.

**Dry run (v3, both agents' changes, commit `28b3ed0`)**:
- Setup: scripted editor, forward budget 3, eval 20/N. The runs, mirror and cache roots were in a
  scratch folder (not `~/wsl-runs`, not the repo). The `dry_run.py` script was extended with three
  adversarial cells: replace `work/notes.md` with a symlink, a `notes.md` write containing a fenced
  block inside a string, and `torch._C._set_grad_enabled(True)`.
- Result: 12 replies in 12 attempts, 6 Python runs, 3 passes, 2 edits. Stop "done: forward budget
  used and final edit submitted", `scored: true`, no crash, 0 tool errors, 0 integrity violations.
- The new sandbox refused the symlink (`PermissionError`), so the driver's link checks were not
  reached here; they are covered by tests. The kernel guard blocked the grad call
  (`HarnessRuleError`). The fenced-string cell ran in full.
- Hidden eval at base level: addition N1-8 0.019, KL 4.9e-9, native 1.00/1.00/1.00, `meets_bar`
  true, strict bar false.
- Audit: 9 violations and 3 flags, all from the scripted guard probes; 12/12 turns tool-free.
- 90 s wall, driver peak RSS 1.19 GB (including the eval), kernel 0.23 GB.

**Still open (driver side).**
- The staged-edit snapshot is size-capped, but `torch.load` of a crafted file still runs in the
  unsandboxed driver (the bounded-storage checks are in `edit.py`).
- A resumed driver is still not possible: `finalize_run.py` only finalizes.
- Usage of an orphaned Codex turn that finishes after its timeout is not recorded. The process
  group kill makes that unlikely, but it is not proven for every Codex internal.
- Turn caps and the practice grader changed, so v3 pilots are a new harness version against
  pilot1b-pilot3.

## Hardening (v3.1)

Kernel, audit, sandbox and watchdog side of the 2026-11-06 re-review
(`docs/reviews/2026-11-06-codex-rereview-v3.md`), kept deliberately small. The harness does not
claim to prevent a determined editor from training: grad mode is thread-local and the integrity
checker lives in the editor's interpreter. It claims best-effort blocking, audit detection of the
known bypasses, and review of every logged cell before a result counts.

**Audit (`audit.py`), findings 1 and 2: detection only.**
- New `concurrency` flag: `threading`, `_thread`, `concurrent`, `multiprocessing`. A new thread starts
  with grad mode on, so it can compute gradients the kernel guards never see.
- New `introspection` flag: `inspect.currentframe`, `sys._getframe`, `_current_frames`,
  `f_globals`/`f_back`/`f_locals`, `gc.get_objects`/`get_referrers`/`get_referents`, `ctypes`. These
  can reach and replace the kernel's integrity checker.
- Imports are also read with `ast`, so a parenthesised or multi-line import such as
  `from torch import (\n nn,\n autograd as ag)` is an `autograd` (or `optimizer`) violation, like
  the single-line form.
- The review's worker-thread probe (thread, parenthesised autograd import, `nn.Parameter`, an
  engine name built by string concatenation) is a regression test. It is flagged, not blocked: run
  in the kernel, it still computes the gradient.
- Forward-mode AD (`torch.autograd.forward_ad`) computes derivatives even with grad mode off in
  the kernel's own thread. It is caught only by the `torch.autograd` pattern and the import scan.

**Kernel cleanup (`kernel.py`), finding 8.**
- The process-group ID is retained at spawn (the kernel is its own session leader). It is never
  re-read from a PID.
- The group is signalled only while the kernel is unreaped, so its PID and group ID cannot have
  been reused, and only if its creation time still matches.
- The monitor no longer reaps (no `poll()`). Descendants are tracked as `(pid, create_time)`, and
  each is killed only if both still match.
- The old monitor thread is stopped and joined before a restart replaces the shared state. Each
  monitor has its own stop event.

**Watchdog (`gpu.py`), finding 11.**
- The legacy PID-file acceptance is gone. `watchdog_healthy(heartbeat=HEARTBEAT,
  max_age_s=10.0) -> bool` needs a heartbeat less than 10 s old (and not from the future). The
  heartbeat must name a live memwatch process whose PID and creation time match.
- `watchdog_running` is kept as an alias for `run_editor.py`.
- The driver should call `watchdog_healthy()` each turn and stop the run when it returns False.
- The running memwatch already writes this heartbeat, and `memwatch.py` is unchanged, so no restart
  is needed for this change.

**Sandbox (`kernel.py:sandbox_profile`), finding 12.**
- Still `(allow default)` with explicit denies. Deny-default was not adopted.
- New denies: `(deny lsopen)`, `(deny appleevent-send)`, and `mach-lookup` of the LaunchServices
  services (`launchservicesd`, `coreservicesd`, `lsd.open`/`mapdb`/`modifydb`,
  `quarantine-resolver`) and the Apple-event service (`com.apple.coreservices.appleevents`).
- Exec allow-list narrowed to the exact interpreter: the venv `bin/python` link and the binary it
  resolves to. It no longer allows the whole venv `bin/` and interpreter home.
- Verified harmlessly with `bootstrap_look_up`, which obtains a send right and launches nothing.
  Outside the sandbox the LaunchServices lookup returns 0; inside it, the three brokers return 1100
  (`BOOTSTRAP_NOT_PRIVILEGED`).
- Other Mach services (pasteboard, WindowServer, cfprefsd, distributed notifications, DNS, ...)
  remain reachable under allow-default.

**CPU-only (this Mac).** `test_mps_is_unavailable_and_a_tiny_mps_allocation_fails` and
`test_a_tiny_mlx_op_fails` pass. `torch.ones(1, device="mps")` raises "MPS backend is unavailable",
and a 1-element MLX op raises "No Metal device available".

**Tests.**
- `test_audit.py`: `test_the_worker_thread_gradient_probe_is_flagged`,
  `test_parenthesised_and_multiline_torch_imports`,
  `test_concurrency_and_introspection_are_flagged`, `test_plain_torch_imports_stay_clean`.
- `test_kernel.py`: `test_broker_mach_services_cannot_be_looked_up`,
  `test_cleanup_never_signals_a_reaped_kernel_group`,
  `test_cleanup_kills_the_spawn_group_while_the_kernel_is_unreaped`.
- `test_gpu_memwatch.py`: `test_a_pid_file_alone_is_not_accepted`,
  `test_a_heartbeat_from_the_future_does_not_count`, `test_the_running_watchdog_is_healthy`.

**Not done (out of scope by decision).**
- Runtime blocking of worker-thread gradients and forward-mode AD.
- Frame-proofing of the integrity checker.
- Deny-default sandbox.
- The full obfuscated-name and `getattr` string-folding audit.
Those bypasses are left to the mandatory log review.

## Driver hardening (v3.1)

Driver side of the 2026-11-06 re-review, kept minimal by decision: the editor is not treated as
adversarial and every run's logs are reviewed. Runs record `harness_version: "v3.1"`.

- **Cell log (#3).** Before the kernel gets a cell, the driver appends `{"event": "start", turn,
  index, code, code_sha256}` to `harness/exec.jsonl`. In a `finally`, it appends `{"event":
  "done", output, ok, restart, error}` with the full, untruncated output. `tool_calls.jsonl` also
  stores the full output; only the editor's copy is cut at `output_chars`. `work/cells.py` is a
  best-effort copy: any failure to write it (e.g. it is a directory) is an integrity note, never a
  lost record.
- **Forward accounting (#4).** The pass is charged when inference starts (`ForwardServer.run(...,
  started)`), after request validation. A failure after that point, such as a refused or failed
  output write, counts the pass. The reply says so, and the `tool_calls.jsonl` row has `counted:
  true`. A request that fails validation still does not count. Publishing is not transactional.
- **Review gate (#5).**
  - `result.json` always starts with `compliance: "pending_review"` and `review: null`. It also
    records `audit_ok` (the audit ran with 0 violations), `budget_verified` and `integrity_count`
    (integrity rows, plus corrupt lines, plus mirror-time findings).
  - `overnight.sh` never creates a release. It publishes logs and `result.json` (checked for
    `pending_review` and `v3.1`). It refuses to launch into an existing run dir. After publishing,
    it stops the chain when the editor exits non-zero.
  - `experiments/phase3/mark_reviewed.py <run_id> --reviewer <name> --review-doc <path> --target
    <pushed sha>` is the only way out. It sets `compliance: "reviewed_ok"` and `review: {reviewer,
    review_doc, reviewed_at}` in the private and mirrored `result.json`. It then runs `gh release
    create checkpoints-<id>` with the evaluated edit. Use `--no-release` to only record the review.
- **Finalize (#7).** `_read_jsonl` keeps valid records and counts corrupt lines. The state load,
  audit, `audit.json` write, evals, manifest, integrity read, result write and mirror are each
  guarded. A truncated integrity log no longer stops recovery.
- **Budget baseline (#9).** `budget_verified` is true only if the pre-run reading is fresh. That
  means weekly, in the current window, and observed within the last hour (`BASELINE_MAX_AGE_S`).
  It is set once at start, so no later reading changes it.
- **Codex cleanup (#10).** After SIGTERM, SIGKILL always goes to the process group while it
  exists, even when the parent was already reaped and the snapshot is empty. Cleanup then waits
  until the group is gone.
- **Stage copy (#6).** Each staged file is read with a byte limit (the per-file and total caps),
  not copied to EOF. A file that grows past the cap is refused.

**Tests** (`test_driver_hardening.py` unless noted):
- #3: `test_cells_py_as_a_directory_keeps_the_code_in_the_private_log`,
  `test_code_is_logged_before_execution_and_output_is_not_cut`.
- #4: `test_partial_forward_output_consumes_the_pass`,
  `test_forward_failure_on_a_file_where_its_dir_goes_is_counted`,
  `test_bad_forward_request_is_not_counted`.
- #5: `test_overnight.py::test_publish_commits_only_the_run_dir_and_never_releases`,
  `test_failed_launch_stops_the_chain_and_an_old_run_dir_is_refused`,
  `test_mark_reviewed_records_the_review_and_releases`.
- #7: `test_truncated_integrity_log_still_finalizes`,
  `test_mirror_findings_are_in_the_published_result`,
  `test_finalize_survives_an_unreadable_state_file`.
- #9: `test_budget_needs_a_fresh_pre_run_baseline`.
- #10: `test_group_member_that_ignores_term_is_killed_after_the_parent_exits`.
- #6: `test_copy_bounds_the_bytes_read_not_just_the_reported_size`.

**Dry run (v3.1, combined code, scripted editor, all under a scratch folder).**
- Hostile probes, each handled as intended:
  - `work/cells.py` made a directory: the code is still in `exec.jsonl` and `tool_calls.jsonl`,
    and an integrity note is logged.
  - `work/fwd_2_edited.jsonl` made a directory before a `model: "both"` pass: the original file
    was delivered and the pass counted (2 of 3).
  - A kernel thread grew a staged tensor by 8 MB writes: `save_edit` was refused at the 64 MB cap.
  - A truncated line was appended to `integrity.jsonl` before `finalize_run.py`.
- Result:
  - Stop "done", no crash, 7 Python runs, 3 passes, 2 edits.
  - `compliance: "pending_review"`, `audit_ok: true` (0 violations, 2 `concurrency` flags),
    `budget_verified: true`, `integrity_count: 9` (8 rows + 1 corrupt line), no finalize errors.
  - The mirror is identical. Hidden eval is at base level (KL 4.9e-9, native 1.00).
  - Peak RSS: 1.23 GB for the run, 0.71 GB for finalize.

**Not done (by decision).**
- The driver does not poll `gpu.watchdog_healthy()` per turn. It is checked only at launch, by
  `run_editor.py`'s preflight.
- Not done either: transactional forward publishing; preallocated forward outputs (m8); stage
  entry, depth and time limits; invocation-id binding; and an eligibility matrix beyond the three
  recorded fields.
