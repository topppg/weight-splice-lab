**NO-GO at `c297a45`.** V3 closes many specific failures, but gradients still succeed undetected, executed cells can disappear from the audit trail, and partially delivered forwards can escape accounting. The new publication gate also permits results with recorded rule violations or a failed audit.

No files were modified. I inspected the changes after `558981f` and ran small, in-memory CPU probes. I did not launch Codex, exercise the live sandbox, allocate GPU memory, stress memory, or run file-writing tests.

In the tables, `harness/` and `toy/` mean directories under `src/weight_splice_lab/`; experiment scripts are under `experiments/phase3/`. “FIXED” means the earlier failure scenario is closed by reading the implementation; it does not imply live sandbox verification.

The earlier Codex review maps as follows:

| Finding | Status | Evidence and assessment |
|---|---|---|
| 1 — critical: driver follows editor symlinks | **FIXED** | `harness/rundir.py:196,236,274` opens directories component by component, rejects links/special files, and replaces output files atomically. Applied at `driver.py:215,429,485`, `forward.py:187,242`. Sandbox denies link creation at `kernel.py:49`. |
| 2 — critical: gradients with zero findings | **PARTIAL** | Original exposed setters are removed and `_C` is patched at `kernel_server.py:34–74`. A new thread still computes gradients with **zero audit findings and zero integrity violations**; see finding 1 below. |
| 3 — critical: unbudgeted supplied-model execution | **PARTIAL** | `audit.py:25–37` catches the original `runpy`/`functional_call` spelling. Runtime checks at `kernel_server.py:272–281` check registered objects, not freshly executed classes. The documented source-reexecution route remains executable; detection still depends on source spelling. |
| 4 — major: staged tensor expansion before validation | **FIXED** | `edit.py:95–115` checks dtype, shape, layout, contiguity and exact storage size before conversion. Names are checked before loading at `edit.py:137–143`. This closes the expanded-storage example. Deserialization remains in the driver; it is not a general hostile-file memory boundary. |
| 5 — major: practice/hidden grading mismatch | **FIXED** | `forward.py:75–78` and `toy/metrics.py:87–101` use `grade_ids`, defined at `metrics.py:62–64`. Termination uses token position directly at `metrics.py:179–185`. |
| 6 — major: stale base cache | **FIXED** | `evaluate.py:174–202` keys and validates checkpoint, evaluation sources, dataset snapshot, seeds, thresholds and sample count. |
| 7 — major: missing/nonweekly budget readings | **PARTIAL** | Weekly identification is enforced at `budget.py:70–85`; stale rollout readings are dropped at `codex.py:298`. Missing readings eventually stop the run. A missing initial baseline can still erase early spending, and `budget_unverified` results remain release-eligible; see findings 5 and 9. |
| 8 — major: usage drift across failures/handoffs | **FIXED** | `driver.py:248–265` advances snapshots per actual thread ID, including failed calls; refusal adoption occurs at `driver.py:345`. |
| 9 — major: ordinary 429 stops the run | **FIXED** | `codex.py:197–227` classifies message text with quota/transient precedence. `driver.py:350–357` backs off transient errors. |
| 10 — major: tool errors skip finalization | **PARTIAL** | Tool boundaries now catch ordinary exceptions at `driver.py:321,451,492`; `run_editor.py:89–94` finalizes in `finally`. Finalization itself still has unguarded failure paths, including truncated integrity logs; see finding 7. |
| 11 — major: publishing after failed runs/Git operations | **PARTIAL** | Separate worktree, path-limited commits, checked pushes and explicit release target are implemented at `overnight.sh:80–105`. Eligibility remains too permissive, and the editor exit status is only logged at `overnight.sh:116`; see finding 5. |
| 12 — major: memory checked only while waiting | **FIXED** | A continuous monitor runs at `kernel.py:129–146`; replies are sampled at `kernel.py:184–187`; vanished processes are handled individually at `kernel.py:157–162`. RSS/overshoot limitations remain. |
| 13 — major: CPU-only not enforced | **FIXED** | `kernel.py:57–67` denies the specified accelerator clients/services; `audit.py:39–42` flags accelerator APIs. Live enforcement was **unverified in this review**. |
| 14 — major: wrong MLX object patched | **FIXED** | `gpu.py:66–70` imports the actual submodule; `mlx_gen.py:102` uses that helper. Installed `mlx_lm/generate.py:246` handles the returned `None` without restoring another limit. |
| 15 — major: watchdog excludes jobs by command substring | **FIXED** | `memwatch.py:34–60` recognizes interpreter/cwd roots and descendants, excluding itself by PID. A run ID containing `memwatch` no longer excludes the driver. |
| 16 — major: stale PID satisfies watchdog requirement | **PARTIAL** | Heartbeat identity/freshness checks exist at `gpu.py:42–54`, but the legacy branch at `gpu.py:55–57` still accepts an indefinitely stalled watchdog. |
| 17 — major: deleting GPU lock breaks exclusion | **FIXED** | `memwatch.py:70–79` no longer unlinks the lock. `gpu.py:98–100` locks the persistent inode. |
| 18 — minor: returned padding contradicts metadata | **FIXED** | `forward.py:320–331` masks residuals, attention rows/columns, head outputs, logits, top-k and token logprobs consistently for ordinary finite outputs. |

The earlier Claude review maps as follows:

| Finding | Status | Evidence and assessment |
|---|---|---|
| C1 — critical: driver symlink escape | **FIXED** | `harness/rundir.py:196,236,274`; driver access points at `driver.py:215,429,485`; sandbox link denies at `kernel.py:49–50`. |
| C2 — critical: kernel GPU access | **FIXED** | Accelerator denies at `kernel.py:57–67`; accelerator audit at `audit.py:39–42`. Live runtime behavior remains **unverified here**. |
| M1 — major: kernel rules bypassable | **PARTIAL** | The requested invariants are checked at `kernel_server.py:254–289` and logged at `driver.py:414–439`. Thread-local autograd, fresh classes and mutable checking machinery defeat the stronger guarantee; findings 1–2. |
| M2 — major: audit false negatives | **PARTIAL** | Tokenization fixes the `"#"` case at `audit.py:61–85`, and patterns were expanded. Multiline imports still evade `audit.py:18,22`; output is still truncated before logging at `driver.py:401,438`; executed code can also be omitted entirely, finding 3. |
| M3 — major: missing provenance/code changes between pilots | **FIXED** | `rundir.py:355–372`, `driver.py:139–145,583–585`; `overnight.sh:22–33` pins and checks the checkout, and publishes elsewhere. |
| M4 — major: crash skips finalization but release proceeds | **PARTIAL** | `run_editor.py:89–94` and `finalize_run.py:14–32` cover ordinary crashes and manual recovery. Truncated integrity logs still defeat finalization; publication eligibility remains incomplete. |
| M5 — major: ordinary tool actions crash driver | **FIXED** | Broad boundaries at `driver.py:321–335,451–465,492–512`; permanent output capture and nonce replies at `kernel_server.py:314–358`. The new recovery creates the audit/accounting regressions in findings 3–4. |
| M6 — major: throttle classification/orphaned Codex | **PARTIAL** | Classification is fixed at `codex.py:197–227`; calls use a new session at `codex.py:256–259`. Process cleanup can still stop after TERM with surviving group members; finding 10. |
| M7 — major: memwatch forks before kill decisions | **FIXED** | In-process `sysctlbyname` at `memwatch.py:24–31`; available-memory-first independent checks at `memwatch.py:101–111,145`. The whole-process scan remains uncached. |
| M8 — major: detached kernel children survive | **FIXED** | Profile denies fork and foreign execution at `kernel.py:53–55`; observed descendants are killed at `kernel.py:222–229`. The new PID handling has its own machine-safety defect, finding 8. |
| M9 — major: permissive sandbox allows external brokers | **PARTIAL** | `/usr/bin/open` and `osascript` execution is denied. `(allow default)` remains at `kernel.py:37`; LaunchServices/Apple-event Mach services are not denied. Direct broker access remains **unverified**, finding 11. |
| m1 — minor: refusal retains old thread baseline | **FIXED** | `driver.py:248–265,345`. |
| m2 — minor: failed attempts consume reply cap | **FIXED** | `budget.py:170–183,216–219` separates replies and attempts; `driver.py:282–283` supplies the success condition. |
| m3 — minor: idle kernel death is silent | **FIXED** | Driver detects the prior death and returns/logs a restart notice at `driver.py:397–412,438`. |
| m4 — minor: parser/message edge cases | **PARTIAL** | Triple-quoted fences, `py`, longer fences and all agent messages are handled at `protocol.py:26–69`, `codex.py:137–140`. Generic `json` fences remain unsupported; they still require an explicit tool label. |
| m5 — minor: practice strips special tokens | **FIXED** | `forward.py:75–78`, `metrics.py:62–64,87–101`. |
| m6 — minor: scalar-multiple check bypass | **FIXED** | Native-slice fraction checking at `edit.py:157–167,187–189` closes the one-entry perturbation. Zero remains explicitly treated as a constant multiple, as documented in v3. |
| m7 — minor: nudge/refusal counters differ from docs | **FIXED** | `driver.py:301,340–345,367–371`. |
| m8 — minor: driver memory/guard overshoot | **PARTIAL** | Tensor validation and continuous monitoring improve this. `forward.py:295–319` still retains parts plus concatenated arrays—near 1.5 GiB for a near-768 MiB request. `kernel.py:159` still measures RSS, not compressed footprint. Use preallocated/mapped outputs and a suitable footprint measurement. |
| m9 — minor: cache keyed only on sample count | **FIXED** | `evaluate.py:174–202`; dry-run cache roots are separate in the v3 workflow. |
| m10 — minor: isolation breach gets ordinary score | **FIXED** | Codex-tool isolation stops produce `void` and `scored:false` at `driver.py:579–592`. Other integrity breaches do not; finding 5. |
| m11 — minor: overnight Git hazards | **FIXED** | `overnight.sh:22–33,83–94,113` removes pull/autostash from the pilot checkout, limits committed paths, checks Git failures and uses `--no-sync`. `runs/memwatch.log` is ignored/untracked. |

The following are the substantive open findings. “New” identifies a regression or newly introduced mechanism; “remaining” identifies incomplete closure of an earlier finding.

1. **Critical — gradients still succeed without audit or integrity findings. Remaining.**

   **Evidence:** [kernel_server.py:39](src/weight_splice_lab/harness/kernel_server.py:39), [kernel_server.py:262](src/weight_splice_lab/harness/kernel_server.py:262), [kernel_server.py:282](src/weight_splice_lab/harness/kernel_server.py:282), [audit.py:18](src/weight_splice_lab/harness/audit.py:18).

   Grad mode is thread-local. Disabling it in the kernel request thread does not disable it in a new `threading.Thread`. A worker can create a `torch.nn.Parameter`, build a graph and invoke the underlying autograd engine. Its tensors can remain worker-local, outside the shallow namespace scan.

   **Verification:** reading plus an in-memory CPU probe using the actual guard installation and integrity checker. The worker computed `d(x*x)/dx = 4.0` at `x=2`. `scan_code` returned `[]`; `_check_integrity` returned `[]`. A parenthesized multiline `from torch import (autograd as ag, ...)` evaded the import regex; split attribute strings evaded the engine-name patterns.

   **Suggested fix:** restrict editor execution sufficiently to exclude these mechanisms, and add this exact adversarial regression. Merely disabling gradients in the main thread or checking tensors after a cell is insufficient. If arbitrary Python remains available, require substantive independent trace review before accepting no-training compliance.

2. **Critical — the new integrity checker is mutable from the editor cell. New.**

   **Evidence:** [kernel_server.py:297](src/weight_splice_lab/harness/kernel_server.py:297), [kernel_server.py:355](src/weight_splice_lab/harness/kernel_server.py:355), [audit.py:33](src/weight_splice_lab/harness/audit.py:33).

   `inspect.currentframe()` can walk into the server’s execution frames. Their `f_globals` exposes `_check_integrity`; replacing it with `lambda *a: []` disables subsequent checks. The same stack exposes other server state. Moving original functions into closures does not establish a trust boundary inside this interpreter.

   **Verification:** reading plus an in-memory probe of the actual `_run_cell` mechanism. Replacement succeeded; the replacement returned `[]`; the cell had **zero audit hits**. No live sandbox escape was attempted.

   **Suggested fix:** move trusted enforcement outside the editor-controlled interpreter or use a restricted operation interface. Immediately audit frame/global/closure access and quarantine affected traces, but do not present that regex mitigation as hard enforcement.

3. **Critical — notebook append failures erase executed code from the automatic audit. New.**

   **Evidence:** [driver.py:399](src/weight_splice_lab/harness/driver.py:399), [driver.py:428](src/weight_splice_lab/harness/driver.py:428), [driver.py:436](src/weight_splice_lab/harness/driver.py:436), [driver.py:327](src/weight_splice_lab/harness/driver.py:327).

   The editor can create `work/cells.py` as a directory. Python execution succeeds, then `append_regular` raises `IsADirectoryError`. Only `IntegrityError` is caught locally. `_dispatch` records a generic Python tool-error row **without the executed code, output or integrity list**. Repeated cells then escape source auditing.

   **Verification:** reading plus an in-memory driver probe. The fake kernel recorded execution; the resulting `tool_calls.jsonl` row contained the error and traceback, but no `code`.

   **Suggested fix:** persist the immutable execution-attempt record in driver-private storage **before execution**, and complete it in `finally`. Notebook mirroring must not control audit completeness. Persist full captured output separately from the editor-facing truncation.

4. **Major — failed forwards can expose results without consuming forward budget. New recovery regression.**

   **Evidence:** [forward.py:185](src/weight_splice_lab/harness/forward.py:185), [forward.py:238](src/weight_splice_lab/harness/forward.py:238), [driver.py:458](src/weight_splice_lab/harness/driver.py:458), [driver.py:466](src/weight_splice_lab/harness/driver.py:466).

   For `model:"both"`, create a directory named `work/fwd_1_edited.jsonl`. The original model’s output file is written first; writing the edited result fails. The driver reports “not counted” and leaves `passes == 0`. The editor can read the first result and repeat with different prompts. Teacher-forced outputs have the same partial-delivery structure.

   **Verification:** reading plus an in-memory forward probe: the second write failed after `fwd_1_original.jsonl` had been delivered.

   **Suggested fix:** charge once inference begins, or publish the entire response transactionally from driver-private storage. A failure after any result becomes visible must consume the pass and retain its request/audit record.

5. **Critical — `scored` and release eligibility fail open on compliance and audit failures. New gate, incomplete policy.**

   **Evidence:** [driver.py:567](src/weight_splice_lab/harness/driver.py:567), [driver.py:579](src/weight_splice_lab/harness/driver.py:579), [driver.py:592](src/weight_splice_lab/harness/driver.py:592), [overnight.sh:65](experiments/phase3/overnight.sh:65), [overnight.sh:75](experiments/phase3/overnight.sh:75).

   A run with a recorded kernel rule violation remains `scored:true` whenever hidden evaluation succeeds. The same holds when `audit_run` fails, base evaluation fails, the manifest is missing, or the run stops `budget_unverified`. `validate` checks neither those errors nor audit/integrity eligibility before releasing the edit.

   Separately, `run_pilot` ignores the launch exit status. A failed launch against an existing run directory can publish an older qualifying result; validation does not bind it to this invocation or requested condition.

   **Verification:** **verified by reading**. No publication was attempted.

   **Suggested fix:** distinguish evaluation availability from experiment eligibility. Require complete audit/manifest/provenance/budget evidence and resolved compliance findings before automatic release. Bind validation to a fresh invocation ID and condition; honor launch failure. Preserve failed-run logs with explicit invalid status.

6. **Major — the new “size-capped” stage snapshot copies unbounded input. New.**

   **Evidence:** [rundir.py:306](src/weight_splice_lab/harness/rundir.py:306), [rundir.py:320](src/weight_splice_lab/harness/rundir.py:320), [rundir.py:325](src/weight_splice_lab/harness/rundir.py:325), [driver.py:485](src/weight_splice_lab/harness/driver.py:485).

   Size is checked once with `fstat`, then `shutil.copyfileobj` reads to EOF. A background kernel thread can grow an initially small file after the check; the driver copies beyond both caps, potentially until disk exhaustion. Arbitrarily many empty files/directories also evade byte caps and create unbounded traversal/listing work.

   Additionally, a disappearing file anywhere in the traversal raises `FileNotFoundError`, which `_snapshot_stage` treats as “no staging directory” and returns a potentially partial snapshot.

   **Verification:** **verified by reading**; no disk-filling probe.

   **Suggested fix:** bound actual bytes copied, entry count, nesting and elapsed time. Copy only the staging schema. Reject mutations/disappearances, discard partial snapshots, and distinguish missing root from failed traversal.

7. **Major — truncated integrity logs still defeat crash recovery. New finalization gap.**

   **Evidence:** [driver.py:548](src/weight_splice_lab/harness/driver.py:548), [driver.py:574](src/weight_splice_lab/harness/driver.py:574), [driver.py:580](src/weight_splice_lab/harness/driver.py:580), [driver.py:608](src/weight_splice_lab/harness/driver.py:608).

   SIGKILL or a failed write during an integrity append can leave a partial JSON line. `finalize_run.py` then calls `finalize`, whose unguarded `_read_jsonl` raises before result creation and mirroring. Repeating recovery fails identically. State loading, audit-file writing and mirroring also sit outside the advertised guarded steps.

   Mirror-only integrity findings are appended **after** result creation and copying, so the result’s integrity count omits them.

   **Verification:** **verified by reading**.

   **Suggested fix:** salvage valid JSONL records while recording corruption as an eligibility failure; guard each artifact operation independently. Write an atomic recovery result before evaluation, and incorporate mirror findings before final result publication.

8. **Major — kernel cleanup can signal an unrelated reused PID’s process group. New machine-safety regression.**

   **Evidence:** [kernel.py:139](src/weight_splice_lab/harness/kernel.py:139), [kernel.py:193](src/weight_splice_lab/harness/kernel.py:193), [kernel.py:217](src/weight_splice_lab/harness/kernel.py:217).

   The monitor can reap a dead kernel while the editor is thinking. Its PID may be reused before the next Python call. Cleanup calls `os.getpgid(old_pid)` and kills the group currently belonging to that PID. Previously, cleanup targeted the original group ID directly. The new descendant set also stores bare PIDs without creation-time validation.

   **Verification:** **verified by reading**; PID reuse and signaling unrelated processes were not exercised.

   **Suggested fix:** retain the original PGID, validate process identity before signaling, and track descendants with PID plus creation time. Join the old monitor before replacing shared monitoring state.

9. **Major — missing or stale starting budget baselines remain accepted. Remaining.**

   **Evidence:** [driver.py:172](src/weight_splice_lab/harness/driver.py:172), [budget.py:179](src/weight_splice_lab/harness/budget.py:179), [budget.py:187](src/weight_splice_lab/harness/budget.py:187).

   Without a pre-run reading, the first post-spending reading becomes the baseline. Earlier spending is permanently omitted. A later reading also clears the unverified counter without recovering that cost. Conversely, pre-run acceptance checks reset time but not reading age.

   **Verification:** reading plus an in-memory budget probe: after a missing-reading turn and a second turn reporting 40%, the guard reported `points_used:0` and `unverified_turns:0`.

   **Suggested fix:** obtain a fresh pre-spending baseline or mark the entire budget unverifiable. A later reading must not retroactively restore verified eligibility.

10. **Major — process-group cleanup may return with live Codex leftovers. Incomplete fix.**

    **Evidence:** [codex.py:233](src/weight_splice_lab/harness/codex.py:233), [codex.py:249](src/weight_splice_lab/harness/codex.py:249), [codex.py:270](src/weight_splice_lab/harness/codex.py:270).

    After normal `communicate`, the parent is already reaped. `psutil.Process(parent_pid)` can fail, producing `procs=[]`. Cleanup sends TERM to the group, then immediately breaks because the tracked list is empty. A remaining group member that ignores TERM never receives KILL. Detached/reparented descendants absent from the one-time snapshot are also not proven gone.

    **Verification:** **verified by reading**; no live Codex process test.

    **Suggested fix:** check group existence independently of the parent’s process object, escalate regardless of an empty snapshot, and verify termination before retrying. Track escaping descendants while the call is alive.

11. **Major — watchdog liveness still has an indefinite legacy bypass. Remaining.**

    **Evidence:** [gpu.py:55](src/weight_splice_lab/gpu.py:55), [run_editor.py:16](experiments/phase3/run_editor.py:16), [gpu.py:93](src/weight_splice_lab/gpu.py:93).

    With no heartbeat file, a correctly named but stalled legacy watchdog passes forever. Health is checked only at job entry; losing the watchdog during a long run does not stop the workload.

    **Verification:** **verified by reading**.

    **Suggested fix:** remove legacy acceptance for unattended runs, require a fresh identity-bound heartbeat, restart memwatch on the hardened implementation, and monitor health throughout the workload.

12. **Major — external broker access remains outside the sandbox deny list. Remaining, runtime unverified.**

    **Evidence:** [kernel.py:37](src/weight_splice_lab/harness/kernel.py:37), [kernel.py:53](src/weight_splice_lab/harness/kernel.py:53), [kernel.py:65](src/weight_splice_lab/harness/kernel.py:65).

    Foreign executable denial closes the `open`/`osascript` examples. It does not deny calling LaunchServices or Apple-event services directly through native APIs in the existing interpreter. With `allow default`, those broker routes remain permitted by the profile.

    **Verification:** profile omission **verified by reading**; successful broker-mediated escape **unverified**. I did not launch applications.

    **Suggested fix:** use a deny-default profile with necessary services explicitly allowed, or at minimum deny the identified LaunchServices/Apple-event services and validate their denial harmlessly. Narrow the execution allow-list, which currently permits entire interpreter/venv directories.

The hidden-evaluation path itself looks materially improved: practice and hidden grading share positional token-ID criteria; termination indexing preserves special tokens; KL direction remains base-to-edited; evaluation splits/seeds remain separated as intended; and the base cache now has useful provenance. I found no new primary-grading defect by reading. Full evaluation numbers were not recomputed.

**Minimum requirements for GO:**

- Close the demonstrated gradient and integrity-check bypasses, or explicitly quarantine all runs for independent compliance review before acceptance or publication.
- Make private execution logging mandatory and complete; fix partial-forward accounting.
- Fail closed on audit/integrity/budget incompleteness, and bind publishing to the successful current invocation.
- Bound snapshot copying and traversal; make interrupted-log finalization recoverable.
- Fix process identity/group cleanup and require continuously healthy hardened memwatch.
- Complete and harmlessly verify the sandbox’s broker restrictions and CPU-only enforcement on this Mac.

**Recommendation: NO-GO for two more unattended runs on the current harness.** The remaining failures can silently change the experiment’s effective rules and budgets, and the current release path does not reliably prevent those runs from being published.