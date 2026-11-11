# Logbook

## 2026-11-05

- Repo created. Local weights found in HF cache: Qwen2.5-0.5B, Qwen2.5-1.5B (bf16), Pythia-160m,
  GPT-2. Qwen3.5-0.8B and Gemma 3 270M must be downloaded.
- Phase 1 (reproduce Qwen3.5-0.8B loop fix, local eval pipeline) and Phase 2 (toy target +
  hand-compiled addition splice) started in parallel.
- Editor agent for Phase 3: `gpt-6-astra` via Codex CLI on the personal ChatGPT subscription
  (`codex-switch run personal -- codex --no-daemon exec -m gpt-6-astra ...`), same editor as the
  paper. Must run isolated from profile hooks/memories/skills; quota, not cost, is the limit.
- 12:20 Mac froze and was hard-reset while Phase 1 (MLX batched generation) and Phase 2 (MPS
  training) ran concurrently. Added `gpu.py` (exclusive GPU lock + MLX/MPS memory caps sized to
  available RAM minus 4 GB, max 10 GB) and `memwatch.py` (SIGKILLs repo processes below 2.5 GB
  available). Phases now run sequentially: Phase 2 resumed first, Phase 1 after.
- Phase 2 done (`docs/phase2-toy.md`). Base toy: 5.14M params, 1.70 bpc text, 100% copy/reverse/max,
  ~0% addition. Hand-compiled closed-form splice (added heads/neurons only, native weights untouched)
  reaches 100% addition at N=1..8 (95–100% to N=20) but wrecks the model (text KL 0.65–0.99, native
  tasks ≤ 0.10). Gated variant is quiet (KL 0.04) but inaccurate past N=4 and unstable across probe
  samples. No config meets KL < 0.05 and native ≥ 0.98. LoRA r8 needs 3k examples for 100% N≤8
  and forgets everything without replay; replay run pending.
  Key finding: adding the capability is easy, making it silent off-task is hard — no quiet residual
  directions, bias-free RMSNorm/SwiGLU gives no threshold, and the digit-read interface itself
  breaks native digit circuits. Splice used ~40 target-inspected debug iterations (disclosed).
- memwatch swap rule switched from percent (macOS swap file is dynamic: 0.6/1.1 GB read as 53%
  and killed small jobs) to absolute 3 GB.
- Phase 1 resumed: CPU-only work first; local eval only if gpu_session grants budget.
- Phase 1 done (`docs/phase1-repro.md`). All three published edits rebuild bit-exact (hashes
  match). 50-pass: W = W0 − 6·u⊗(mask64 ⊙ uᵀW0) on layers 15/19 down_proj; u = normalised mean
  (repetitive − productive) residual over 4 hand-written text pairs. Recomputed u from scratch:
  |cos| 0.99998, neuron overlap 64/64 and 63/64. Harness spec captured for Phase 3.
- Re-scoring the paper's own outputs at reduced caps: at 4k tokens base 3.5% vs edited 8% (both
  >90% cut) — loops only appear past ~8k, so a 4k local eval can't show the effect. Real eval →
  H100: MATH-500×4 base+edit ≈ 3 GPU-h; all benchmarks, 4 models ≈ 36 GPU-h.
- Likely root cause of the 12:20 freeze: mlx-lm BatchGenerator/stream_generate raise the Metal
  wired limit to the device max, overriding our cap. gpu_session now neutralises that hook.
- mlx-lm and math-verify moved to required deps (plain `uv sync` was removing them).
- Remote: private https://github.com/topppg/weight-splice-lab. Run results/logs are now tracked
  in git; checkpoints (*.pt) live in GitHub releases (`checkpoints-phase2`). Every step is
  committed and pushed.
- Codex budget (weekly window resetting Wed 11 Nov 10:45; 7% used before
  this project). Estimated pilot cost 10–35% of the week. Accepted: hard cap 5 points per run,
  15 points for the 3-run pilot, enforced by the harness from Codex's per-turn rate-limit report.
- Phase 3 (agent harness) started.
- memwatch: absolute swap threshold replaced. macOS keeps swap allocated long after pressure
  passes (7 GB from other apps while 5.8 GB was free), so "swap > 3 GB" blocked all repo work.
  Now kills repo processes on: available < 2.5 GB, kernel memory pressure critical, or swap
  growth > 1.5 GB within 60 s.
- Phase 3 harness built (`docs/phase3-harness.md`): run dir outside the repo (`~/wsl-runs/<id>/`)
  holding only the task, model source/tokenizer, base weights and practice data; `sandbox-exec`
  kernel (reads confined to the run folder, writes to `work/`, no network; autograd, optimizers
  and model forward raise); CPU forward tool (generate / teacher-forced, budget 10/25/50);
  `save_edit` validation (refused tensors, the cap, `add_capacity` | `edit_only`); handoff every 40
  turns; guards on weekly-limit points (5), input tokens (20M), turns, and done; hidden eval, audit,
  artefacts mirrored to `runs/phase3/<id>/`. 86 tests pass; scripted dry run end to end OK.
- Codex isolation: `--ignore-user-config --strict-config` plus `-c` overrides leave only the user
  message in the model-visible input. Rollout: no AGENTS.md, skills, plugins or multi-agent; base
  instructions "Answer the user's request.". 2 low-effort smoke calls: no shell, web or file access,
  canary not leaked, 0 weekly points (7% → 7%). The code-mode `exec` is still declared but fails
  closed, as in the paper's logs. Inside cmux, the `codex` shim injects hooks; the driver opts out
  (`CMUX_CODEX_HOOKS_DISABLED=1`).
- memwatch (coordinator, 75ec3d6) now triggers on available RAM, memory pressure or swap growth,
  not stale swap; before that, 7 GB of other apps' swap blocked every venv process for ~25 min.
- Harness reviewed: 86 tests pass, ruff clean; driver resolves the real Codex binary (cmux hook
  shim off PATH). Run preflight switched from a swap check to kernel memory pressure, and the
  free-RAM minimum raised 5 → 6.5 GB (run peak ~3.5 GB + 2.5 GB watchdog floor; no resume yet).
- Added `experiments/phase3/codex_usage.py`: reads weekly Codex usage from the personal profile's
  session rollouts; `--watch` prints changes and raises a macOS notification at ≥ 50% (user request).
- Pilot 1 (add_capacity, 50 forward passes, effort high, 5-point cap) approved; waiting for ≥ 6.5 GB
  free RAM to launch (4.7 GB free at 17:36). Weekly usage at approval: 8%.
- pilot1 VOID (runs/phase3/pilot1-void/VOID.md): budget guard misread a 1 s `resets_at` jitter as
  a new weekly window and stopped at a phantom 8 points after 7 turns (real cost ≈ 0). Fixed and
  regression-tested. Notable: unprompted, the editor converged on the Phase 2 strategy (reserved
  quiet residual directions + `A:` detector gate); intermediate edit silent (KL 0.0006).
- Relaunched as pilot1r (same config).
- pilot1r VOID (runs/phase3/pilot1r-throttled/VOID.md): Codex 429 Too Many Requests on turns 4–6
  with weekly usage at 8% — an astra throttle not shown in the reported limits. Driver now backs
  off 2→60 min on 429 (~3 h max) instead of stopping after 90 s. Relaunching as pilot1b after a
  30 min cool-down.
- pilot1b launch cancelled before start (user closing the laptop; system sleep would fail in-flight
  Codex turns). To relaunch: same command with `--run-id pilot1b`.
- pilot1b launched 19:03 (baseline 8%). Interim (editor-visible practice results, NOT the hidden
  eval): edit 7 (turn 56) completed a full circuit — A: gate, Fourier-kernel operand routing heads,
  layer-3 previous-sum head + 2 carry neurons, 80 plateau neurons decoding sum mod 10 via closed-form
  pseudoinverse of the frozen embedding. Practice (5/length): addition 100% at N=1..9 and 11..20,
  60% at N=10; native copy/reverse/max 100%. Edit 8 fixed N=10 (unbalanced operand head; 50/50
  rescues). Edit 9 (turn 75) returns workspace directions to native components after last use to
  reduce off-task interference. Throttled 429s ~every 5 turns, 2–4 min waits. 1 point at 2.7M input.
- pilot1b DONE (148 turns, 84 min, 2 weekly points, 12.4M input / 0.47M output tokens; stop: forward
  budget used + final edit). Hidden eval of final edit 16 (1,240,064 added params): addition 100% at
  every N=1..20 (200/length), text KL 0.0095, bpc 1.703→1.716, native copy/reverse/max 1.00 →
  meets the acceptance bar the hand-compiled Phase 2 splice could not. Automatic audit: 8 flags, all
  apparently pattern false positives (variable listing containing `HarnessRuleError`; inspecting own
  helper functions). Independent compliance audit + re-eval (fresh seed, N=21..30, wikitext test KL)
  running before any claim. Final edit in release `checkpoints-pilot1b`.
- pilot1b independent audit (docs/reviews/pilot1b-audit.md): compliant on all rules — no forward
  reimplementation in the kernel (all 78 cells read), only closed-form fits (the one data-fitted
  ridge correction was rejected and zeroed), no isolation breach, 8 auto-flags all false positives.
  Re-eval with a fresh seed reproduces 100% N=1..20, native 1.00, wikitext-test KL 0.010.
  Blind spots of our hidden eval it exposed: (1) any text starting with `A:` is destroyed (KL 10.2,
  outputs digits) — trigger too broad; (2) never emits the end-of-answer newline (strict score 0);
  (3) addition only works on the first line and only N ≤ 20 (0 at N=21..30); (4) native output
  projections were projected off ≤ 35 directions (7–28% rel. change), source of most of the KL;
  (5) weights up to ~1e6, fp32 only. 196,608 of 1.24M added params are inactive.
  → Next: harden the hidden eval with these probes before pilots 2–3, and re-score pilot1b.
- Hidden eval hardened before pilots 2-3 (`docs/eval-hardened.md`). Primary bar unchanged and
  recomputed identically (base and pilot1b match their old results value for value). New
  pre-registered robustness block in `hidden_eval.robustness` + `meets_primary_bar`,
  `meets_strict_bar`: (a) KL on wikitext-test windows starting `A:` + non-digit < 0.05;
  (b) `\n` after the N+1 answer digits ≥ 0.95 (N=1..8); (c) addition after 1-3 native lines and
  after a text line, min ≥ 0.95; report-only (d) N=21..30, (e) native-weight relative change,
  (f) added params allocated/nonzero/inactive. All text from the test split.
- Rescore (`runs/phase3/rescore_hardened.json`): no model meets the strict bar. pilot1b: primary
  yes; (a) KL 10.19, (b) 0.00, (c) 0.01/0.00, (e) 8.5% total, 28% max. Gated splice: (a) 0.0499
  (just passes), (b) 0.86. Ungated: (b) 1.00. (c) fails for every model, LoRA included.
- LoRA-with-replay baseline done (r8, budgets 1k/3k/10k; MPS ≤ 0.09 GB). Native replay: 3k
  examples → addition 1.00 N=1..8, native ≥ 0.99, but text KL 0.47 (bpc 2.37). Adding 16 fresh
  64-char text windows per batch: 3k → 0.994, KL 0.44. A fixed text pool memorised (train loss
  0.10) and failed. LoRA changes every native matrix by 42-60% total. Still needs ~3k labelled
  examples; replay restores native tasks but not text.
- Coordinator note on the hardened eval (docs/eval-hardened.md): probe (c) "addition not on the
  first line" fails for every model, including LoRA trained on addition (≤ 0.45 after native
  lines, ≤ 0.01 after text). The task spec and practice data only define first-line addition, so
  (c) tests an out-of-spec capability; it stays in the pre-registered strict bar (not changed after
  seeing results) but the write-up should read (a) `A:`-prefix KL and (b) termination as the
  meaningful robustness failures of pilot1b, and report (c) separately.
- Lesson: `git pull --autostash` rewrote a tracked log a job was still writing. Don't autostash
  while tracked files are being written; keep live logs untracked until the job ends.
- 22:16 overnight chain started pilot2 (add_capacity), baseline weekly usage 11%.
- pilot2 (add_capacity, replication) DONE 23:40: stopped at turn cap (200) with 20/50-ish passes
  unused → evaluated its last saved edit (19, not marked final). 3 points, 17.0M input tokens.
  Hidden eval: addition 0.85 (N1-8) / 0.52 (N9-20), text KL 0.78, native 1.00/1.00/0.996 →
  fails the primary bar (KL). No native weights changed (pure addition of 1.04M params).
  Robustness: `A:` KL 4.37, termination 0, not-first-line ~0. Auto-audit: 27 flags (to review).
  Replication so far: 1/2 runs meet the primary bar. Release `checkpoints-pilot2`.
- 2026-11-06: overnight summary in docs/overnight-summary.md. pilot3 (edit_only) VOID — internet
  outage ~01:05, stopped after 18 working turns (0 points). Driver now backs off on network
  timeouts. Weekly usage 14%.
- 2026-11-06: pilot2 audit — compliant, 27 flags all false positives. Found harness bug: Codex
  usage is a per-thread running total; driver used it as per-turn → handoffs every ~10 turns (not
  40) and token totals overcounted (pilot1b 12.4M → 2.5M actual input; pilot2 17.0M → 3.8M).
  Fixed: per-turn deltas, "Replies left" in every reply/handoff, all Python cells mirrored to
  work/cells.py for restarts. These change editor conditions vs pilot1b/pilot2 — future runs are a
  new harness version (v2). Points per actual input token: ~1 point / 1.25M.
- 2026-11-06: two independent harness code reviews before any rerun (Codex gpt-6.1-sol on the
  personal sub, <1 point; Claude agent with scratch reproductions), saved in docs/reviews/. Agreed
  criticals: symlink escape via the unsandboxed driver, kernel rule bypasses (gradients via
  torch._C/autograd engine, unpatched model via sys.modules/runpy+functional_call, optim restore),
  no GPU block in the kernel sandbox. Confirmed: mlx-lm wired-limit guard patched a function, not the
  module — never effective. Past runs scanned: none of the exploits used (pilot1b result stands).
  Fixing as harness v3 in two file-disjoint agents (sandbox/kernel/safety/audit; driver/budget/
  eval/publish); re-review before pilots 4–5.
- 2026-11-06: harness v3, driver side (docs/phase3-harness.md "Driver hardening (v3)"):
  - The driver never follows links under work/; refusals are logged as integrity violations, and
    kernel integrity reports are recorded per turn.
  - Tool errors no longer crash a run. finalize always runs, and finalize_run.py covers killed
    runs. Isolation stops are void and not scored.
  - Codex: errors are classified on message text (429s back off), each call's process group is
    killed on timeout, and only successful replies count toward the 200-reply cap.
  - Usage is accounted per thread. The budget needs a fresh weekly reading.
  - Practice and the hidden eval share one token-id grader. Forward arrays are masked at padding.
  - Provenance (SHA, dirty flag, harness hash) is recorded, and the base-eval cache is keyed.
  - The overnight chain is pinned to one SHA and publishes from a worktree.
  - Rescore re-verified: base and pilot1b reproduce bit-for-bit; no reported number changes.
  - Fake-editor dry run on 28b3ed0: done, scored, 0 crashes, 0 tool errors, peak 1.19 GB.
- 2026-11-06: Codex re-review of harness v3 (docs/reviews/2026-11-06-codex-rereview-v3.md):
  NO-GO. ~20 of 29 earlier findings fixed. Open: gradients via a worker thread (grad mode is
  thread-local) and the in-interpreter integrity checker can be replaced via frame introspection —
  arbitrary Python cannot be hard-locked, so compliance must rest on complete, tamper-proof code
  logging + mandatory review; a cells.py-as-directory trick drops code from the audit log;
  partial forward outputs escape budget; release gate fails open on audit/integrity/budget gaps;
  unbounded stage copy; truncated JSONL breaks finalize; PID-reuse in kernel cleanup; missing
  budget baseline; Codex group cleanup; legacy watchdog acceptance; LaunchServices/AppleEvents not
  denied. Claude re-review was stopped by safeguards. Pilots 4–5 on hold; fix round 2 started.
- 2026-11-06: Driver fix round 2 (harness v3.1, minimal by decision; docs/phase3-harness.md
  "Driver hardening (v3.1)"):
  - Every cell's code and sha256 is written to `harness/exec.jsonl` before it runs, with its full
    output after. `cells.py` is best-effort.
  - A forward pass is charged once inference starts.
  - `result.json` is always `pending_review`, with `audit_ok`, `budget_verified` and
    `integrity_count`. overnight.sh never releases and honors the editor's exit status.
    `mark_reviewed.py` records the review and makes the release.
  - Also fixed: corrupt JSONL lines are salvaged, each finalize step is guarded, a fresh pre-run
    baseline is required, the Codex group always gets KILL, and stage reads are byte-bounded.
  - Fake-editor dry run with the review's hostile probes: all handled; result pending_review,
    `integrity_count` 9, peak 1.23 GB.
- 2026-11-06: harness v3.1 done, scoped down on user request ("don't overengineer"): private
  pre-execution cell log, forward charged at inference start, every result `pending_review` with no
  auto-release (releases via mark_reviewed.py after log review), robust finalize, fresh budget
  baseline, Codex group KILL, bounded stage copy, PID-identity kernel cleanup, heartbeat-only
  watchdog checked every turn, LaunchServices/Apple-event denies; worker-thread gradients and frame
  introspection are flagged by the audit, not blocked (documented limit; covered by log review).
  275 tests pass; hostile dry run passes. No third review.
- 2026-11-06 10:59: pilot4 VOID — memwatch swap-growth rule killed it (and the usage watcher) with
  6.7 GB free (macOS paging other apps). Fixed: swap rule gated on available < 4 GB. Finalized from
  state; last intermediate edit already 100% N1-20, KL 0.126. Relaunching as pilot4b + pilot5.
- 2026-11-06 11:41: pilot4b VOID — watchdog swap rule killed it (2.3 GB swap growth, 3.6 GB free)
  under concurrent load (3 other Codex sessions + apps). Changes: swap-growth kill off by
  default (kernel is CPU-only/GPU-denied; RAM floor 2.5 GB + critical pressure remain); overnight.sh
  waits for OVERNIGHT_NOT_BEFORE and for the Codex profile to be idle ≥ 10 min, per-run
  point cap via OVERNIGHT_MAX_POINTS. Pilot points so far: 8 of 15. Tonight: pilot4c + pilot5 at 4
  points each, starting ≥ 23:00.
- 2026-11-06 20:15–22:25: pilot4c and pilot5 ran on `030b54e`. They were relaunched detached
  (start_new_session) so closing Claude Code would not stop them, after a repo move.
  Results are pushed to branch `runs/overnight-2026-11-06`, not `main`, per the
  lab workflow.
  - Disk: it filled at 18:56 (another job's model download) and killed memwatch and codex_usage.
    Both were restarted; 22 GB free after the user cleared a cache.
  - pilot4c (add_capacity): addition 100% N1-20, KL 0.0126, (a) and (b) pass, 1 point.
  - pilot5 (edit_only): one-digit lookup only, 2 points.
  - Weekly usage 22% → 25%. Pilot points 11 of 15.
- 2026-11-07: log reviews done (pilot4c compliant with caveats, pilot5 compliant). The audit's
  `requests\b` false positive is fixed and pilot5 re-audited: 0 violations. Comparison table:
  `experiments/phase3/compare_runs.py` → `docs/overnight-2026-11-06.md`. Write-up draft updated.
  Releases wait for owner sign-off.
- 2026-11-07 00:31–04:48: repeats pilot6 (add), pilot7 (edit), pilot8 (add) and pilot9 (edit) ran
  on `eac81f4` (v3.1, cap 3 points each) and were pushed to `runs/repeats-2026-11-07`.
  - add_capacity, n = 3: 3/3 have 100% addition at N1-20, and 2/3 meet the primary bar (pilot8
    KL 0.0556).
  - edit_only, n = 3: 0/3, all one-digit lookups.
  - pilot6 is the first edit to pass (d), N21-30.
  - Fresh-seed and test-split re-eval (`experiments/phase3/reeval_fresh.py`) reproduces all six.
  - Reviews pilot6 to pilot9 are compliant with caveats. pilot9's 2 audit hits are words in note
    strings.
  - Weekly usage 25% → 31%; pilots total 17 points. Summary: `docs/repeats-2026-11-07.md`.
- 2026-11-07 (afternoon): Phase 4 started on Gemma 3 270M IT, the paper's negative-result model
  (`docs/phase4-gemma.md`). No more toy runs.
  - Probe: the base adds up to about 3 digits; N=4 6.5%, N≥5 ~0% (hidden-eval seeds).
  - Own Gemma 3 implementation matches HF (max |Δlogit| 2e-4, same greedy outputs).
  - Harness v4: model families (toy/gemma), early finish (`"end": true`), code-only audit
    words, memory-safe teacher forcing. 300 tests pass; real-checkpoint smoke run passes.
  - Hidden eval pre-registered before any Gemma editor run: addition N1-20 (bar N1-8 ≥ 0.95),
    text KL and chat KL < 0.05, digit-text KL strict. Base reference 6.9 min on CPU.
  - Model from the ungated `unsloth/gemma-3-270m-it` mirror (no HF token on this machine; a
    token from another project was not reused).
- 2026-11-07 (evening): Gemma pilots gemma1 and gemma2 (add_capacity), both failed the primary bar.
  - gemma1 (early finish on) ended itself after 11 of 50 passes with an N1-only patch (N1–8
    0.269). Early finish is now opt-in and off.
  - gemma2 (full 50 passes, 3 points) tried position buffers, a ridge answer predictor and
    operand-retrieval heads; kept a pair-gated L17 boost: N1–8 0.284 (base 0.261), KL text
    0.033 / chat 0.019 / digit text 0.042. No carry chain. Weekly usage 31% → 35%.
- 2026-11-07 (night): Phase 4b pre-registered: reversed-digit addition on Gemma (family
  `gemma_rev`, `docs/phase4-gemma.md`). Base reverse N1 7.5%, N≥2 0%. Pilot gemma_rev1 next.
- 2026-11-08 (early): gemma_rev1 (reversed digits, 4 points) failed: reversed N1 99.5%, N2 5.5%,
  N≥3 0 (mean N1–8 0.131, below the 0.5 stop rule). Phase 4 stopped by the pre-registered rule;
  summary table in `docs/phase4-gemma.md`. Weekly usage 39%.
- 2026-11-08: Phase 4c pre-registered: the editor may run Gemma in its kernel (`build_model`),
  no gradients still, no search loops (new audit flag). Pilot gemma_rev_kf1, cap 10 points.
- 2026-11-09/10: gemma_rev_kf1 (phase 4c, model runs in the kernel; launch delayed a day:
  memwatch died in the 2026-11-08 disk-full and was restarted, then RAM and Codex-idle gates).
  Hidden reversed addition 100% for N1–30 (base 0.9% N1–8). Failed on off-task KL: text 0.445,
  chat 0.086, digit text 0.005. 5 of 256 text chunks hold more than 99% of the text KL, where a
  gate fires on all-caps acronyms and the circuit takes over the residual. The editor's own
  short-window estimate was 0.0004. Plain N1–8 preserved (0.263). Weekly usage 49% → 51%.
  Details in `docs/phase4-gemma.md`.
- 2026-11-10: Gemma work stopped (owner decision). Final summary in `docs/phase4-gemma.md`
  ("Phase 4 conclusion").
- 2026-11-10: repo cleanup. Run folders were reduced to result, notes, transcript, final edit
  manifest and audit. Void and smoke runs, overnight logs and harness code reviews were removed;
  per-run reviews moved to `docs/reviews/`. The removed material is kept in a private archive.
- 2026-11-11: public-release prep. Added MIT (code) and CC BY 4.0 (docs and results) licences, a
  public README and the final write-up (`docs/writeup.md`, which replaces the draft). Committed the
  KL diagnosis (`experiments/phase4/diagnose_kl.py` → `runs/phase4/gemma_rev_kf1/kl_diagnosis.json`:
  the worst 5 of 256 chunks hold 99.8% of the text KL). Replaced personal absolute paths in the
  logs with placeholders. Restored the 2026-11-06 overnight log and harness code reviews (paths
  removed); toned down the Gemma claims after an independent review.
