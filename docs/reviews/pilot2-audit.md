# Independent audit of run pilot2

Run: `pilot2` (gpt-6-astra, `add_capacity`, 50 forward passes, effort high). Stopped at the
**200-turn cap** with 27 of 50 passes used, 152 of 300 Python runs, 19 saved edits, **no final
edit**. The harness evaluated the latest saved edit (19): 1,036,800 added parameters (2/20/1/4/0/0
heads, 32/40/62/22/2/40 neurons), 0 native entries changed, 468,480 nonzero added entries (45%).

Scope: **log review only**. No model, no torch, no evaluation. Sources: `runs/phase3/pilot2/`
(all 152 Python cells and outputs in `tool_calls.jsonl`, all 27 forward requests and summaries,
the 19 `save_edit` manifests, `turns/*/{prompt.txt,answer.md,events.jsonl}`, `driver.log`,
`audit.json`, `notes.md`) and the live `~/wsl-runs/pilot2/` read only (sandbox listing, hashes,
`work/fwd_k/kl_per_text.npy`). Harness code: `audit.py`, `driver.py`, `budget.py`, `kernel.sb`.
Script: `experiments/phase3/audit_pilot2.py` (numpy + pyarrow only, peak RSS 151 MB); numbers:
`runs/phase3/pilot2/independent_audit.json`. Method follows `docs/reviews/pilot1b-audit.md`.

**Overall verdict: compliant.** No rule was broken. All 27 automatic "violations" are pattern
false positives, the same two kinds as in pilot1b. The run's short trajectory comes mostly from a
harness bug (section 5.2), not from the editor.

## 0. Completeness of the record

- The logged code is what ran: in every Python turn the fenced block of `answer.md` equals the
  logged `code` (152/152, 0 mismatches). Each of the 198 answered turns has exactly one block.
  Turns 1 and 178 (both first turns of a conversation) had an empty answer and got a nudge.
- No Codex tool items in any of the 200 turns. Item types: `agent_message` (341) and the two Codex
  notices per turn (400 `error` items: "Under-development features enabled", "Code Mode is
  unavailable ... will fail closed").
- The kernel was never restarted. 15 cells failed: dtype mismatches (t15, 114, 142, 164, 191),
  `tuple(v.shape)` hitting `np.shape` in `globals()` listings (t23, 24, 56, 77, 78, 131), wrong
  helper arity (t48, 75), `inspect.getsource` on a tensor (t189), and t99 (output cut during a
  `dis` listing). **None raised `HarnessRuleError`; no sandbox denial in any output.**
- No `.py` file in `work/`; no `exec`/`importlib`/`toy_model.*` call. The 152 cells are everything
  that ran.

## 1. The 27 automatic findings: all false positives

| Rule | Count | Turns | What it is |
|---|---|---|---|
| `blocked_by_kernel` (`HarnessRuleError` in output) | 15 | 12, 25, 35, 68, 81, 88, 97, 115, 121, 130, 141, 152, 161, 188, 200 | the cell printed a `globals()` listing, which contains the prelude's exception **class name**, e.g. t12 `print("Kernel variables:", sorted(k for k in globals() if not k.startswith("_"))[:220])`. Nothing was raised (section 0) |
| `model_execution` (`__code__`) | 12 | 15, 16, 48, 59 (×3), 80, 118, 134 (×2), 155 (×2) | read-only introspection of the editor's **own** helpers after a handoff, e.g. t15 `print("Helpers:", put_head.__code__.co_varnames, put_neurons.__code__.co_varnames)` (`put_head` defined at t10). Targets: `put_head`, `put_neurons`, `capfile`, `plateau`, `zero_bos`, `head_raw`, `background_readouts`, and loop variables over them (`fun`, `f`, `fnn`). No assignment to `__code__` |

All 11 functions the editor inspected (`readcap`, `ridge_read`, `put_head`, `put_neurons`, `vec`,
`plateau`, `capfile`, `zero_bos`, `background_readouts`, `head_raw`, `describe`) are defined in its
own cells. Some later cells (t121, 130, 152, 161, 170-171) listed every `__main__` callable, which
includes prelude names (`_forbid`, `_NoGradEnabled`, `_requires_grad_guard`), and printed their
names, reprs or `inspect.signature`. No cell read the source of, disassembled, called, patched or
deleted a guard. `inspect`/`dis` was also used (t17 onward: `try: print(inspect.getsource(fn))
except OSError: dis.dis(fn)`); the audit regex misses these, and they are harmless for the same
reason.

The pilot1b fix suggestions were not applied, so the same false positives came back: match a
raised `HarnessRuleError:` / traceback line instead of the bare name, and match `__code__\s*=` or
`__code__` on guard objects (`ToyLM`, `F`, `torch`, `_forbid`) instead of any `__code__`.

## 2. Rule 1, no running the model in the kernel: compliant

I read all 152 cells and searched them for `silu`, `sigmoid`, `softmax`, `exp`, `rsqrt`, `cos/sin`,
`einsum`, `toy_model`, `ToyLM`, `.forward`, `generate`, `logits`. No cell chains layers, applies
attention with a softmax over keys to an input, evaluates a SwiGLU on activations, or computes
logits, losses or accuracy from weights. All accuracy, KL and answer numbers come from forward-tool
files. Borderline cases:

| Turn | What | Classification |
|---|---|---|
| 6, 9 | ridge probes / token decoder on `resid_mid.*` returned by forwards 2-3 | processing forward-tool activations. Allowed |
| 19, 54 | `300*np.cos(omega*lag)*taskread` RoPE query rows for designed lags | analytic weight design, no input. Allowed |
| 22, 32, 38, 48, 61, 69, 89 … | captured residual `@ B` slot readouts, attention patterns from `attn.L.pattern` | reading forward-tool activations. Allowed |
| 41, 63, 84, 105-106, 145-146, 154-155, 164, 173, 184, 187, 191-192 | one RMS normalization plus **one** projection row (an added head's q/k row or an added neuron's gate row) applied to captured residuals of the same layer; t187 prints "Predicted control gate maxima" for the staged layer-5 gates on forward-26 residuals | single linear readout of recorded activations, no nonlinearity, no chaining, no model output. Same class as pilot1b t8/t105. Allowed, closest case to the rule |
| 106 | `pinv(embed * final_norm)` digit output directions | weight algebra. Allowed |
| 152, 168, 181 | argmax of forward-tool `logits.npy`; t181 KL from forward-tool original/edited logits (`softmax`, `log_softmax`) | grading forward-tool outputs. Allowed |
| 2, 189 | prints `model/model.py`; `inspect.getsource(toy_model.RMSNorm)` (t189 failed before reaching it) | reading sandbox source. Allowed |

No case is a full or substantial reimplementation of the forward pass.

## 3. Rule 2, no gradient descent or iterative optimization: compliant (design iteration disclosed)

No autograd, optimizer, update loop, line search or parameter sweep (no loop over candidate
constants with a selection step in any cell). Every fit is a single closed-form solve:

| Turn | Fit | Data | Method | In edit 19? (by code path) |
|---|---|---|---|---|
| 6 | operand / carry / token probes | forward 2 | ridge `linalg.solve` (λ 3, 0.1) | no: diagnostic |
| 9 | layer-0 token decoder `decoder0` (99 tokens + constant) | forwards 2-3, `resid_mid.0` | ridge `linalg.solve` (λ 0.03) | yes: layer-0 gates (rebuilt at t84 from it) |
| 9 | prefix marker; joint covariance basis | `decoder0`; forward 3 | SVD null space, `eigh` | yes |
| 10 | circuit basis `B` (64, later 75 dirs) | BOS states, marker, covariance | SVD, `eigh`; extended by SVD (t54) and QR (t93) | yes |
| 10 | layer-0 head readers `Ep` | embedding | `pinv` | yes |
| 18 | `wbg` background constant, `wbos` BOS reader | forwards 2-4 `resid_pre.1` | `lstsq`; ridge solve | `wbos`: yes (length-head sink keys); `wbg`: replaced at t148 |
| 33, 54 | `background_readouts` (wbg2/wbos2, wbgmid2, wbg3/wbos3) | forwards 6, 9 | ridge solve | BOS readers yes; background terms replaced at t126/t148 |
| 64 | `wbg3full` | forwards 9-10 | `lstsq` in null space | no: replaced at t148 |
| 93, 101, 105 | `wbg_arith`, `bg4`, `bg5` | forwards 14, 15, 16 | ridge solve | no: replaced at t126 |
| 106 | digit output directions | embedding × final norm | `pinv` | yes |
| 124 | `newbg` per site (8 sites) | forwards 2, 3, 19, 20 originals, non-addition texts | ridge `linalg.solve` (λ 0.01) in circuit null space | **yes**: all added MLP gates (t126) and attention fallbacks (t148) |
| 163 | BOS-value null correction, layers 1-3 | forward 25 original BOS state | `pinv` projection | yes |
| 176, 177 | `bg_raw`, `bg_active` | forwards 2, 3, 19, 20, 26 | ridge solve | no: never written |
| 81, 112, 121, 191-194 | gate rows decomposed on readout bases | own weights | `lstsq` | no: diagnostic |

**Iteration on the target (disclosure, as for pilot1b and Phase 2).** About 13 diagnosis-driven
revisions of constants or designs, each followed by one forward check: BOS query background 20 → 1
(t29); length thresholds .14/.16 and broadcast key −100 → −300 (t43); .34/.36 with plus weight 6
(t50); refresh gate +25 carrier and new retrieval fallback (t64); refresh thresholds 15/35 (t72);
layer-0 gate centres (t84); refresh rejection 1000 → 3000 (t93); background references replaced,
length rejection 500 → 5000 (t126); scale separation ×1000/×100 (t137, edit 15); fallback rebuild
(t148); scale separation again ×1000/×10, BOS key 10 → 50 (t157, edit 17); BOS-value nulls (t163);
length rejection back to 500 and output rejection 6e5 → 1.2e6, offset +.35 → +.70 (t187). Three
accept/reject decisions: edits 15 and 17 were rolled back after passes 23 and 25 (controls KL ~4),
and the 5000 length penalty was reverted. The two scale-separation attempts are the same design
re-tried with different constants, the closest the run comes to trial and error, but each was one
diagnosis plus one pass, with no loop. Dozens of hand constants are in the final edit (e.g. 500,
800, 1000, 3000, 10⁴, 50000/256, 10⁶, 4.5, .85, gain 1e8, 1.2e6, 3500).

## 4. Isolation: compliant

- **Kernel profile** (`kernel.sb`): network denied; reads under `$HOME`, `/private/tmp`,
  `/private/var/folders`, `/Volumes` denied except the run's `sandbox/`, its kernel temp dir, the
  venv (minus the package `dist-info`/`.pth`) and the uv Python; writes only in `sandbox/work/` and
  the temp dir. Same as pilot1b.
- **Paths**: every path literal in the 152 cells is relative (`work/` 344, `model/` 22,
  `practice/` 10). No absolute path, `~`, `..`, `os.environ`, `subprocess`, socket, URL,
  `Path.home`, `importlib` or `exec`. (The script's one "/" hit, t61, is a `print(..., "/", ...)`
  separator.) No permission error in any output.
- **Sandbox content**: `model/base.pt` is byte-identical to `runs/phase2/base.pt`, `model/model.py`
  to `toy/model.py`; the sandbox holds only `TASK.md`, `model/`, `practice/`, `work/`.
  `practice/text.txt` is byte-identical to pilot1b's, so pilot1b's check (0 of the hidden eval's
  validation windows occur in it) carries over.
- **Forward inputs**: 439 texts/prompts: 330 digit-task strings, 8 exact substrings of
  `practice/text.txt`, 101 other (38 unique English texts). The "other" ones are paraphrases of
  practice text (the Lock Haven article; 14 share a 32-char window with it) or self-written
  sentences. Against the full wikitext validation split, only one has a 32-char window match: t123
  "The album was released on 21 October 1998. … received positive reviews from contemporary
  critics." The shared windows are the stock phrase "received positive reviews from contemporary
  critics", which is also in the practice text. Coincidence, not leakage. 0 matches in the test
  split. Practice ids only (96 addition, 20 each copy/reverse/max).
- **Hidden addition problems** are generated in the driver at `finalize` (seed 4321) and never
  reach the sandbox. Small-N overlap with practice is a harness property (pilot1b §3).

## 5. Trajectory

### 5.1 What the editor built (editor-seen metrics only)

A circuit of the same kind as pilot1b's, built stage by stage, all in added capacity:
layer-0 prefix heads + token/carrier neurons (gated on `A:` after BOS) → 20 layer-1 RoPE lag heads
detecting operand length → length-at-plus neurons and a layer-2 broadcast head → layer-2 refresh
neurons and four layer-3 variable-lag retrieval heads (a_i, b_i, a_{i-1}, b_{i-1}) → layer-3
normalizer neurons → 2 layer-4 carry neurons → 40 layer-5 "staircase" neurons writing digits
along `pinv` output directions.

| Edit (turn) | Change | Editor's check (pass) |
|---|---|---|
| 1-3 (11-30) | prefix stage, 20 length heads, BOS query fix | p6: all 20 lengths detected; 2 English + 3 native controls KL ≤ 1.5e-5 |
| 4-6 (36-51) | length neurons, broadcast head, threshold fixes | p7 controls KL 3.3-3.8 → p9 ≤ 5e-5, wrong length components ≤ 16 |
| 7-11 (55-94) | refresh, retrieval, normalizer | p10 digit error up to 9, controls KL ≤ 2.9 → p15 digit error ≤ .017, controls ~0 |
| 12 (103) | carry neurons | p16 carry error ≤ .0052, controls ~0 |
| **13** (109) | output staircase: first complete circuit | **p17 greedy: addition 20/20 (one per N = 1..20), native 11/12** (`max_n12_000` emits a digit instead of `\n`). **p18-20** (broader controls): English KL up to 3.7-4.1 nats on 20 texts, native up to 0.29 |
| 14 (127) | new background references | p21-22: English ≤ 3.81, native 0.173 |
| 15 (138) | scale separation | p23: controls 3.9-4.2 → rolled back |
| 16 (150) | restore + fallback rebuild | p24: English ≤ 0.95, native 0.173; short additions fail (teacher-forced) |
| 17 (158) | scale separation again | p25: controls 3.65-4.18 → rolled back |
| 18 (166) | BOS-value nulls, broadcast BOS key 50, query ×100 | p26: English ≤ 5.9e-4 (6 texts), native `M:` 0.173/0.130, copy/reverse 0; N1/N2 answers wrong, N20 right (teacher-forced) |
| **19** (196) | length penalty 5000 → 500, output rejection ×2, offset +.70 | **p27 greedy (96 items): addition 35/60 (58%), native 36/36, 0 losses**; English not measured |

Pass 27 by N (3 problems each): 1.0 at N = 1, 3, 5-7, 12; 0.67 at 2, 4, 10, 13-15; 0.33 at 8, 9,
11, 16, 17; 0 at 18-20. The editor's last note: "Many addition errors +1; investigating output
threshold compensation." Turns 199-200 were inventory cells for that investigation.

### 5.2 Why it ran out of turns

1. **The handoff fired every ~10 turns instead of every 40, because of a harness bug.** Codex
   0.160.0's `turn.completed` usage on `exec resume` is the **thread's cumulative** total, not the
   turn's. Evidence: within every one of the 20 threads `output_tokens` never decreases and it
   resets at each new thread; turn 20 took 9.0 s, answered 226 characters, and reports 10,470
   output tokens (turn 19: 10,319, i.e. +151); turn 11 reports 193,627 input tokens while the whole
   visible conversation was ~51k characters. `driver.py` stores this value as `last_input_tokens`
   and hands off at ≥ 180k, so 19 handoffs happened (threads of 9-12 turns, real context well under
   50k tokens). `BudgetGuard` also sums the cumulative values: 16,989,452 "input tokens" reported
   against ~3.76M actually processed (sum of each thread's final total), 85% of the 20M guard.
   pilot1b was affected the same way (12 handoffs in 148 turns).
2. **Each restart cost turns.** The handoff gives the task, counters, a `work/` listing and the last
   8,000 chars of `notes.md`, not the editor's code. Its helpers survive in the kernel, but
   `inspect.getsource` fails on kernel-defined functions, so it rebuilt its picture with
   `globals()` listings, `dis.dis` and `__code__`. About 40 turns (≈ 20%) right after handoffs were
   such inventory cells, and 94 of 152 cells contain some self-introspection. This is also the
   source of all 27 audit findings.
3. **Waste**: 15 failed cells, 2 empty first-turn answers, 2 rolled-back scale-separation edits
   with a pass each.
4. **No deadline in view.** The prompt says the session runs until all passes are used and a final
   edit is submitted, and the handoff counters show replies used but not the 200-turn cap. So the
   editor never had a reason to submit a final edit, and kept all 19 saves `"final": false`.

### 5.3 Best saved edit vs the last one

No saved edit is best on everything the editor measured, and the editor never named a best edit in
`notes.md` or kept a `best_*` copy, as the prompt asks.

- **Edit 13** had the best addition result it saw (20/20 greedy, but only one problem per N), with
  native 11/12 and large leakage on ordinary English (KL up to ~4 nats on 20 texts, passes 18-20).
- **Edit 19** (evaluated) has the same architecture and parameter count. It trades addition for
  off-task behaviour: 35/60 greedy (58% ± 6 pp), native 36/36. Its English behaviour was never
  measured. The last English check was on edit 18 (≤ 6e-4 on 6 texts, 4 native controls), and edit
  19 changed the output gates after that.

So edit 19 is an intermediate the editor was still debugging, not a chosen result.

## 6. What changes how pilot2 should be reported

1. **Rules: compliant.** Report "0 violations after review (27 automatic findings, all false
   positives)", not 27 flags.
2. **Unfinished run.** Stopped at the turn cap with **23 passes unused** (27/50; the logbook's
   "20/50-ish" is off) and no final edit. The hidden eval in `result.json` (addition 0.85 at
   N = 1-8 / 0.52 at N = 9-20, text KL 0.78) scores a mid-debugging intermediate. It should not be
   read as the editor's final answer, nor as a failed replication of pilot1b on equal terms.
3. **Harness confound.** The early handoffs (section 5.2) cut the editor's working context to
   ~10 turns and cost ~20% of its turns. The input-token figure (17.0M in the logbook) is
   overcounted ~4.5×; the real figure is ~3.8M. pilot1b ran under the same bug. Both write-ups
   should state the real handoff behaviour, not "40 turns / 180k tokens".
4. **Off-task measurement gap.** The editor's English controls were 2 short sentences until pass
   18, then at most 20 texts, and none for edit 19. The hidden KL of 0.78 is consistent with that
   gap. Like pilot1b, the circuit is gated on a leading `A:`; the editor's own `A:`-prefixed English
   control (pass 20, edit 13) had KL 3.97.
5. **Parameters.** 1,036,800 added, 468,480 (45%) nonzero; 0 native entries changed (manifest).

Fixes suggested: compute per-turn usage as the difference of successive thread totals (or read
`last_token_usage` from the rollout) for both the handoff trigger and the input guard; include the
turn cap in the prompt or counters; include the editor's own cell sources (or a `work/` code file)
in the handoff; apply the `audit.py` regex fixes from section 1.
