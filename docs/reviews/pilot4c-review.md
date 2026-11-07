**Verdict: compliant with caveats.** The logs show no rule violation. The caveats concern what the write-up claims, not compliance.

# Log-only compliance review of run pilot4c

Run: `pilot4c` (gpt-6-astra, `add_capacity`, effort high, harness v3.1, git `030b54e`, clean). 81 turns
in 3 Codex threads (handoffs at t41 and t81): 25 Python runs, 50 forward passes, 6 saves. Wall time
about 33 min. Sources: all of `runs/phase3/pilot4c/`, and the live `~/wsl-runs/pilot4c/` (read only:
listing and hashing). No model load and no re-evaluation; every number comes from logs or forward files.

## 0. Record completeness

- Each of the 81 `answer.md` files has exactly one fenced block, and its kind matches the logged
  tool: 25 python, 50 forward, 6 save_edit. For all 25 Python turns the block equals the `code` in
  `tool_calls.jsonl` and the `start` record in `exec.jsonl` (sha256 checked), and the `done` record's
  output equals the logged output. 0 mismatches. The 50 forward blocks parse to the logged
  `request`. `work/cells.py`, which the harness writes, contains all 25 cells.
- Archived logs (`exec`, `turns`, `driver.log`, `audit`, `state`, `kernel.sb`, `result`) are
  byte-identical to the live `harness/` copies.
- `events.jsonl` contains no Codex tool items in any turn (no shell, file edit, web or MCP). Besides
  `agent_message` the only items are the two standard notices in every turn ("Under-development
  features…", "Code Mode is unavailable…"). `stderr.log` is empty everywhere, and there are no
  failures, throttles, refusals, nudges or kernel restarts. All 25 cells ran without a traceback,
  `HarnessRuleError` or sandbox denial.
- **Logging quirk (t41-t80, thread 2):** Codex sent the explanation and the block as two
  `agent_message` items. `answer.md` (`-o`, last message) holds only the block, so those 40
  explanations exist only in `events.jsonl`. The first message never contains a fence.
- The t1 prompt is exactly `TASK.md`, and the handoff prompts (t41, t81) add only counters, the
  file list and notes.md. The sandbox is fresh (created 20:46) and holds only inputs plus the
  editor's `work/` files, none of them `.py`.

## 1. Rule checks

**Gradients and optimizers: none.** All 25 cells were searched for `backward`, `autograd`,
`requires_grad`, `torch.func`, `optim`, `SGD/Adam/LBFGS`, `scipy.optimize`, `lr=`, `while` loops and
update loops. The only hits are false: "optimization" in a notes string (t30) and "while" in comments
(t19, t75). No cell contains a loop that evaluates candidates and selects one.

**Closed-form fits (allowed):**

| Turn | Fit | In final edit? |
|---|---|---|
| 8 | ridge probes (`linalg.solve`) for length, token and operand on forward-3 residuals | no, diagnostic |
| 9 | SVD of `embed.weight` to get its null space, then `eigh` of null-space covariance of forward 2/3 residuals; lowest 35 eigvecs → reserved channels `R` | yes (first 24 columns from t20) |
| 14 | `pinv` of RMS-normalized embedding (token readers); `pinv(embed*final_norm)` (digit writers) | yes |
| 19 | piecewise-linear spline coefficients from analytic cos/sin targets (no model data) | yes |
| 30 | ridge from retained to projected-away channels on forward-8 residuals (R² probe) | no, not applied |

**Running the model in the kernel: none.** No cell chains layers, applies softmax/SiLU attention or
an MLP to activations, or computes logits from weights. All accuracies and KLs come from
forward-tool files. Borderline cases, all with pilot1b precedent: one RMSNorm applied to *captured*
residuals (t8, t30) and to the 99 embedding rows (t14), and diagnostics that read captured channels,
patterns and logits (t17, t26). These are allowed.

**Files and harness:** every path the editor opens is relative and inside `model/`, `practice/` or
`work/`. No absolute path, `..`, `os.environ`, subprocess, socket, `importlib`, `exec/eval`,
introspection of guards or GPU use. The only environment change is `torch.set_num_threads(4)` (t5).
The editor never touched harness files and could not have read hidden-eval data: the eval runs in the
driver at finalize and `kernel.sb` denies reads outside the sandbox. The editor measured English KL
only on `practice/text.txt`.

**Grey areas (disclosed, not violations):**
1. *Data-derived subspace:* `eigh` on forward-captured activations (t9). Closed form, as in pilot1b.
2. *Design iteration on the target:* 3 diagnosis-driven revisions, each checked once, after 1
   isolation trial (edit 1): 35→24 channels (t19-t20, for KL), ×10/÷10 rescale (t27, after
   forward 10), newline neuron (t31). There was no sweep and no re-save with varied constants. About
   20 hand constants are each set once (`eps=.03`, `sharp1=2e6`, gains 4000-40000, `gain5=1e10`, ...).
3. *Selection on practice data.* Edit 3 replaced edit 2's 35-channel design partly because of KL on
   11 texts (0.0106 vs 0.0109/0.015, forwards 6/8). These are small samples and one decision. It is
   less iteration than pilot1b (about 6) or pilot2 (about 13).

## 2. Budget and evaluated edit

- There are 50 forward records with consecutive indices 1-50, all `ok`, and no failed requests. The
  driver log ends at `passes 50/50`, and `state.passes=50`, `budget_verified=true`. Every request is
  within limits (at most 144 items per pass, `max_new_tokens` at most 24). 46 passes used `both`.
- Edits 1-5 were saved non-final at t11, t15, t21, t28 and t32. Edit 6 (t81, `final:true`) is
  **byte-identical** to edit 5: `edit.pt` sha256 is `641d7ff6…` for both. Its manifest tensor
  stats equal edit 5's, and t66 confirms the staged tensors equal `best_3` (`differences=[]`).
  `result.json` reports `evaluated_edit=6`, `edit_was_final=true`, and the hidden eval loaded
  `harness/edits/edit_6/edit.pt`.
- Edit 5 was frozen from pass 12, so **39 of 50 passes** validated an unchanged edit, and the design
  used only 11 passes. The editor never read pass 50 (all 100 one-digit pairs). From
  `work/fwd_50_edited.jsonl` I count 100/100 exact answer+newline.

## 3. Strategy summary

- **Look (t1-t8, passes 1-3):** the base model gets 0 additions. Existing heads retrieve operands
  poorly (best top-1 attention 45%/16%), and ridge probes cannot read length or operand digits
  (t8). The editor decided to build an explicit circuit (notes, t9).
- **Isolate (t9-t10, edit 1):** it chose 35 embedding-orthogonal, low-energy residual directions
  (0.03% of residual energy). It projected them out of every native block's inputs (`(W·g)P/g`
  for q/k/v/gate/up) and outputs (`P·W` for o/down). This alone keeps native 36/36, with mixed KL
  0.014 (pass 5).
- **Circuit (t14, edit 2, one cell, no in-kernel testing):** 7 heads and 231 neurons. In L0, a
  self-token head writes the digit value, a constant and a '+' marker, and a uniform head writes
  BOS/'A' counts, giving inverse position. In L1, 60 spline "hat" neurons decode N at '+'. In L2, a
  head broadcasts the length code from '+'. In L3, 4 RoPE retrieval heads have query phases set
  analytically to the offsets of a_i, b_i, a_{i-1} and b_{i-1}. In L4, saturating step neurons
  compute `a_i+b_i+0.1(a_{i-1}+b_{i-1}-s_{i-1})`, a carry identity that rounding makes exact. In L5,
  80 window neurons decode the rounded total (0-19) to the digit token via `pinv(embed·final_norm)`.
  The first check gave 60/60 across N=1-20, native unchanged (pass 7).
- **Compress (t19-t20, edit 3):** 9 Fourier length features, 24 channels: 54/60 (pass 9), from
  phase/cancellation errors (t24-t26). **Repair (t27, edit 4):** an equivalent ×10/÷10 rescale of 5
  early channels gave 60/60 (pass 11).
- **Terminate (t31, edit 5):** the a_{i-1} head also marks '+', and one gated L5 neuron writes '\n'
  after N+1 digits. Final size: 7 heads, 194 neurons, 378,368 params (only 177,152 nonzero).
- **Validate (passes 12-50):** 1000/1000 practice additions with newline, all 720 native outputs
  identical, English KL 0.0122 on 32 excerpts, and 48 carry/extreme stress cases.
- **Why native weights moved by about 19%:** the rank-24 projection, not the circuit. All 42
  matrices lose their components along the reserved directions (0.09-0.26, total 0.195): the weights
  carry real mass there, the activations do not. Unchanged: 12 norms, `embed`, `final_norm`.

## 4. Caveats for the write-up

1. **The strict bar fails.** Robustness (c) not-first-line is 0.01 after native lines and 0.0006
   after text (the threshold is 0.95). Addition works only as the first line after BOS, because the
   circuit counts positions from BOS. N=21-30 scores 0.00 (20-length code). Only (a), prefixed-text
   KL 0.0147, and (b), termination 1.00, pass.
2. **"Added a capability" understates the edit.** All 42 native matrices were rewritten by a
   rank-24 projection, which is probably the main source of text KL. Edit 1 (projection only, no
   circuit) already had KL 0.014 mixed and 0.023 English. Report the edit as added capacity plus a
   native projection.
3. **Very large hand weights** (down-proj gains about 10⁵-10⁶), tuned for fp32 CPU. Other dtypes
   and devices are untested. **About half the added parameters are zero** (177,152 of 378,368
   nonzero). Design iteration (§1) is light but not zero.
4. **Convergent design, not leakage.** The approach (embedding-null low-energy workspace, `pinv`
   readers, analytic RoPE kernels) closely matches pilot1b, pilot4 and pilot4b, which suggests a
   stable prior of this editor. The logs show no channel between runs: the sandbox is fresh, the
   prompt is TASK.md only, memories and history are disabled, and the editor has no tools. The
   working circuit came in a single cell (t14) at turn 14, after 3 forward passes.
5. Log-only review: no re-evaluation was run. Reading all 25 cells found no in-kernel simulation,
   which the audit cannot detect.
