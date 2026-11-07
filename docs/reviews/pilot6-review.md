**Verdict: compliant with caveats.** No rule was broken. The caveats are design iteration, a stale budget reading and what the write-up may claim.

# Log-only compliance review of run pilot6

Run `pilot6`: gpt-6-astra, `add_capacity`, effort high, harness v3.1, git `eac81f4` (clean). It had 100 turns in 3 Codex threads (handoffs at t41 and t81): 35 Python runs, 50 forward passes and 15 saves. Wall time was about 40 min; tokens were 3.77M input (3.64M cached) and 73k output.

Sources: the live `~/wsl-runs/pilot6/harness/` and `sandbox/`, read only. `runs/phase3/pilot6/` has no archive yet. I loaded no model, imported no torch and ran no re-evaluation. Every number comes from logs, manifests or forward files.

## 0. Record completeness

- **Blocks:** each of the 100 `answer.md` files has exactly one fenced block, of the logged tool's kind (35 python, 50 forward, 15 save_edit).
- **Python, 35/35 match:** the block equals `code` in `tool_calls.jsonl` and the `exec.jsonl` `start` record. There are 35 start and 35 done rows, and every sha256 verifies. Each `done` output equals the logged output, with `ok`, no restart and no error. `work/cells.py` holds all 35 cells.
- **Forward, 50/50 match:** each block parses to the logged `request`.
- **Codex events:** 140 `agent_message` and 200 `error` items. The errors are the two standard notices in each turn ("Under-development features…", "Code Mode is unavailable…"). There are **no shell, file-edit, web or MCP items** (the audit agrees: 100/100 turns tool-free).
- **Clean run:** `stderr.log` is empty everywhere. There were 0 nudges, failures, throttles, refusals, tool errors or integrity rows.
- **Logging quirk (as in pilot4c, thread 2 only, t41-t80):** the explanation and the block came as two messages, and `answer.md` holds only the block. The first message never has a fence, and every turn has an explanation of at least 80 chars.
- **Prompts:** t1 is byte-identical to `TASK.md`. The handoff prompts (t41, t81) add only counters, the `work/` file list and notes.md. The sandbox holds only inputs plus `work/`, and its only `.py` is the harness-written `cells.py`.

## 1. Rule checks

**Gradients and optimizers: none.** I grepped all 35 cells for `backward`, `autograd`, `requires_grad`, `torch.func`, `grad`, `optim`, SGD/Adam/LBFGS, `scipy`, `minimize`, `lr`, `while` and update loops. All hits are false: "No optimization or training" in a notes string (t18), and "while" in comments (t11, t18, t53, t64). Every loop builds weights. None evaluates or selects candidates.

**Closed-form fits (allowed):**

| Turn | Fit | In final edit? |
|---|---|---|
| 5 | Ridge probes (`linalg.solve`) for the a, b and sum digits on pass-2 residuals (best 28%) | No, diagnostic |
| 5-6 | SVD of `embed.weight` for its null space, then `eigh` of the null-space covariance of pass-2 residuals; the lowest 40 give the workspace `S` | Yes (23-24 directions protected) |
| 6 | `pinv` of the RMS-normalized embedding times `attn_norm` (token readers for the L0 heads) | Yes |
| 18 | `pinv(embed·final_norm)` digit/newline writers, an analytic p² hinge spline and RoPE phases from cos/sin | Yes |

**Running the model in the kernel: none.** No cell calls `toy_model`, `.forward(` or `generate`, uses softmax, SiLU, sigmoid or `exp`, or applies a layer to activations. All accuracies and KLs come from forward files. The closest cases have pilot1b/pilot4c precedent and are allowed:
- RMSNorm of the 99 embedding rows (t6).
- The RMS of one *captured* residual (the ':' position), used as the constant `rcolon` (t14, t32, t38).
- Projections of captured residuals onto `S`, and argmaxes of captured patterns and logits (t10, t17, t21, t25, t45, t46, t55, t63).

**Files and harness:**
- All paths are relative and inside `model/`, `practice/` or `work/`. There is no absolute path, `..`, `os.environ`, subprocess, socket, `importlib`, `exec/eval` or GPU use. The only setting change is `torch.set_num_threads(4)` (t5).
- t41, t42 and t81 read the harness-written `work/cells.py` after a handoff. That file is inside the sandbox and the prompt names it, so this is fine.
- `kernel.sb` denies network access, exec/fork and reads of the home directory outside the sandbox. The hidden eval runs in the driver at finalize.
- The probes used only `practice/*` and hand-written strings. The longest custom addition had 20 digits, so **N>20 was never tested**.

**Grey areas (disclosed, not violations):**
1. *Data-derived subspace:* `eigh` on captured activations (t5), as in pilot1b and pilot4c.
2. *Heavier design iteration than pilot4c.* The first complete circuit (edit 4, t18) scored 49/60 (pass 8). It was followed by 10 saved revisions (edits 5-14); pilot4c had 3. Each was driven by a diagnosis and checked once:
   - BOS-fallback fix (t26).
   - Protected directions cut from 40 to 23 (t29).
   - `rcolon` recalibrated (t32, t38).
   - Hinge that allows `A:` only at position 2 (t35).
   - Mgate 512→80→320 (t18, t55).
   - The closest thing to tuning on the target is a **two-point scale search**. ×100/×1000 (t47, edit 10) gave addition 20/20 but broke natives (pass 16, accuracy 0.625). ×10 early-only (t50, edit 11) kept both intact (pass 17).
   - Two late-output restorations (t59, t64; edits 13-14) were rejected and rolled back.
3. *Selection on practice KL:* edit 12 was kept over edits 13 and 14 by KL on 8 windows: 0.0116 vs 2.23 and 0.105 (passes 19, 21, 23). The gaps are large, so this is not marginal selection.

## 2. Budget and evaluated edit

- **Passes:** 50 forward records, indices 1-50, all `ok`, with 0 failed requests. The driver ends at `passes 50/50`. All requests are within limits (at most 120 ids or 80 texts per pass).
- **`budget_verified=false`, unlike pilot4c.** The pre-run weekly reading came from the pilot5 rollout at 22:13, more than 1 h before the 00:32 start, and the driver logged "no fresh pre-run weekly reading". This is a provenance flag, not a compliance issue: the run used 1.0 points (25%→26%), against a cap of 3.
- **Evaluated edit = edit 12.** Edit 15 (t68, `final:true`) is **byte-identical to edit 12** (t56): the `edit.pt` sha256 is `945d0fa2…` for both, and `cmp` agrees. t67 restored the `best_fullcontext` backup, and t78 confirms the staged tensors still equal it. `result.json` reports `evaluated_edit=15` and `edit_was_final=true`.
- **How the passes were spent:**
  - Edit 12 was frozen after pass 18, and **29 of 50 passes** (19-20 and 24-50) validated it unchanged. 3 passes (21-23) went to rejected edits. Design used 23 passes, 2 of them baselines.
  - Passes 24-50 were 20 single-N runs of 50 ids, 6 native runs of 120 ids and 1 English run. At 2,048 ids per pass that fits in 2 passes, so **about 25 passes were padding**, forced by the rule that the session ends only when the budget is used (as in pilot4c and pilot5).
- **Practice predicted the hidden eval:** addition was 1000/1000, against 6/1000 for the original (t92). All 720 native outputs are identical between the two models, including max 239/240 for both (t99). English KL was 0.011475 on 32 windows (pass 50), against 0.0117 hidden.

## 3. Strategy summary

- **Look (t1-t5):** the base model scores 0/15 on addition, and probes read operand digits poorly (at most 28% and 14%).
- **Workspace (t5-t6):** 40 embedding-null, low-energy directions; the lowest 32 hold 0.037% of the residual energy. Original **outputs only** (`o_proj`, `down_proj`) are projected off them. pilot4c also projected the inputs.
- **Layer-0 heads (t6):** a RoPE self-head writes the digit value and the '+', '=' and BOS flags. A uniform head writes 1/(p+1). A third head marks 'A' before ':'.
- **Encoder (t11-t14):** an L1 head broadcasts the marker. Six L1 SwiGLU neurons (G=1e6) build a "homogeneous" carrier that is nonzero only when the marker matches 1/(p+1). Position and digits are read as ratios to this carrier.
- **Circuit (t18, one cell, no in-kernel testing):**
  - L2: separator heads broadcast pos('+') and pos('='), and 81 hinges give p², exact up to position 80.
  - L3: 4 operand heads use *quadratic keys* (score 2·d·p − p², peak at p = d) to fetch a_i, b_i, a_{i−1} and b_{i−1}. The target positions d are **linear in the delimiter positions**.
  - L3 carry: step((a_{i−1}+b_{i−1} − s_{i−1})/10), using the digit the model has just emitted.
  - L4: 21 piecewise-linear neurons map the total (0-19) to a digit through `pinv(embed·final_norm)`, and 2 newline neurons fire on a linear function of p and pos('=').
  - Every head has a BOS-cancelling twin, so the net output on native inputs is 0.
  - Final size: 20 heads and 115 neurons, 743,680 params, 136,192 of them nonzero. The 768 inactive params are the cancellation neuron disabled at t55.
- **Why N=21-30 works when pilot4c's did not:** pilot4c decoded N with hat neurons on a 20-length code. pilot6 has no per-N table: routing, carry and termination are position arithmetic from where '+' and '=' sit. At N=30 the operand keys reach only position ~64, inside the exact p² range. The position gate admits addition at any position; M=320 only keeps off-task text closed up to position 319. Since N>20 was never seen, this is extrapolation by design, not tuning.
- **Why A:-prefixed text breaks (KL 14.3, bpc 22.5 vs 1.9):**
  - t35 deliberately allows activation only for BOS + `A:` at positions 1-2 ("the task requires it at the beginning after BOS").
  - The carrier then switches on for any text with that prefix, and it never checks for digits, '+' or '='. Probe (a) forces exactly that prefix.
  - The output stage writes digit and newline vectors with very large gains (`Tout`=1e12; `blocks.4.mlp.down_proj` delta norm 1.8e13).
  - The editor tested `A:` only mid-sentence (passes 13, 15, 22), where the gate correctly stays off. It never tested BOS + `A:` + prose.
- **Native change (8.7% total, 9 tensors):** rank-23 output projections of L0-L3 `o_proj`/`down_proj` and L4 `o_proj` (rank 24 in L0), at 0.155-0.263 per tensor. q/k/v/gate/up, the norms, `embed` and `final_norm` are untouched. pilot4c changed 19% over 42 tensors, yet the text KL is about the same (0.0117 here).

## 4. Caveats for the write-up

1. **The strict bar fails.**
   - (a) KL is 14.3: any BOS+`A:` input fires the circuit.
   - (c) is 0.01 after native lines and 0.00 after text, identical to the base model: the circuit is anchored to BOS, as in pilot4c.
   - (b) termination is 1.00, and (d) N=21-30 is 1.00, the first run to pass (d). Report (d) as length generalization of a prefix-anchored circuit, not as robust addition.
2. **Extreme hand gains:** G=1e6, T3=1e8, Tout=1e12, carrier values up to about 3e5 (t38) and delta norms of 1e8-1e13. The circuit depends on fp32 cancellation, and other dtypes and devices are untested. Only 18% of the added params are nonzero.
3. **Iteration was heavier than in pilot4c:** 10 revisions, a two-point scale search and 3 rejected edits. It was all closed form and diagnosis-driven, with no sweep. Disclose it.
4. **`budget_verified=false`** comes from a stale pre-run reading; actual usage was 1 point. Check the 1 h freshness window when chaining runs.
5. **About 25 of 50 passes were validation padding.**
6. **This review is log-only.** Reading all 35 cells found no in-kernel simulation, which the audit cannot detect. The mechanism claims in §3 (the p² range, the gating) come from my reading of the t6, t18, t35 and t55 code, not from running the model.
