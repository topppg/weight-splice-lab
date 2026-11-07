**Verdict: compliant with caveats.** No rule was broken, and the audit's 0/0 holds up on re-reading. The caveats are about interpretation: the evaluated edit (15) is not the edit that the last 10 passes validated (16), and the result is again a one-digit lookup.

# Review of run pilot7 (edit_only, gpt-6-astra, 50 passes, effort high)

Scope: log-only review following `docs/reviews/pilot5-review.md`. Sources: the live `~/wsl-runs/pilot7/harness/` (`tool_calls.jsonl`, `exec.jsonl`, `turns/t001-t134`, `turns.jsonl`, `driver.log`, `edits/*/manifest.json`, `audit.json`, `result.json`, `kernel.sb`) and `sandbox/` (`TASK.md`, `work/notes.md`). There is no `runs/phase3/pilot7/` archive yet. The `edit.pt` files and `model/base.pt` were read as raw float32 storage with numpy. Nothing was loaded with torch or run.

## 0. Record completeness

- 134 turns and 134 tool rows: 68 python, 50 forward and 16 save_edit. Each turn has exactly one fenced block across all of its agent messages, and the block kind matches the logged tool.
- **Python: 68/68 match.** The `answer.md` block equals the logged `code` and the `exec.jsonl` `start` code, every `code_sha256` verifies, and the `done` output equals the logged output. `exec.jsonl` has 68 start and 68 done rows. There were no restarts and `integrity=[]`. Two cells raised ordinary exceptions and changed nothing: a float32/float64 dtype mismatch at t43 and a missing `import math` at t95. Each was rerun in the next cell.
- **Forward: 50/50** requests equal the JSON in `answer.md`. **Save: 16/16** `final`/`note` values equal the manifests.
- **Codex events:** 228 `agent_message` items and 268 `error` items. The error items are the two standard startup notices per turn. There are **no shell, file-edit, web or MCP items**, and all 134 turns are tool-free. `stderr.log` is empty everywhere. There were 0 nudges, failures, throttles or refusals. The t1 prompt is byte-identical to `TASK.md`.
- **Handoffs:** three, at t41, t81 and t121, giving 4 threads. The restart prompts add only counters, the file list and `notes.md`, plus "Latest saved edit: 16. Submitted final edit: 15" at t121.
- **Logging quirk (t41-t134, same as pilot4c):** Codex sent the explanation and the block as two messages, so `answer.md` holds only the block. The 94 explanations exist only in `events.jsonl`. All of them are at least 80 characters.

## 1. Rule checks

**Clear violations: none.**
- **Gradients and optimizers:** no `backward`, `autograd`, `requires_grad`, `torch.func`, `optim`, `scipy` or `lr=` in any of the 68 cells. The only hit is the comment "without optimizing any weights" (t27).
- **Model execution:** no `toy_model`, `.forward(`, `.generate(`, `load_state_dict` or SDPA. No attention weight was ever loaded (`q/k/v/o_proj` never appear), and no `silu`, `sigmoid` or `exp` was applied to activations.
- **Files:** every path is relative under `model/`, `practice/` or `work/`. There is no absolute path, `..`, `os.environ`, subprocess or socket. `kernel.sb` denies network access, exec, fork and reads outside `sandbox/`.
- **Hidden eval:** it ran in the driver at finalize, and the editor never referenced it. English KL was measured only on `practice/text.txt` slices.

**Grey areas, all with precedent and all judged allowed:**
- **RMSNorm on captured residuals.** RMSNorm times `mlp_norm`/`attn_norm` was applied to captured `resid_*` arrays, followed by single linear readouts: probes, gate pre-activation margins and null-space residuals (t8, t13, t26, t29, t44, t68, t115). This is the same class as pilot5 t12/t20 and pilot1b t8.
- **Boost amplitudes.** These were set from recorded teacher-forced logits and the RMS of `resid_final`, through `pinv(embed*final_norm)` (t18, t20, t91). This is a linear map of recorded values, the same as pilot5's dual directions.
- **Leakage and reconstruction decompositions.** These are captured `mlp.5.hidden` times `down_proj` columns (t43-t45, t62, t89, t114). They are linear readouts of forward-tool outputs, not simulations. The 1,282 residual norm at the copy failure (t114) was computed this way.
- **A design-test loop of 16 saves.** Hand constants were revised after diagnoses:
  - Marker widths: a data-derived safety margin `min(.15, 0.45*min distance)` at t69, then the planned 0.15 restored at t74.
  - Lookup band width: 0.002 to 0.0005 at t116. This was one geometric step: the measured collision distance was 0.00186 and the float32 intended residual is at most 1.5e-7.
  - There is no sweep or select-by-score loop in any cell. Each candidate was a new mechanism or a one-step fix. This is not iterative fitting to a loss.
- **Reads of its own records.** t121-t122 read `work/cells.py` (the harness-written record of the editor's own code) and list `globals()` names. Both are inside `work/`, and neither touches the guards.

**Fine:** every write into weights came from one closed-form solve: ridge `linalg.solve`, `lstsq`, `pinv`, SVD null spaces or `eigh`. All gates and codes were hand-set constants (triangular SiLU triplets `[1,-2,1]`, sharpness 32, icosahedral digit code). Condition `edit_only` holds: `added_params=0`, and `embed` and `final_norm` are untouched.

## 2. Budget and the evaluated edit

- **Passes:** 50/50 at indices 1-50, all `ok`, `budget_verified=true`, 2.0 of 3 limit points used, 68 of 300 python runs.
- **Edits:** 16 saved, with two byte-identical restorations. Edit 12 equals edit 4 (sha256 `0ebbebd2...`), and **edit 15 (t103, `final: true`) equals edit 13 (t92)** (sha256 `45029298...`). `result.json` gives `evaluated_edit: 15` and `edit_was_final: true`.
- **The evaluated edit is not the one the editor ended on.**
  - At t117 the editor saved edit 16 (bands narrowed fourfold) as non-final. It wrote that it "will replace the submitted edit only after matched … checks" (t117).
  - Its notes call edit 16 "CURRENT BEST" (t126). At t134 it says "The final submission will …".
  - Passes 41-50 all ran on edit 16, because "edited" means the latest saved edit. The session then stopped at pass 50 because a final edit (15) already existed. This follows the documented stop rule (`phase3-harness.md` §Done), and the t121 restart prompt and the "passes left" counters warned of it.
  - So the hidden eval scored edit 15, which has a known 1/360 native collision (copy_n06_016, t112). The 10 passes of validation describe edit 16, which was never evaluated.
- **How the passes were spent:**
  - Design used 32 passes (P1-P32), against 12 in pilot5.
  - The audit of edit 15 used P33-P40. The audit of edit 16 used P41-P50.
  - The split was partly padding: six native batches of 60 (P34-P39), three of 240 (P42-P44) and two addition batches (P46-P47) would each have fit in one pass (limit 2,048). That is about 8 padding passes, against 37 in pilot5.
  - Unlike pilot5, this audit found a real regression (P39) and produced a fix.

## 3. Why it failed

**Final strategy (edit 13 = 15, t91).** It repurposes 12 layer-5 neurons [121, 130, 293, 382, 413, 438, 452, 461, 471, 557, 621, 749]. Eight of them are also among pilot5's 10 repurposed neurons, which suggests both runs picked the same low-impact units.
- Each group of 3 neurons is a triangular SiLU band, so it fires only when the normalized input lies within 0.002 of a hyperplane. The hyperplane normal lies in the null space of the **100 teacher-forced answer states of all one-digit problems**, one group per phase: units or carry.
- `up` carries a `lstsq` readout of a 3-D icosahedral digit code (9 neurons) or a signed carry code (3 neurons).
- `down` writes the digit's dual output directions.
- The 756 retained neurons reconstruct the removed features with one ridge solve.
- The result is an exact lookup over the whole N=1 domain, not addition. Hidden N1 = 100% follows from that.

**Multi-digit attempts were real installed circuits, unlike pilot5's diagnostics.** All targeted N=2 only, and none generalised to N>=3 by design:

| Edit (turn) | Construction | Practice N2 | Native losses (of 36) | Mixed KL |
|---|---|---|---|---|
| 5 (t37) | 200-neuron (a, carry) x b table, `lstsq` readouts from 192 N2 examples | 17/50 | 5 | 0.52 |
| 6 (t47) | same, with a signed task gate | 7/50 | 4 | 0.20 |
| 7 (t59) | 222-neuron Fourier-product mod-10 circuit | 20/50 | 32, N1 down to 8/10 | 1.16 |
| 8 (t64) | same, preservation-regularized | 0/50 | 2 | 0.008 |
| 11 (t80) | layer-4 phase markers + Fourier circuit | 8/50 | 4 | 0.32 |
| 14 (t99) | 51-neuron harmonic circuit | 4/50 | 19 | 0.62 |

- **Why it gave up:**
  - The N1 domain could be enumerated exactly, but N2 states could not. N2 gates therefore had to be broad category detectors, and they leaked onto native answer states (t40, t45, t53).
  - Exact-interpolation readouts extrapolate with off-task norms of 6-10 (t63). Regularizing them killed second-digit decoding (44%, t97).
  - Earlier diagnostics: RoPE-phase routing was not linearly recoverable (length MAE about 1 digit, t26-t27). Max-style prefixes helped only the selected operand (t56). The layer-4 marker broke the N1 gates (t73).
  - It stopped at t102: "no general multidigit solution achieved".
- **It did not believe multi-digit worked.**
  - The final note says "This does NOT implement general multi-digit addition".
  - Its edit-16 checks found 0/190 at N2-20 (P46-P47), 100/100 exhaustive N1, and 719/720 natives identical to the original.
  - The hidden eval agrees for edit 15: N1 1.0, N2-20 0 (N3 0.005 equals base), natives 0.999/1.0/0.998.
  - It underestimated text KL: its edit-15 estimate was 0.0002-0.0005 on 8-text batches (P30, P40), and the hidden value is 0.0019.

**What the 228x/1134x means.** It is 12 rows of gate weights, not the whole model. Measured from `edit.pt` against `base.pt`:
- **`gate_proj`, 1134x:**
  - The 12 rows equal `-normal*(32/0.002) ± 32*ball`, with normal norms 0.579 and 0.534. That gives **row norms 8,531-9,275**, against a median row norm of 0.99.
  - The other 756 rows have norm 27.58, against 27.77 originally. The max entry is 2,263.
  - The 1/width factor is the cause: edit 16's 0.0005 band gives 4,536x.
- **`up_proj`, 25x:** the 12 readout rows have norms 27-254.
- **`down_proj`, 0.51x:** the 12 replaced columns (Δ 13.7) plus the reconstruction on the 756 retained columns (Δ 1.42).
- **Total 228x:** this is the mean over the 3 changed tensors, dominated by `gate_proj`. The other 53 tensors are unchanged.

**Comparison with pilot5:** the same idea, a one-digit lookup in repurposed layer-5 neurons, built differently and better isolated.
- pilot5 interpolated 10 gate rows with min-norm `lstsq` (26,529x) and added a layer-0 marker that touched 42 tensors.
- pilot7 uses geometric null-space bands and touches only 3 tensors.
- On the robustness checks:
  - Prefixed-text KL is 0.0021, against 0.016 for pilot5.
  - Not-first-line accuracy equals the base model exactly (N1 0.075), so the lookup is inert there rather than harmful.
  - Termination is 0%, the same as the base model.

## 4. Caveats for the write-up

- Report hidden N1 = 100% as an exhaustive lookup over the 100 one-digit problems, built from teacher-forced `<bos>A:a+b=` states. It works only on the first line and adds nothing elsewhere.
- **The evaluated edit 15 is not the editor's final preferred edit 16.** Passes 41-50 and the late notes describe edit 16. The hidden native losses (copy 0.9992, max 0.9983) are consistent with the collision that edit 16 targeted, but edit 16 was never scored, so this is unverified. Consider ending the session on an explicit final save after the last pass, or warn the editor when the last pass will end the run while a newer non-final edit exists.
- pilot7 did attempt multi-digit (six installed N2 circuits, best 20/50 at the cost of native collapse). Do not describe it as "diagnostics only" like pilot5.
- The weight-change metrics overstate the functional change: 1134x is 12 band-gate rows whose scale is set by 1/width.
- About 32 passes went into design and about 8 into split-batch padding. The padding is forced by the "end only when the budget is used" rule, as in pilot5.
- There is no `runs/phase3/pilot7/` archive yet. This review used the live directory, so archive it and check byte-identity before sign-off.
