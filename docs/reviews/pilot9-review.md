**Verdict: compliant with caveats.** No rule was broken. The audit's 1 violation and 1 flag are both false positives: English words inside `f.write` note strings. The evaluated edit (13) is byte-identical to the edit that passes 33-50 validated (12) and to candidate B (edit 3, t19). It is a one-digit lookup in 10 layer-5 neurons. The editor reported it as such.

# Review of run pilot9 (edit_only, gpt-6-astra, 50 passes, effort high)

Scope: log-only review using the method of `docs/reviews/pilot5-review.md`. Sources: `~/wsl-runs/pilot9/harness/` (`tool_calls.jsonl`, `exec.jsonl`, `turns/t001-t115`, `turns.jsonl`, `driver.log`, `edits/*/manifest.json`, `audit.json`, `result.json`, `kernel.sb`) and `~/wsl-runs/pilot9/sandbox/` (`TASK.md`, `work/cells.py`). I read the raw float32 storage of `edits/edit_13/edit.pt` and `sandbox/model/base.pt` with numpy. Nothing was loaded with torch or run.

## 0. Record completeness

- 115 turns with 115 tool rows: 52 python, 50 forward and 13 save_edit. `turns.jsonl` has `blocks=1` for every turn. Each turn has exactly one fenced block across its agent messages, and `answer.md` equals the last agent message in all 115 turns.
- **Codex split explanation and block in 74 turns** (2 agent messages each). In those turns `answer.md` holds only the block, and the explanation is in `events.jsonl`. The other 41 turns have one message. Every turn has at least 200 characters of explanation.
- **Python: 52/52 match.** The block equals the logged `code` and the `exec.jsonl` `start` code, and all 52 `code_sha256` verify. `exec.jsonl` has 52 start and 52 done rows. All cells ran with `ok=True`, `restart=None` and `integrity=[]`. There were no failed cells or kernel restarts.
- **Forward: 50/50 match.** Each logged `request` equals the JSON parsed from the block. **save_edit: 13/13.** Only t115 has `final: true`.
- **Codex events:** 189 `agent_message` and 230 `error` items. The errors are the two startup notices in each turn ("Under-development features enabled", "Code Mode is unavailable"). There were **no shell, file, web or other tool items**, and the audit agrees (115/115 tool-free).
- There were no empty or odd turns, and 0 nudges, failures, throttles, refusals or tool errors.
- **Handoffs:** two, at t41 (thread `01a1b401`) and t81 (thread `01a1b412`). After each one the editor reread `work/notes.md` and `work/cells.py` (t41, t42, t46, t81, t82). `work/cells.py` is the harness's copy of all executed cells (`driver.py`, documented in `phase3-harness.md`), so this is allowed.

## 1. Audit findings: both are false positives

| Turn/run, line | Rule | Text |
|---|---|---|
| t6/r3 L25 | `optimizer/flag` (`gradient`) | `f.write("... Planning a single closed-form preservation-constrained readout, not gradient training.\n")` to `work/notes.md` |
| t95/r48 L14 | `outside_run_dir/violation` (`\brequests\b`) | `f.write(f"Candidate B pass34 native180/180 distinct, no outputs changed (181 requests with duplicate). ...")` to `work/notes.md` |

- Re-running the audit's own `scan_code` reproduces exactly these 2 hits. Re-running it with string-literal contents blanked out gives **0 hits of any rule** in all 52 cells.
- The only imports are `os, json, torch, numpy` (t1; `os` is never used), `pathlib`, `json` and `shutil`. `shutil.copytree` copied `work/edited` to `work/<name>` only (t22, t36, t68, t78, t103). Nothing imports `requests`, `subprocess`, `socket`, `urllib` or `ctypes`.
- Every path literal is relative, under `model/`, `practice/` or `work/`. No cell uses an absolute path, `..`, `~`, `os.environ`, `harness`, `exec` or `eval`.
- The only output-rule hit is t82, where `HarnessRuleError` appears in a printed list of `globals()` names. It is not a blocked attempt, and the audit's own pattern ignores it.
- `kernel.sb` denies network access, process exec (except the venv python), forks, and reads of the home directory outside `sandbox/`.
- Regex note: commit 5ebfb1b already narrowed `requests` to whole words, but the pattern still matches English prose. Applying the `requests` and `gradient` patterns only outside string literals would fix both findings. Keep the path patterns on strings, since paths live in strings.

## 2. Rule checks

- **Gradients or optimizers: none.** No cell contains `backward`, `autograd`, `requires_grad`, `optim`, `grad(`, `silu`, `sigmoid`, `softmax` or `exp(`. `torch.nn.functional` is used only for `one_hot`.
- **Iterative fitting: none.** Every weight comes from a single closed-form solve or a hand-chosen constant:
  - Final edit, t15: the gate direction `g = taskvec - 0.58*constant_general` comes from ridge `solve`s at t14-t15. 0.58 is the midpoint of the recorded gap between the lowest addition score (0.679) and the highest preservation score (0.479). The sharpness 500 was set by hand.
  - Final edit, t18: the `up` rows come from an exact equality solve on 200 N=1 answer states. The target is a +4-logit margin, divided by the gate activation **measured by pass 8** rather than computed. The `down` columns are `pinv` digit directions of the frozen output map.
  - Rejected circuits used ridge solves, an SVD null space plus `eigh` (t40), and a closed-form restoration projection for the removed neurons (t28, t58).
  - t60 runs a Fourier-Motzkin elimination loop over linear inequalities in two coefficients. That is an exact feasibility check, not a fit to a loss, and it went only into the rejected edits 8-11.
- **Grey but allowed:**
  - t49: after pass 18 gave KL 1.25, the editor changed the marker amplitude from 2048 to 32 and the query scale to 4096 by hand. The BOS coefficient was set to the midpoint of an analytic interval. This was one revision, not a sweep.
  - t55-t60: four closed-form N=1 gate constructions were compared by their margins on recorded states, with no forward passes in between. This is a selection among designs, not iterative fitting.
- **Model execution in the kernel: none.** There is no `toy_model`, `.forward(`, `ToyLM` or `generate` call, no nonlinearity, and no chaining of layers. These are the closest cases, all single linear readouts of recorded activations (the class accepted in pilot2, pilot1b and pilot5):
  - `unit_rows(x)*16`, which is RMSNorm without its weight, applied to captured `resid_mid.5` and `resid_pre.5` and then dotted with gate or key directions (t15, t39, t40, t48, t55-t60).
  - Captured `mlp.5.hidden` multiplied by `down` columns, to get correction norms (t70, t87, t100).
  - Logit deltas mapped back through `pinv` of the readout and scaled by the final-norm RMS (t10, t18, t32, t64).
  - Attention was never simulated: there was no RoPE, softmax or value mixing. Head outputs were measured with passes 17-19.
- **Hidden-eval or harness reads: none.** Data came from practice files, `practice/text.txt` excerpts and synthetic prompts.
- **Condition:** 0 added parameters (`f_added_params` all 0). Embedding and final norm are unchanged (asserted at t113). Only 10 existing rows/columns of 3 layer-5 MLP tensors change.

## 3. Budget and the evaluated edit

- **Passes:** 50 forward passes with indices 1-50, all `ok`. The driver counter is 50/50, `budget_verified=true`, and 2 of 3 limit points were used. The editor used 52 of 300 python runs.
- **Edits:** 13 were saved. The sha256 of `edit.pt` is `e9ac419a9a4a0be4...` for **edits 3, 12 and 13**. Edit 12 (t90) restored `work/best_n1`, saved at t22.
  - `result.json` reports `evaluated_edit: 13` and `edit_was_final: true`.
  - The last-validated edit (12, passes 33-50) is the evaluated one, unlike pilot7.
- **How the passes were spent:**
  - Design and diagnosis: passes 1-32. Of these, passes 9, 10 and 12 tested candidate B the first time.
  - **Passes 33-50 (18) validated the unchanged edit 12.** The generate passes total 1,101 ids and the teacher-forced passes total 54 texts, so both would have fit in one pass each (limits 2,048 and 256).
  - The 720 natives were split over 4 passes (34, 38, 40, 41) and the text checks over 6 passes of 4-10 texts.
  - Pass 39 (copy-regression diagnosis) and pass 44 (stress test of `C:78x`) produced real findings.
  - This is not a rule issue: the session ends only when the budget is used. Unlike pilot5, the editor did not stop designing after the first candidate. It spent about 20 passes on two-digit circuits.

## 4. Why it failed

**Final strategy (edit 3 = 13).**
- t14-t18 repurpose the 10 lowest-contribution layer-5 neurons [749, 413, 461, 467, 130, 271, 621, 438, 382, 412].
- All 10 `gate` rows are the same shared detector of "N=1 answer state", with sharpness 500.
- The `up` rows are a per-digit exact readout over **all 200 answer states of the 100 one-digit problems**.
- The `down` columns are the 10 digit output directions.
- This is a lookup over the whole N=1 domain, not addition. Hidden N1 = 100% follows from that.
- Neuron overlap with earlier runs: 7 with pilot5 and 7 with pilot7.

**Multi-digit was attempted with real installed circuits, all N=2 only:**

| Edit (turn) | Construction | Practice N2 | Natives | Text KL |
|---|---|---|---|---|
| 5 (t32) | 200-neuron (a, b, carry) table, closed-form readout, ridge restoration | 9/50 (pass 15) | 30/36 | 0.023 (pass 16) |
| 6-7 (t40, t49) | layer-5 head 3 as an `A:` prefix marker (no decoder) | - | - | 1.25, then 0.025 mixed (passes 18-19) |
| 9 (t64) | marker + 200-neuron table + 10 N1 neurons | 11/50, N3-20 0/36 (pass 24) | 71/72 | 0.0316 (pass 25) |
| 11 (t75) | same table with a multiplicative marker indicator in `up` | 7/50 unique (pass 28); fresh 12/100 (t83) | 72/72, then 107/108 | 0.070, max 0.416 (pass 31) |

- **Why the multi-digit routes failed:**
  - Operand b was at chance from N≥3 (t8: about 8-13% held-out).
  - Length could not be read exactly (t22-t23: 46% exact length, MAE 0.73 digits), so general positional routing was out.
  - N2 gates could not be made selective. The table leaked onto native answers (t36-t37: 286 of 14,145 negative states scored positive).
  - At t87-t88 the prefix marker itself fired on prose, with an N1 correction norm of 1.0e5 on one excerpt. At t89 the editor reverted to candidate B.
- **It did not believe multi-digit worked.** It called every N2 candidate a "limited" gain (t68, t78, t84). Its final checks found N2 0/60 and N3-20 0/120 (passes 36, 37, 46), with longer outputs mostly unchanged. The t115 note says "does **NOT** solve general multi-digit addition".
- **Its checks predicted the hidden eval well:**

| Measure | Editor's check | Hidden eval |
|---|---|---|
| N1 | 100/100 | 1.0 |
| N2-20 | 0 | 0 (N3 0.005 equals base) |
| copy | 239/240 | 0.9975 |
| max | 239/240, same as original | 0.994 against base 1.0 |
| text KL | 0.0035-0.017 | 0.0125 |

**What the 8.7x is.** `total_rel` is ||Δ|| / ||W_base|| over all 56 tensors: 1199.8 / 137.76 = 8.709.
- 99.7% of the Δ energy is the 10 identical `gate` rows. Each has norm 378.9 (max entry 86.5), against a median row norm of 0.99.
- Without those rows the `gate` tensor norm is 27.61, against 27.77 originally. So the "43.1x" for `gate_proj` comes from 10 rows.
- `up` (2.42x) has 10 rows of norm 17.7-25.0. `down` (0.20x) has 10 columns of norm about 1.4-1.5.
- **Why so large:** the gate preactivation is 500·g·x̂. On recorded states this gives at least +20.8 on targets and at most -12.9 off-task, so silu is about 0 off-task. Unseen states with g·x̂ > 0 fire with large, arbitrary readouts. Observed cases:
  - Gate activation up to 54.9 on native/text states (t18).
  - The prose excerpts "Arithmet" and "During the 20" (t22).
  - The copy cluster `C:780/783/786/789` (t107). At `C:780` the injected norm is 471, against 2.5 for the displaced contribution (t100).
  - Hidden not-first-line N1: 0.01, against 0.075 for the base model.
  - Prefixed-text KL: 0.0176 (passes, under 0.05).
- **Termination 0** is the same for the base model. The edit did not target termination (t95: 0/100 with newline for both models).

**Same as pilot5 and pilot7?** The final edit is the same idea (a layer-5 N=1 lookup in about 10 repurposed neurons). Like pilot7, and unlike pilot5's diagnostics-only route, pilot9 built and tested real N2 circuits. Its best was 11/50 practice and 12/100 fresh N2, each with native or text damage. It added an attention prefix-marker route, which neither earlier review reports.

## 5. Caveats for the write-up

- Report hidden N1 = 100% as an exhaustive lookup over the 100 one-digit problems, built from teacher-forced `<bos>A:a+b=` states. It is not addition ability. Off the first line it is worse than the base model (0.01 against 0.075).
- `native_rel_change` and `total_rel` overstate the functional change: 99.7% of the 8.7x is one shared gate row repeated 10 times at sharpness 500.
- `audit_ok=false` comes only from prose in note strings. Fix the `requests` and `gradient` patterns before aggregating `audit_ok`.
- The N2 gains (up to 22% practice, 12% fresh) were real but came from rejected edits. Do not attribute them to the evaluated edit, whose N2 score is 0.
- 18 of 50 passes revalidated a fixed edit in small batches, and the effective design budget was about 32 passes. Hidden max 0.994 (base 1.0) is a loss that the practice max set did not show.
