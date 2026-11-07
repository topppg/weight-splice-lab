**Verdict: compliant.** No rule was broken. All 10 automatic violations are false positives from the `requests\b` pattern. The edit is a one-digit lookup table, and the editor reported it as such.

# Review of run pilot5 (edit_only, gpt-6-astra, 50 passes, effort high)

Scope: log-only review following `docs/reviews/pilot2-audit.md` and `docs/reviews/pilot1b-audit.md`, but lighter. Sources: `runs/phase3/pilot5/` (`tool_calls.jsonl`, `exec.jsonl`, `turns/t001-t080`, `driver.log`, `edits/*/manifest.json`, `audit.json`, `result.json`, `notes.md`, `TASK.md`) and the read-only `~/wsl-runs/pilot5/` (`harness/edits/*/edit.pt` hashes, `harness/kernel.sb`, the raw float32 storage of `stage_snapshot/.../blocks.5.mlp.gate_proj.weight.pt`, read with numpy). Nothing was loaded with torch or run.

## 0. Record completeness

- The run had 80 turns with 80 tool rows: 26 python, 50 forward and 4 save_edit. Each turn has exactly one fenced block across all its agent messages, and `answer.md` equals the last agent message in all 80 turns.
- **Python: 26/26 match.** The block in `answer.md` equals the logged `code` and also the `exec.jsonl` `start` code, and every `code_sha256` verifies. `exec.jsonl` has 26 start rows and 26 done rows. All cells ran with `ok=True`, `restart=None` and `integrity=[]`; there were no failed cells and no kernel restarts.
- **Forward: 50/50 match.** Each logged `request` is the JSON parsed from `answer.md`.
- **Codex events:** 120 `agent_message` items, plus 160 `error` items. The error items are the two startup notices per turn ("Under-development features enabled", "Code Mode is unavailable ... fail closed"). There were **no shell, file-edit, web-search or other tool items** (the audit agrees: 80/80 turns tool-free).
- **No empty or odd turns.** Every turn has at least 80 characters of explanation. There were 0 nudges, failures, throttles or refusals. One handoff happened at turn 41 (new thread `01a1b2b5`), and work continued from `notes.md`.

## 1. Audit findings: 10 of 10 are false positives

All 10 are `outside_run_dir/violation` hits. Each comes from `requests\b` in `audit.py`. That pattern has no left word boundary, so it matches inside identifiers and filenames:

| Turn/run | Lines | Text |
|---|---|---|
| t32/r17 | 2, 7, 8, 11, 14, 15 | the list variables `validation_gen_requests` and `validation_text_requests` (built, appended to and printed) |
| t32/r17 | 12, 13 | `Path("work/validation_*_requests.json").write_text(...)`: writes to the relative path `work/` |
| t41/r19 | 4 | a list of those two filenames, then read back from `work/` |
| t51/r21 | 13 | `json.loads(pathlib.Path("work/validation_text_requests.json").read_text())` |

- Re-scanning all 26 cells with the audit's own `scan_code` and the pattern changed to `(?<![\w.])requests\b` gives **0 hits of any rule**.
- I found **no real network, subprocess or out-of-sandbox access.** The only paths in the code are relative ones under `model/`, `practice/` and `work/`. There is no import of `os.environ`, `subprocess`, `socket` or `urllib`, and no absolute path, `..`, `harness`, `toy_model` or `exec`/`eval` call.
- Some printed output, such as `PermissionError` or `HarnessRuleError`, would trigger the audit's output rules. None of it appears.
- `kernel.sb` denies network access, process execution and forks, and denies reads of the home directory outside `sandbox/`.

Fix suggestion for `audit.py`: change `requests\b` to `(?<![\w.])requests\b`.

## 2. Rule checks

- **Gradients or optimizers: none.** No `backward`, `autograd`, `requires_grad`, `torch.optim`, `torch.func`, `grad`, `silu`, `sigmoid` or `softmax` appears in any cell.
- **Iterative fitting: none.** Every weight comes from a single closed-form solve or a hand-set constant:
  - t8: a covariance-weighted `linalg.solve` for `down_proj` (candidate 1).
  - t16: a projection `I - cc^T` plus an SVD/`pinv` reparameterization of layer-0 head 1, and a `lstsq` '+' detector.
  - t20: a min-norm `lstsq` for 10 gate rows, a ridge `solve` to compensate the replaced neurons, and `pinv` dual directions for the down columns.
  - The hand constants were boost +4 (t8), +8 and -30 (t20), and `marker_scale=1`. The +4 to +8 change was not a response to a score: candidate 1 already scored 20/20 on one digit and was rejected for collateral damage.
  - There were three candidates, each a redesign after a diagnosis, with no sweep. The candidates after candidate 1 did not refit it (t12 note).
- **Ridge probes (t12, t27, t30, t31):** diagnostics on captured activations. None was written into weights.
- **Model execution in the kernel: none.** No `toy_model`, `.forward(` or `.generate(` call, no attention or MLP nonlinearity, and no chaining of layers. These are the closest cases:
  - t12 and t20: RMSNorm times `mlp_norm` applied to `resid_mid.5` captured by forward passes 3 and 7. t20 prints the `lstsq` constraint residual, a single linear readout of recorded activations. This is the same class as pilot2's "closest case" and pilot1b t8, so it is allowed.
  - t15 and t16: RMSNorm of the 99 embedding rows, then the layer-0 OV map `En@v.T@o.T` (SVD). This is weight analysis of the token-only layer-0 input, the same as pilot1b t9. Allowed.
- **Hidden-eval or harness reads: none.** t1 printed `model/model.py`, which is allowed. The data used was practice files plus synthetic prompts generated with fixed seeds (t25, t28).
- **Condition:** 0 added parameters (`f_added_params` all zero), and `embed`/`final_norm` are unchanged (asserted at t79). The edit repurposes 10 existing layer-5 neurons, which edit_only permits.

## 3. Budget and the evaluated edit

- **Passes:** 50 forward passes with indices 1-50, all `ok`, and none failed. The driver counter is 50/50. `budget_verified=true`, 2 of 4 limit points were used, and the editor used 26 of 300 python runs.
- **Edits:** 4 were saved, at t9, t17, t21 and t80.
  - Edit 4 (t80, `final: true`) is **byte-identical to edit 3** (t21): the sha256 of `edit.pt` is `7094e7b4...` for both, and `cmp` reports no difference.
  - `result.json` reports `evaluated_edit: 4` and `edit_was_final: true`, so the evaluated edit is candidate 3.
- **How the passes were spent:**
  - Design and diagnosis used passes 1-12.
  - At t32 the editor concluded that multi-digit addition was out of reach. It then spent **passes 13-50 (38 of 50) validating the unchanged edit 3**: 20 generate passes of about 86 ids each, and 18 teacher-forced passes of 8 texts each.
  - Both sets would have fit in one pass each (limits: 2,048 ids and 256 texts per pass).
  - This is not a rule issue. The session ends only when all 50 passes are used, so splitting the validation was the way to end it. As a result, only about 12 passes went into design.

## 4. Why it failed

**Strategy (edit 3):**
- t15-t16 (edit 2): reserve one residual direction `c`, orthogonal to the embedding and output maps (a null space of dimension 58). All block inputs are projected away from `c` and all outputs preserve it. Layer-0 head 1 is rewritten to keep its top 31 OV components and to use the 32nd as an exact '+' detector that writes `c`.
- t20 (edit 3): repurpose the 10 layer-5 neurons that contribute least ([749, 461, 413, 467, 130, 452, 557, 438, 112, 621]):
  - `up` reads only the marker.
  - `gate` is a min-norm `lstsq` interpolating all 200 answer states of **all 100 one-digit problems**: the correct digit's neuron gets a large target, the others get -30.
  - `down` is the frozen-output dual direction of each digit.
- The result is a lookup over the entire N=1 domain, not an addition circuit. Hidden N=1 = 100% follows from that.

**Multi-digit attempts were diagnostics only, and all were negative:**
- t26-t27 (pass 11):
  - Aligned attention to operand b is weak, and attention to both operands fades by N≥12.
  - Held-out probes at `resid_mid.5` read a at 43%, b at 9.9% and carry at 76.1%.
- t29-t30 (pass 12): two-digit sum probes fall within half a digit only 44% and 34% of the time.
- t31: the RoPE pointer-phase probe has a mean error of 1.47 positions for a and 0.65 for b.
- t32: the editor stopped there: "does not justify installing the proposed routing circuit".

**It did not believe multi-digit worked:**
- Its own practice sweep (t57) showed N1 50/50 versus 6/50 for the original, and **0/950 for N2-20** for both models.
- Natives were 720/720 identical, and prose KL was 0.00099 (t79).
- The t80 submission says "**partial result**, not a successful multi-digit solution".
- Its checks predicted the hidden eval almost exactly: N1 1.0, N2-20 0, natives 1.0, text KL 0.00093.

**Where the 26,500x change in `gate_proj` comes from:** this is not RMSNorm or SwiGLU saturation. In SwiGLU, `h = silu(gate·x) * (up·x)`.
- The `up` input of the 10 neurons is the marker coordinate after RMSNorm, which is tiny on addition states: 0.0045-0.062 (t20 output).
- The gate targets were set to `amount/marker`, so the gate pre-activation has to carry the whole magnitude. silu is close to linear for large inputs and close to 0 at -30.
- The `lstsq` interpolates 200 states in 256 dimensions on ill-conditioned inputs (smallest singular value 0.05, t12). This gives **gate solution norm 736,693, max entry 56,501** (t20).
- The tensor read from the snapshot confirms it:
  - Those 10 rows have norms of 1.9-2.8e5, against a median row norm of 0.99.
  - Without them the tensor norm is 27.6, while the original norm is 27.8.
  - So the "26,529x" is 10 rows, not the whole tensor.
- Off the addition format, only a marker leakage of at most 6.2e-6 keeps these neurons quiet. On any '+' input the huge, arbitrary gate values fire:
  - Prose containing '+' has KL 2.6-2.9 (t24).
  - Hidden N1 accuracy when the problem is not the first line is 0.01, against 0.075 for the base model.
  - The prefixed-text KL is 0.016, against 0 for the base (threshold 0.05).
- The editor flagged this risk at t21.

**Termination 0%** is the same for the base model and is not caused by the edit: in t38, `A:2+2=` generates `401`. The stated addition score checks only N+1 characters, and the editor did not target termination.

## 5. Caveats for the write-up

- Report hidden N1 = 100% as a lookup over all 100 one-digit problems, built from teacher-forced states. It does not show addition ability, and it collapses outside the first-line `<bos>` context.
- The per-tensor `native_rel_change` overstates the functional change. `gate_proj` 26,529x comes from 10 repurposed rows, and `blocks.0.attn.o_proj` at 0.98 is an SVD basis change that preserves 31 of 32 components. 42 of 56 tensors changed, mostly through the rank-1 marker projection (about 4-5% each).
- `audit_ok=false` comes only from the `requests\b` false positive. Fix the regex before aggregating `audit_ok` across runs.
- 38 of 50 passes were validation padding forced by the "end only when the budget is used" rule. The effective design budget was about 12 passes. Consider letting the editor end early, or count batches by size.
- The off-task metrics that pass (text KL, natives) hide '+'-context interference, which shows only in prefixed text and the not-first-line check. Prose containing '+' was never in the hidden text KL.
