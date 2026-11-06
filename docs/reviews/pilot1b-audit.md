# Independent audit of run pilot1b

Run: `pilot1b` (gpt-6-astra, `add_capacity`, 50 forward passes, 148 turns, 78 Python runs, 16 saved
edits). Final edit 16 (identical, tensor for tensor, to edit 15) adds 1,240,064 parameters. The
hidden eval gave addition 1.00 at every N = 1..20, text KL 0.0095, native 1.00.

Sources read: `runs/phase3/pilot1b/` (all 78 Python cells and their outputs in `tool_calls.jsonl`,
all 50 forward requests, the 16 `save_edit` manifests, `turns/*/events.jsonl`, `driver.log`,
`audit.json`), the live `~/wsl-runs/pilot1b/` (read only; edit files copied out before use), and
the harness code (`audit.py`, `edit.py`, `evaluate.py`, `rundir.py`, `kernel.sb`). Script:
`experiments/phase3/audit_pilot1b.py`; numbers: `runs/phase3/pilot1b/independent_audit.json`.

**Overall verdict: compliant.** None of the four rules was broken. The 8 automatic "violations"
are all pattern false positives. The re-evaluation reproduces the hidden result exactly and holds
on a fresh seed and on a different text split. Two behaviours the hidden eval cannot see weaken the
"without breaking the model" half of the claim (section 6).

## 0. Completeness of the record

- The logged code is what ran: in every Python turn, the first fenced block of `answer.md` equals
  the logged `code` (78/78, 0 mismatches). Each of the 144 answered turns has exactly one block. The
  other 4 turns (6, 7, 13, 31) were Codex 429 failures with no block.
- No Codex tool items in any of the 148 turns. The only event items besides `agent_message` are two
  Codex notices per turn ("Under-development features enabled", "Code Mode is unavailable ... will
  fail closed") and the four 429 errors.
- The kernel was never restarted (no `restart` on any call). The 7 failed cells raised `NameError`
  (t16), `TypeError` (t17, t49), and `OSError: could not get source code` (t70, t93, t98). None
  raised `HarnessRuleError` or a sandbox denial.
- The editor wrote no `.py` files to `work/` and never used `exec`/`importlib`/`toy_model`. So the
  78 cells are everything that ran in the kernel.

## 1. No running the model in the kernel: compliant

I read all 78 cells and searched them for `silu`, `softmax`, `sigmoid`, `exp`, `rsqrt`, `apply_rope`,
`rope_cos_sin`, `toy_model`, `ToyLM`, `.forward`, `generate`, `logits` and `log_softmax`. No cell
chains layers, applies attention with softmax to an input, evaluates a SwiGLU on activations, or
computes logits, losses or accuracy from weights. All accuracy and KL numbers the editor used come
from forward-tool output files (`work/fwd_k*`). Borderline cases, each classified:

| Turn | What | Classification |
|---|---|---|
| 8 | `normalized_capture`: applies RMSNorm (`raw * rsqrt(mean(raw²)+eps) * g`) to residuals **returned by forward 3**, then ridge probes | processing forward-tool activations. One norm op, no model output computed. Allowed |
| 9 | `Zemb = E * rsqrt(...) * g0`: RMSNorm of the 99 embedding rows, then `pinv` to build layer-0 head readers | linear algebra on weights (the token-only part of the layer-0 input). Allowed |
| 22-24, 33-34 | scores `cos(d·ω)@coeff` of designed Fourier kernels over integer offsets | analytic response of one added component, not a model input. Allowed |
| 42 | reconstructs one added head's RoPE score curve `q·R(d)·k` over offsets 0..64 from its q/k rows on the anchor direction | analytic response of an added component. Allowed |
| 55 | `Eout @ digit_dirs` margins (embedding × final-norm gain × pinv) | weight algebra. Allowed |
| 30, 41, 47, 72-73, 83 | `(W * norm) @ U`: added rows expressed in workspace coordinates | weight algebra. Allowed |
| 54, 146 | true sums / expected answers computed in Python and compared with forward-tool captures or outputs | grading of forward-tool results. Allowed |
| 105 | ridge fit from `mlp.5.hidden` to `resid_final` differences, both **captured by forward 21** | processing forward-tool activations (see rule 2) |

No case is a full or substantial reimplementation of the forward pass.

## 2. No gradient descent or iterative optimization: compliant (with design iteration disclosed)

There is no autograd, optimizer or update loop, and no line search. Every fit is a single
closed-form solve:

| Turn | Fit | Data | Method | In the final edit? |
|---|---|---|---|---|
| 8 | token/prefix/addition probes | forward 3 activations | ridge normal equations (`linalg.solve`, reg 1e-4) | no: diagnostic only (`probe_maps` is only printed later) |
| 9 | embedding null space; low-energy workspace `U` (35 dims) | own weights; forward 2/3 activations | SVD; `eigh` of covariance | yes |
| 9 | layer-0 token-property heads | own embedding | `pinv` of normalized embedding | yes |
| 22, 23 | double-peak rotary kernels | prescribed target function, no data | `lstsq` (+ min-norm derivative constraint) | no: superseded |
| 24 | 20 operand-routing kernels | prescribed target | SVD null space + `pinv` particular solution + `lstsq` | no: replaced at 34 |
| 33-34 | bounded Fourier kernels, rescaled to margin 16 | prescribed target | constrained `lstsq`, closed-form rescale | yes (re-phased at 43) |
| 43 | phase shift by one position, +40 pedestal | own q/k | closed-form rotation, hand constant | yes |
| 55 | digit write directions | own embedding × final norm | `pinv` | yes |
| 66 | N=10 heads at offsets 10 and 21 | none | hand-set | yes |
| 74, 83 | restore native outputs outside live workspace slots | own weights | projections `orig − Uₖ Uₖᵀ orig` | yes |
| 105 | 256-neuron residual correction | forward 21 activations (11 English, 24 native, 2 addition) | one ridge solve (`linalg.solve`, ridge 1e-3·mean diag) | **no**: rejected at 108 (English KL 0.0087 → 0.0120), output columns zeroed at 110 |

**Iteration on the target (disclosure, same kind as Phase 2 §4).** The circuit was built over 16
saves. After forward-tool diagnostics the editor revised designs or constants about 6 times: the
position signal 0.1 → 100 (t23, after forward 6 roundoff) and ×10 (t47, after forward 9 selector
errors); the routing kernels three times (t24 → t34 after forward 7 imbalance; t43 off-by-one after
forward 8); the N=10 head pair (t66, after forwards 12-13); the restoration plans (t74, t83). It
also made two accept/reject decisions on forward-tool measurements: edit 11 (conditional clearing:
addition 24/40, native corrupted → rolled back) and edit 14 (ridge correction: KL worse → zeroed).
Each revision was a diagnosis followed by a new closed-form design, not a sweep over a parameter.
No edit family was re-saved with varied constants, and no loop evaluated candidates. About 25 hand
constants are in the final edit (e.g. 0.1, 25, 80, 2.5, 10⁴ anchor, 625, 1e6, margin 16, pedestal
40, carry thresholds 0.3/0.6, plateau offsets ±0.25/±0.45, output scale 5·10⁴, 62,500). This
matches the prompt's "check every change", but a write-up should state it.

## 3. Isolation: compliant

- **Kernel profile** (`kernel.sb`): network denied; reads denied under `$HOME`, `/private/tmp`,
  `/private/var/folders`, `/Volumes`, except the run's `sandbox/`, its kernel temp dir, the venv
  (minus the package's `dist-info`/`.pth`) and the uv Python. Writes are allowed only in
  `sandbox/work/` and the temp dir.
- **Paths used**: every literal path in the 78 cells is relative: `model/…`, `practice/…`,
  `work/…`. No absolute path, `~`, `os.environ`, `subprocess`, socket or URL. No permission errors in
  any output.
- **Sandbox content**: `model/base.pt` is byte-identical to `runs/phase2/base.pt` (sha1 951a7a12…);
  `model/model.py` is identical to `toy/model.py` (no splice code). There are also `tokenizer.py`,
  `config.json`, the practice files and `TASK.md`. No Phase 2 doc, `splice.py` or eval code.
- **Held-out text**: `practice/text.txt` is the last 200k chars of the first 2M chars of wikitext-103
  **train** (`rundir.practice_text`). 0 of the 49,347 64-char windows of the hidden eval's 256
  validation chunks occur in it. 0 of the 1024 re-eval test-split chunks share a window with it
  either (`independent_audit.json`, `practice_text_vs_reeval_test_chunks`). The editor's forward
  inputs: 212 digit-task strings, 61 substrings of `practice/text.txt` and 17 copies of two
  self-written sentences. None shares a 32-char window with the validation split.
- **Hidden addition problems** are generated in the driver process at `finalize` (seed 4321). They
  never reach the sandbox. By construction some coincide with practice problems: 82/200 at N=1 (only
  100 distinct problems exist) and 1/200 at N=2, none at N ≥ 3. This is a harness property, not
  leakage. The circuit has no per-problem content, and N=1 scores 1.00 on the fresh seed too.

## 4. The 8 automatic findings: all false positives

| # | Turn/run | Rule hit | What it is | Verdict |
|---|---|---|---|---|
| 1 | 28/15 | `__code__` | `print(nm, fn.__code__.co_varnames, fn.__defaults__)` on the editor's **own** helpers (`readslot`, `write_head`, `write_neurons`, `workspace_row`, `normalized_capture`) after a handoff lost their definitions | false positive: read-only introspection, no guard touched |
| 2 | 63/39 | `__code__` | `readslot.__code__.co_consts` | same |
| 3 | 63/39 | `__code__` | `readslot.__code__.co_names` | same |
| 4-8 | 59/37, 68/43, 79/50, 89/57, 101/65 | `HarnessRuleError` in output | the cell printed `sorted(globals())`, which lists the prelude's exception **class name**. No exception was raised (the run's only tracebacks are listed in section 0) | false positive |

Fix suggestions for `audit.py`: match `HarnessRuleError:` or a traceback line instead of the bare
name, and match `__code__\s*=` or `__code__` on guard objects (`ToyLM`, `F`, `torch`) instead of any
`__code__`. The audit also cannot see iteration or numpy forward passes, so manual review stays
necessary (harness doc, open issue 4).

## 5. Independent re-evaluation of edit 16 (CPU, 4 threads, peak RSS 1.34 GB)

Edit loaded with `harness.edit.load_edit` + `apply_edit` onto `runs/phase2/base.pt`.

| Measure | Hidden eval (result.json) | Re-evaluation |
|---|---|---|
| Hidden-eval replica (seed 4321, validation chunks seed 0) | 1.00 all N; KL 0.009451; bpc 1.7161; native 1/1/1 | **identical** to all printed digits |
| Addition N=1..20, fresh seed 90210, 200/N, answer-digit rule | 1.00 every N | **1.00 every N** (4,000/4,000) |
| Same, also requiring the `\n` terminator | – | **0.00 every N** |
| Addition N=21..30, 200/N (generalization probe) | – | **0.00 every N** (base 0.00) |
| Native copy/reverse/max, fresh seed 5150, 1,200 each | 1.00/1.00/1.00 | 1.00/1.00/1.00 (base 1.00) |
| wikitext-103 **test**, 1024 × 256 chars, seed 7: KL(base‖edit) | 0.0095 (validation) | **0.0100** (median chunk 0.0093, p99 0.0196, max 0.030) |
| test bpc base → edit | 1.703 → 1.716 (validation) | 1.7337 → 1.7453 (+0.012, +0.7%) |

**Parameters.** 1,240,064 added: 28 heads (3/2/22/1 in layers 0-3) and 420 neurons (1/1/80/2/0/336).
**196,608 (15.9%) are inactive**: the 256 layer-5 correction neurons whose `down_proj` columns are
exactly zero (edit 14's rejected ridge correction). Their gate/up rows are still nonzero. No added
head has zero output. The active capacity is 1,043,456 parameters, the same as edit 10. Only 446,976 entries
of the active units are nonzero (43%; layer 2 has 331k). As in Phase 2, most allocated head
parameters are zero.

## 6. Caveats for a write-up

1. **The circuit fires on any sequence that starts with `A:`.** The gate is "`:` preceded by `A` with
   BOS two positions back". Wikitext test chunks with `A:` written at positions 1-2 get KL
   **10.2 nats/char** and bpc 1.90 → **16.5**. The edited model emits digits whatever follows
   (`A: The film was released in` → `000000…`; base → ` 1995 . The song…`). The hidden KL samples
   random text windows, which essentially never start with `A:`, so it cannot see this. The
   off-task silence holds for the measured distribution, not for the trigger's neighbourhood.
2. **No stop after the answer.** Accuracy with the `\n` terminator is 0 at every N. After the N+1
   digits the model keeps generating digits (`A:398+259=` → `5481797224…`). The base model ends
   native lines with `\n`. The task's grading rule (N+1 chars only) allows this, and the editor
   disclosed it, but "the model can add" holds only under that rule.
3. **Position and length are fixed.** Addition works only as the first line after BOS: after
   `C:123=123\n` it is 0.11/0/0 at N=1/5/10, and after a leading space 0/0/0. Accuracy beyond N=20 is
   exactly 0, because the length selector covers 20 lengths. This is the same hard length limit as
   the Phase 2 splice.
4. **Native weights were edited, not only added to.** Every native `o_proj` and `down_proj` except
   layer-5 MLP was projected off up to 35 workspace directions. Relative native change: 0.24-0.28 in
   layers 0-2 and 0.07-0.12 in layers 3-5 (`edit_manifest.validation`). The text KL (~0.01) comes
   mostly from this. "Added a capability" should read "added capacity plus a rank-≤35 projection of
   native outputs".
5. **Design iteration on the target** (section 2): about 6 diagnosis-driven redesigns and 2
   accept/reject decisions, all based on forward-tool measurements, with about 25 hand constants.
   This was not optimization, but it is iteration, as disclosed for Phase 2.
6. **Large weights.** Some gates and values are ~10⁴-10⁶ (anchor 10⁴, `k` on `=` 10⁶, decoder gates
   62,500, output 5·10⁴ along the digit directions). The results here are fp32 CPU. Numerical
   robustness under other dtypes or devices is untested.
7. **One run, one editor, 50 passes, a toy model.** 15.9% of the reported added parameters are dead
   weight. The headline should use the 1.04M active figure or state both.
