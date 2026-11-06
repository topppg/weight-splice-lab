# Hardened hidden eval (pre-registered before pilots 2-3)

The pilot1b audit (`docs/reviews/pilot1b-audit.md`) found five blind spots in the Phase 3 hidden eval. This
eval keeps the old primary bar unchanged, so comparisons with pilot1b and Phase 2 stay valid, and
adds a robustness block with thresholds fixed **before** pilots 2 and 3 run. The thresholds below
must not be changed after those results are in.

Code: `harness/evaluate.py` (`robustness`, `evaluate_model(..., test_text=...)`), probes in
`toy/metrics.py` (`termination`, `prefixed_target_accuracy`, `native_weight_change`,
`added_param_report`) and `toy/data.py` (`prefixed_text_chunks`, `text_before_newline`,
`native_lines`). Tests: `tests/toy/test_metrics.py`, `tests/harness/test_evaluate.py`.
Rescore: `experiments/phase3/rescore_hardened.py` → `runs/phase3/rescore_hardened.json`.

## Primary bar (unchanged)

Computed by exactly the same code, seeds and data as before: addition N=1..20 (200/N, seed 4321,
answer-digit rule), wikitext **validation** KL(base‖model) and bpc over 256 × 256 chars (seed 0),
copy/reverse/max exact match with `\n` (1,200 each, seed 1234).

- `meets_bar` (old key, old meaning): text KL < 0.05 and every native task ≥ 0.98.
- `meets_primary_bar` (new): `meets_bar` and `addition_mean_n1_8` ≥ 0.95.

The base and pilot1b primary numbers in the rescore are identical to `base_eval_n200.json` (old)
and `runs/phase3/pilot1b/result.json`.

## Robustness block (`hidden_eval.robustness`)

All text comes from **wikitext-103 test** (never train, which the sandbox's `practice/text.txt` is
cut from; validation stays reserved for the primary KL). Addition problems use fresh seeds.

| key | probe | threshold |
|---|---|---|
| (a) `a_prefixed_text` | 256 × 256-char test windows, positions 1-2 forced to `A:`, the next char a non-digit (windows starting with a digit are resampled; seed 7). KL(base‖model) over all positions, plus bpc of both | KL **< 0.05** |
| (b) `b_termination` | N=1..8, 200/N (the primary problems, seed 4321). Greedy N+2 tokens; `terminated` = the token after the N+1 answer chars is `\n`, the token the base model ends every native answer with (native exact match, which requires it, is 1.00 for the base). `strict_accuracy` = answer + `\n` exactly (reported) | `terminated` mean **≥ 0.95** |
| (c) `c_not_first_line` | N=1..8, 200/N, two conditions: `after_native_lines`, 1-3 training-format native lines (`C:x=x\n`, `R:…`, `M:…`, lengths 1-12) before the `A:` prompt (seed 4323); `after_text`, a 32/64/96/128-char test-split window that ends with `\n` (seed 4324). Answer-digit rule | **min** of the two means **≥ 0.95** |
| (d) `d_length_n21_30` | addition N=21..30, 200/N (seed 4325) | report only |
| (e) `e_native_weight_change` | per original tensor ‖W[native slice] − W₀‖_F / ‖W₀‖_F (only changed tensors listed), `total_rel` over all 56 native tensors, `max_rel` | report only |
| (f) `f_added_params` | per added head (4·d·head_dim) / neuron (3·d): `allocated`, `nonzero` entries, `inactive` = units whose output weights (o_proj columns / down_proj column) are all zero, `active`, `nonzero_active` | report only |

`meets_strict_bar` = `meets_primary_bar` and (a), (b), (c) pass.

Added keys in `hidden_eval`: `robustness`, `meets_primary_bar`, `meets_strict_bar`. No old key
changed. `base_reference` recomputes the cached base eval if it lacks `robustness`.

## Rescore (`runs/phase3/rescore_hardened.json`, CPU, 4 threads, 200/N, ~3-4 min and 1.2 GB RSS per model)

Columns: primary = addition mean N=1..8 / text KL (validation) / native copy·reverse·max /
`meets_primary_bar`. Robustness (a)-(f) as defined above; **bold** = fails its threshold.
"after native"/"after text" are the two (c) conditions. (e) = native-weight relative Frobenius
change, total / largest tensor. (f) = added params allocated / nonzero / inactive.

| model | add N1-8 | KL | native | primary | (a) `A:` KL | (b) terminated | (b) strict | (c) after native | (c) after text | strict bar | (d) N21-30 | (e) total / max | (f) alloc / nonzero / inactive |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 0.013 | 0 | 1.00/1.00/1.00 | no | 0 | **0.00** | 0.00 | **0.01** | **0.00** | no | 0 | 0 / 0 | 0 / 0 / 0 |
| gated splice (N_max 8) | 0.596 | 0.043 | 0.96/0.67/0.74 | no | 0.0499 | **0.86** | 0.59 | **0.02** | **0.00** | no | 0 | 0 / 0 | 1,006,848 / 607,488 / 0 |
| ungated splice (N_max 20) | 0.944 | 0.991 | 0.10/0.08/0.06 | no | **0.99** | 1.00 | 0.94 | **0.03** | **0.00** | no | 0 | 0 / 0 | 1,381,632 / 644,352 / 0 |
| pilot1b edit 16 | 1.000 | 0.0095 | 1.00/1.00/1.00 | **yes** | **10.19** | **0.00** | 0.00 | **0.01** | **0.00** | no | 0 | 0.085 / 0.280 | 1,240,064 / 512,768 / 196,608 |
| LoRA r8 + native replay, 1k | 0.117 | 1.458 | 1.00/1.00/0.97 | no | **1.59** | 0.995 | 0.12 | **0.02** | **0.00** | no | 0 | 0.60 / 1.27 | 0 / 0 / 0 |
| LoRA r8 + native replay, 3k | 1.000 | 0.466 | 1.00/1.00/0.99 | no | **0.52** | 1.00 | 1.00 | **0.45** | **0.01** | no | 0 | 0.43 / 1.03 | 0 / 0 / 0 |
| LoRA r8 + native replay, 10k | 1.000 | 0.694 | 1.00/1.00/1.00 | no | **0.76** | 1.00 | 1.00 | **0.01** | **0.00** | no | 0 | 0.42 / 1.01 | 0 / 0 / 0 |
| LoRA r8 + native + text replay, 1k | 0.117 | 0.480 | 1.00/0.98/0.94 | no | **0.52** | 0.98 | 0.12 | **0.02** | **0.00** | no | 0 | 0.50 / 0.95 | 0 / 0 / 0 |
| LoRA r8 + native + text replay, 3k | 0.994 | 0.441 | 1.00/1.00/0.98 | no | **0.47** | 1.00 | 0.99 | **0.09** | **0.00** | no | 0 | 0.49 / 1.00 | 0 / 0 / 0 |
| LoRA r8 + native + text replay, 10k | 0.878 | 0.513 | 1.00/1.00/1.00 | no | **0.54** | 1.00 | 0.88 | **0.09** | **0.00** | no | 0 | 0.48 / 1.05 | 0 / 0 / 0 |

Checks: the base and pilot1b primary keys are identical (every value) to `base_eval_n200.json` and
`runs/phase3/pilot1b/result.json`; pilot1b's (a) KL (10.19), inactive (196,608) and
nonzero-active (446,976) match the audit. Gated-splice addition by N (0.93, 0.97, 0.97, 0.99,
0.74, 0.12, 0.01, 0.05) matches Phase 2 seed 99. The ungated splice's primary mean is 0.944 here
(Phase 2 reported 1.00 at N=1..8 with its own probe set; here N=4..8 are 0.92-0.97).

Reading the table:
- **No model meets the strict bar.** pilot1b is still the only one that meets the primary bar, and
  it fails (a) (KL 10.2: any `A:` text is destroyed), (b) (never emits `\n`) and (c).
- **(c) is failed by everything, including LoRA.** No model adds after a preceding text line
  (≤ 0.01), and after native lines the best is LoRA native replay 3k at 0.45. LoRA and pilot1b were
  only ever shown or tested on addition as the first line. (c) is the hardest pre-registered
  probe; that is a property of the task setup, not only of the editor.
- **(b) separates the methods:** the splices and LoRA emit `\n` (LoRA trains on it; the hand splice
  has an explicit end-of-answer lookup), pilot1b never does.
- **(e)/(f) separate "added capacity" from "rewrote native weights":** the splices change no native
  weight; pilot1b changes 11 native output projections by 7-28% (total 8.5%); LoRA changes every
  native matrix, by up to 100-127% of its norm (total 42-60%), with no added capacity.
- (d) is 0 for every model, as expected (the splices and pilot1b have no detector past their N_max;
  LoRA trained on N ≤ 8).
- The gated splice passes (a) only just (0.0499 < 0.05).

## LoRA with replay: data budget vs accuracy (Task 3 of the eval hardening)

`experiments/phase2/run_lora.py`, rank 8 (α 16) on all q/k/v/o/gate/up/down, 246k trainable
params, 3000 steps × 64, AdamW 3e-3 with linear decay, loss on answer + `\n` (whole window for text),
on MPS inside `gpu_session` (peak 0.06-0.09 GB MPS, 0.74-1.35 GB RSS, 2.3-4.1 min of training per budget).
Budget = labelled addition examples (N uniform in 1..8). Merged LoRA checkpoints
`runs/phase2/lora_replay_{native,text}_b<k>.pt` (local only, not in a release) were scored with the
hardened eval above.

| run | addition examples | native replay | text replay | add N1-8 | N9, 10 | text KL | bpc | copy / rev / max | train loss |
|---|---|---|---|---|---|---|---|---|---|
| no replay (Phase 2) | 1,000 / 3,000 / 10,000 | 0 | 0 | ≈0.10 / 1.00 / 0.88 | 0 | 1.50 / 1.49 / 2.48 | 3.86 / 3.83 / 5.26 | ≤ 0.02 | ≤ 0.07 |
| `--replay` | 1,000 / 3,000 / 10,000 | same count | 0 | 0.117 / **1.000** / 1.000 | ≤ 0.025 | 1.46 / **0.47** / 0.69 | 3.79 / 2.37 / 2.69 | ≥ 0.96 | ≤ 0.002 |
| `--replay --text-replay 16` | 1,000 / 3,000 / 10,000 | same count | 16 fresh 64-char train windows per batch (48k total) | 0.117 / **0.994** / 0.878 (N=8: 0.065 at 10k) | 0 | 0.48 / **0.44** / 0.51 | 2.37 / 2.31 / 2.41 | ≥ 0.94 | 0.74-0.84 |
| `--replay --text-pool` (failed variant) | 1,000 / 3,000 / 10,000 | same count | a fixed pool of `budget` windows | 0.10 / 0.06 / 0.05 | 0 | 1.52 / 1.20 / 0.75 | 3.86 / 3.38 / 2.75 | ≥ 0.90 | 0.10 / 0.39 / 0.79 |

(`runs/phase2/lora_replay_results.json`, `lora_replay_text_results.json`,
`lora_replay_textpool_results.json`; the MPS numbers there and the CPU hardened eval agree.)

- **Data cost unchanged:** with replay LoRA still needs ~3k labelled addition examples for
  ≥ 0.98 at every N=1..8 (1k: only N=1, 0.82-0.84); the editor and the splice use none.
- **Replay fixes the native tasks, not text.** Native-task replay brings copy/reverse/max back to
  ≥ 0.98 (from ≈ 0), so LoRA-with-replay passes the native half of the bar. Text KL stays at
  0.44-0.69, about 10× the bar and ~50× pilot1b's 0.0095. Fresh-text replay helps at 1k (1.46 → 0.48)
  but not at 3k (0.47 → 0.44).
- The fixed text pool memorised its windows (train loss 0.10 nats/token at 1k vs ~1.2 for the base
  on text) and did worse; kept as a record, not a baseline.
- Per-position KL of the text-replay 1k model: 0.15-0.26 at positions < 64 (inside the replay
  window length), 0.52-0.55 at 64-256. Longer replay windows would likely help, but batch 64 × 257
  needed > 4 GB of MPS in Phase 2, so this was not run (see open issues).

## Open issues

1. LoRA text replay used 64-char windows only; a 256-char replay (micro-batched to fit memory), a
   lower learning rate, or more steps were not tried and could lower LoRA's text KL. LoRA
   checkpoints are local only (`runs/phase2/lora_replay_*.pt`), not uploaded to a release.
2. (c) has no model that passes it, including LoRA trained on addition. If pilots 2-3 also fail it,
   that says more about the first-line-only practice data than the editor. The thresholds stay as
   pre-registered anyway.
3. (a) only forces `A:` at the start of the sequence. A trigger keyed on `\nA:` mid-sequence is
   probed for addition by (c) but not for text damage.
4. The `runs/phase2/run_lora_replay.log` per-budget lines were lost (a `git pull --autostash`
   rewrote the tracked log mid-run); the results JSON is complete.
