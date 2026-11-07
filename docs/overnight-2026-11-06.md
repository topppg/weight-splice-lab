# Overnight 2026-11-06: pilot4c (add_capacity) and pilot5 (edit_only)

Both runs used harness v3.1 at `030b54e` (clean), gpt-6-astra with effort high, 50 forward passes and a cap of 4 weekly points.
They ran back to back from `overnight.sh`: pilot4c from 20:46 to 21:22 and pilot5 from 21:38 to 22:25.
The run logs and hidden-eval results are in `runs/phase3/{pilot4c,pilot5}/`.
The log reviews are in `docs/reviews/pilot4c-review.md` and `docs/reviews/pilot5-review.md`.

| Run | Verdict | Turns | Python / forward / saves | Points | Input tokens (cached) | Stop |
|---|---|---|---|---|---|---|
| pilot4c | compliant with caveats | 81 | 25 / 50 / 6 | 1 | 3.12M (97%) | budget used, final edit 6 (= edit 5) |
| pilot5 | compliant | 80 | 26 / 50 / 4 | 2 | 3.20M (97%) | budget used, final edit 4 (= edit 3) |

The weekly usage meter went from 22% to 25% over both runs. Pilot points used to date: 11 of 15.

## All models on the same hidden eval

Each row uses 200 examples per length and the same seeds.
Rows other than pilot4c and pilot5 come from `runs/phase3/rescore_hardened.json`.
To regenerate the table, run `python experiments/phase3/compare_runs.py`.

| Model | Weights written | Add N1–8 | Add N9–20 | Text KL | Native (min) | (a) `A:` KL | (b) stops | (c) not 1st line | (d) N21–30 | (e) native Δ (rel) | (f) added params (nonzero / allocated) | Primary bar |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hand splice, gated (N≤8) | by hand, closed form | 60% | 0% | 0.0433 | 67% | 0.050 | 86% | 0% | 0% | 0 | 607,488 / 1,006,848 | no |
| hand splice, ungated (N≤20) | by hand, closed form | 94% | 91% | 0.9908 | 6% | 0.992 | 100% | 0% | 0% | 0 | 644,352 / 1,381,632 | no |
| LoRA + native replay, 1k | gradients | 12% | 0% | 1.4575 | 97% | 1.588 | 100% | 0% | 0% | 0.602 | 0 | no |
| LoRA + native replay, 3k | gradients | 100% | 0% | 0.4658 | 99% | 0.524 | 100% | 1% | 0% | 0.425 | 0 | no |
| LoRA + native replay, 10k | gradients | 100% | 0% | 0.6936 | 100% | 0.759 | 100% | 0% | 0% | 0.416 | 0 | no |
| LoRA + text replay, 1k | gradients | 12% | 0% | 0.4797 | 94% | 0.523 | 98% | 0% | 0% | 0.501 | 0 | no |
| LoRA + text replay, 3k | gradients | 99% | 0% | 0.4413 | 98% | 0.475 | 100% | 0% | 0% | 0.487 | 0 | no |
| LoRA + text replay, 10k | gradients | 88% | 0% | 0.5129 | 100% | 0.544 | 100% | 0% | 0% | 0.482 | 0 | no |
| pilot1b (add_capacity) | agent, closed form | 100% | 100% | 0.0095 | 100% | 10.189 | 0% | 0% | 0% | 0.0849 | 512,768 / 1,240,064 | yes |
| pilot4c (add_capacity) | agent, closed form | 100% | 100% | 0.0126 | 100% | 0.015 | 100% | 0% | 0% | 0.195 | 177,152 / 378,368 | yes |
| pilot5 (edit_only) | agent, closed form | 12% | 0% | 0.0009 | 100% | 0.016 | 0% | 0% | 0% | 5.35e+03 | 0 | no |

## Findings

- **The agent added the capability when it was allowed new capacity (pilot4c).**
  - Addition is 100% at every N from 1 to 20 and text KL is 0.013, roughly 35x better than the best LoRA-with-replay baseline (0.44).
  - It is the first edit that also passes (a), `A:`-prefixed text, and (b), termination; pilot1b failed both.
  - How it works: the editor reserved 24 unused residual directions and projected them out of all 42 native matrices.
    That projection alone gives text KL about 0.014 and accounts for the 19% change in native weights.
  - It then wrote the circuit in one closed-form cell (t14), which passed 60/60 on its first forward check.
    The circuit is a token head and a position head, a length decoder, 4 rotary digit-fetch heads, carry/sum neurons, a digit readout and a newline neuron.
  - Write-up wording: say "added capacity plus a rank-24 projection of native weights", not "native untouched".
- **It failed when edits were limited to existing weights (pilot5).**
  - The editor built a lookup table for the 100 one-digit problems.
    Its own checks showed 0/950 on N = 2–20 (t27–t32), and it reported the edit as one-digit only.
  - The 26,500x relative change in `blocks.5.mlp.gate_proj` comes from 10 rows (norm about 2.5e5) of a badly conditioned fit.
    It costs KL 2.6–2.9 on text containing `+`, which plain-text KL does not measure.
  - Along with the hand splices, this is consistent with interference, not expressivity, being the bottleneck.
    With n = 1 per condition, it is suggestive only.
- **Failed by every model, including LoRA:**
  - (c): addition only works on the first line;
  - (d): no model generalises to N = 21–30.

## Harness findings

- **Audit false positive.** The pattern `requests\b` matched names such as `validation_gen_requests`.
  It produced all 10 of pilot5's "violations".
  The fix is in `audit.py`, as `\brequests\b` with regression tests.
  Re-auditing gives 0 violations for pilot5 (`runs/phase3/pilot5/audit_rerun.json`) and 0 for pilot4c, 1b and 2.
  pilot5's `result.json` is left as produced, so `audit_ok` there is still false.
- **Forward budget padding.** A run only ends once all 50 passes are used, so both editors spent most of their budget re-checking an edit that had not changed.
  The design used 11 passes in pilot4c and 12 in pilot5.
  Before repeats, let the editor finish early with a final `save_edit`.
- **Ops.**
  - At 18:56 the disk filled (another job's 6 GB model download) and killed the watchdog and the usage watcher.
    Both were restarted, and the runs started with 22 GB free.
  - The chain waited 30 minutes for another agent on the personal Codex profile to go idle.

## Not done yet

- `mark_reviewed.py` (review record and checkpoint release) is waiting for owner approval.
- There are no repeats (n = 1 per condition).
