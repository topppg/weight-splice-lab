# Reproducing Sparks of Weight-Space Programming, and an agent that adds a capability

Status: final (2026-11-11); the project has stopped. Every number here comes from a file in this
repo (`runs/` or `docs/`). It is independent work and not peer reviewed.

## TL;DR

- **Gemma 3 270M IT** (the paper's negative case; section 6, `docs/phase4-gemma.md`):
  - With only a 50-pass forward-pass tool, one run in each of three setups (plain addition ×2,
    reversed ×1) found parts of a circuit but no multi-digit algorithm. This matches the paper.
  - When it could also run the model in its own Python session (still no gradients, and no
    search over weight values), it hand-built a circuit in 10 added heads and 423 added neurons.
    Reversed-digit addition went from 0.9% (mean N1–8) to 100% at every N = 1–30 (n = 1). The
    edit also adjusts 34 native tensors (norm compensation).
  - That edit fails the pre-registered bar: text KL 0.445 and chat KL 0.086. 5 of 256 hidden
    Wikipedia chunks hold 99.8% of the text KL. The chat KL is not explained.
  - Compliance has had only a first-pass check (section 6). This is evidence that the ability can
    be written in, not a clean counterexample.
  - The editor's own drift checks, on short practice windows, measured 0.0004.

- We reproduce the headline edit of Takehi et al. (MIT CSAIL, Nov 2026, "Sparks of Weight-Space Programming"):
  all three published Qwen3.5-0.8B edits rebuild bit-exact, and we re-derive the "loop direction"
  from the logged texts (|cos| 0.99998 with the published one, 64/64 and 63/64 neuron overlap).
- Re-scoring the paper's own published generations at shorter output caps shows the gain is a
  long-context effect: at 4k tokens base 3.5% vs edited 8% (both >90% truncated); at 32k 7.5% vs 46%.
- The paper concludes that on Gemma 3 270M editing "could not add an ability the model lacked".
  On a 5M-parameter toy
  transformer we show a capability (multi-digit addition) can be added with closed-form weight writes
  and no gradients — by hand (Phase 2) and by the same editor model, gpt-6-astra, in our open
  harness (Phase 3).
- The bottleneck is not expressivity but interference: our hand-compiled splice reaches 100%
  addition but destroys native behaviour (text KL 0.65–0.99). The agent's splice meets a
  pre-registered bar (addition 100% at N=1..20, text KL 0.0095, native tasks 1.00), audited as
  rule-compliant, with robustness gaps listed below.
- Three runs per condition on harness v3.1 (`docs/repeats-2026-11-07.md`):
  - When the editor may add heads and neurons, it adds full addition (100% at every N = 1..20) in
    3 of 3 runs. 2 of 3 also meet the KL bar; pilot8 reached 0.056.
  - Limited to existing weights (the paper's setting), it fails 3 of 3 times and ends with a
    one-digit lookup table.
  - A fresh-seed, test-split re-evaluation reproduces all six results.
  - Fisher exact test: p = 0.05 one-sided. With n = 3 per condition this is consistent but small,
    not a significance claim.

## 1. Reproduction of the paper (`docs/phase1-repro.md`)

- Edits: p10c (rank-1 on 30 matrices, α=1.5), p25c (zero L15 head 7), p50c
  (W = W0 − 6·u⊗(mask64 ⊙ uᵀW0) on L15/L19 down_proj, 130,188 bf16 entries changed). All hash-verified.
- u = normalised mean (repetitive − productive) residual over 4 hand-written text pairs (last 36
  tokens). Recomputed from scratch; matches.
- Grader port agrees with the paper on 200/200 published samples.
- Re-scored published outputs (MATH-500 problems 0–199, first sample):

  | Cap | Base acc / cut / loop | 50-pass edit acc / cut / loop |
  |---|---|---|
  | 4,096 | 3.5% / 98% / 14% | 8% / 92% / 0% |
  | 8,192 | 6.5% / 94.5% / 63% | 14.5% / 85.5% / 0% |
  | 32,768 | 7.5% / 85.5% / 96% | 46% / 31% / 11% |

- New observation: 16% of the edited model's truncated answers at 32k degenerate into whitespace.
- Not done: our own full 32k vLLM eval (≈3 H100-hours for MATH-500×4, base + edit); not needed for
  the reproduction claim since the edits are bit-exact and the grader agrees.

## 2. Adding a capability by hand (`docs/phase2-toy.md`)

- Toy: 6-layer Qwen-style decoder (RMSNorm, RoPE, SwiGLU, tied embeddings), 5.14M params,
  char-level, trained on wikitext + copy/reverse/max digit tasks; ~0% on addition (`A:` + reversed
  operands + `=` + reversed sum).
- Hand-compiled splice: closed-form writes only (SVD/pseudoinverse of the unembedding, covariance
  eigendecompositions, ridge probes), added heads/neurons, native weights untouched.
- Result: ungated 100% addition N=1..8 but text KL 0.65 and native ≤ 0.17 (copy 0.17, reverse 0.06, max 0.09); gated KL 0.04 but
  collapses past N=4. No configuration met the bar (KL < 0.05, native ≥ 0.98).
- Why: no quiet residual directions (superposition), bias-free RMSNorm/SwiGLU gives no threshold,
  and the digit-read interface itself breaks native digit circuits.
- Disclosure: ~40 target-inspected debug iterations.

## 3. Harness (`docs/phase3-harness.md`)

- Editor: gpt-6-astra via Codex CLI, all built-in tools disabled, profile hooks/memories/skills
  off; acts only via `python` (sandboxed CPU kernel: run dir only, no network, autograd/optim and
  the model's forward disabled), budgeted `forward` (50), and `save_edit` (rule validation,
  ≤ 1.5M added params). Hidden eval and audit after the run.
- Budget guards from Codex's own rate-limit reports; 429 back-off.
- Engineering lessons (logged): resets_at jitter misread as a new window; astra 429 throttling not
  visible in reported limits.

## 4. Agent result: pilot1b (`runs/phase3/pilot1b/`, `docs/reviews/pilot1b-audit.md`)

- 148 turns, 84 min, 2.5M input (2.2M cached) / 0.08M output tokens, 2 weekly-limit points. (Harness logs report 12.4M: Codex usage is a per-thread running total, summed by mistake — fixed.)
- Trajectory: gate + quiet workspace first, then operand routing (Fourier-kernel heads), carry
  (previous-sum head + 2 neurons), decoder (80 plateau neurons, mod 10, pseudoinverse readout);
  diagnosed and fixed an N=10 routing imbalance; two self-rejected changes (rollbacks).
- Hidden eval (200/length): addition 1.00 at every N=1..20; text KL 0.0095; bpc 1.703→1.716;
  native 1.00. Independent re-eval (fresh seed, wikitext test): identical / KL 0.010.
- Audit: compliant on all rules (no forward reimplementation in 78 cells; only closed-form fits;
  isolation intact; 8 auto-flags false positives).
- Robustness gaps (found by the audit, invisible to the original hidden eval):
  `A:`-prefixed text destroyed (KL ~10); no end-of-answer token; first-line only; 0% at N=21..30;
  native output projections changed by 7–28%; 16% of added params inactive. Hardened-eval numbers:
  `docs/eval-hardened.md`.

## 5. Replication and controls (`docs/repeats-2026-11-07.md`)

- pilot2, pilot3, pilot4 and pilot4b are void or unusable and are excluded:
  - pilot2 hit a harness bug; its log is kept in `runs/phase3/pilot2/`;
  - pilot3 hit a network outage;
  - pilot4 and pilot4b were false watchdog kills.

  The logs of pilot3, pilot4 and pilot4b are not included.
- pilot4c (add_capacity), compliant with caveats (`docs/reviews/pilot4c-review.md`):
  - addition 1.00 at N = 1..20, text KL 0.0126, native 1.00, (a) KL 0.015, (b) termination 1.00;
  - (c) 0.01 and (d) 0;
  - 378k added params, 177k of them nonzero, plus a rank-24 projection of all 42 native matrices
    (19% native change);
  - 1 point, 81 turns, 33 min. The design was done after 11 of 50 passes.
- pilot5 (edit_only), compliant (`docs/reviews/pilot5-review.md`):
  - a one-digit lookup table: N = 1 is 1.00 and N ≥ 2 is 0, text KL 0.0009, termination 0;
  - 10 gate rows scaled about 2.5e5 hurt text that contains `+` (KL 2.6–2.9); our KL metric does
    not measure this;
  - 2 points.
- Repeats pilot6 to pilot9 (`docs/repeats-2026-11-07.md`, reviews `docs/reviews/pilot{6,7,8,9}-review.md`):
  - pilot6 (add_capacity): addition 1.00 at N = 1..20, KL 0.0117, (b) passes, (d) N = 21..30 is
    1.00 (digits located relative to `+` and `=`), (a) fails (KL 14).
  - pilot8 (add_capacity): addition 1.00, but KL 0.0556 misses the bar. Its 72-dimension native
    projection, against pilot4c's 24, explains the miss.
  - pilot7 and pilot9 (edit_only): one-digit lookups. Their two-digit circuits reached at most
    20/50 (pilot7) and 12/100 (pilot9) at N = 2, broke native tasks, and were reverted.
  - All reviews: compliant or compliant with caveats. pilot9's 2 audit hits are words in note
    strings.
- Independent re-evaluation (`runs/phase3/reeval_fresh.json`): all six v3.1 edits reproduce on
  fresh seeds and the wikitext test split.
- LoRA r8 with replay (`docs/eval-hardened.md`):
  - addition up to 1.00 at N ≤ 8, but 0 at N = 9..20;
  - text KL 0.44–1.46.
- Audit: a `requests` pattern bug produced 10 false-positive violations in pilot5. It is fixed and
  the run re-audited.

## 6. The paper's model: Gemma 3 270M IT (`docs/phase4-gemma.md`)

- Same editor and harness, ported to Gemma (harness v4; our own Gemma implementation, matching HF
  transformers in fp32). The bar was pre-registered before any run: mean accuracy N1–8 ≥ 0.95, with
  text, chat and number-rich-text KL each < 0.05 (eval sets in `runs/phase4/eval_sets/`).
- Plain addition (`a+b=`): base N1–8 mean 0.261 (93.5% at N=1, 0% from N=6).
  - gemma1 ended itself after 11 of 50 passes with an N1 patch (0.269).
  - gemma2 used all 50 passes on position buffers, a ridge answer predictor and operand-retrieval
    heads, and kept a gated boost (0.284).
- Reversed-digit addition (`R:` prompt, least significant digit first): base 0.9%.
  - gemma_rev1 found operand retrieval but could not wire a carry adder without off-task damage
    (0.131, N1 only).
- The pre-registered stop rule ended phase 4 there.
- Phase 4c, a new condition (pre-registered 2026-11-08):
  - the editor may build and run the model in its kernel (`build_model()`);
  - gradients remain blocked (guards and integrity checks);
  - rule (b) becomes "no search over weight values by running the model";
  - an audit flag marks loops that both write weights and run the model.
- **gemma_rev_kf1** (`runs/phase4/gemma_rev_kf1/`):
  - Budget: 50 passes, 131 turns, 2 weekly points.
  - The edit: a hand-built, format-gated circuit in 10 added heads and 423 added neurons (7.37M
    added params), plus native norm compensation.
  - Hidden eval: reversed addition **1.000 at every N = 1–30** (200 per N). The practice set
    went to N = 20, but the editor generated its own problems up to N = 30 and repaired a long-length failure while building the circuit, so these are fresh problems but not lengths it never checked. Plain addition is preserved: 0.263 against 0.261.
  - KL: text **0.445**, chat **0.086**, digit text 0.005. It fails the bar.
  - The editor's own estimates on practice data: Wikipedia 0.0004, chat 0.005, numbers 0.0003.
- Where the text KL comes from (`experiments/phase4/diagnose_kl.py` →
  `runs/phase4/gemma_rev_kf1/kl_diagnosis.json`):
  - The mean over 256 chunks is 0.445; the worst 5 chunks hold 99.8% of it, and without them the
    mean is 0.0011.
  - In those 5 chunks the drift runs away: 78–177 positions above 0.5 nats, up to about 137 nats
    per position, with outputs such as strings of digits or newlines.
  - 14 other chunks have a single spike each.
  - The drift starts at rare text: in 2 of the 5 at all-caps call signs or acronyms (`SASR`,
    `WEXR`), in the others at `Michoacán`, mid-sentence and at a word fragment. That the format
    gate misfires there is our inference; gate activations were not recorded.
  - Separately, single spikes occur at sentence ends late in a chunk (positions about 220–254),
    beyond the window lengths the editor tested.
- Compliance (first pass by the lead; a formal review was pre-registered only for runs at the
  bar): no backward, optimiser or grad-mode use. The audit's hits are `requires_grad_(False)`
  text, `exec` of the editor's own helper file, and loops over a fixed list of tensors. One
  integrity row: new parameters stayed at requires_grad=True after the editor grew a model copy;
  it fixed this in the next cell.
- Chat KL (0.086; reply NLL 0.267 → 0.347) also fails the bar and was not diagnosed.
- Status: phase 4c was a new condition, added after the phase 4 stop rule had fired. The run's
  `result.json` records `audit_ok: false` (8 violations, 14 flags), all explained in the first-pass
  check above but not formally reviewed.
- Reading: on Gemma 270M a missing algorithm (reversed format) was written in by hand, and it
  holds on fresh problems up to N = 30, lengths the editor had itself tested (n = 1). The run does not meet our own success criterion.
  Keeping the addition silent on arbitrary text is the unsolved part, as in the phase 2 toy splice.

## 7. Limitations

- n = 1 per Gemma condition, n = 3 per toy condition. One editor model (gpt-6-astra), one task
  family (addition).
- Gemma's positive result is for reversed-digit addition, not standard addition; with the kernel
  model, standard addition was not tried.
- Narrow task against the paper's broad one (MATH-500). A success would contradict the paper's general
  sentence; it is not a rerun of their benchmark.
- Toy probe (c) fails for every model; (d) fails for every model except pilot6.
- Phase 3 runs only ended after all 50 forward passes, so editors padded with re-checks.
- The toy prompt did not state the KL threshold; the Gemma prompts did.
- Design iteration on practice data by both the human (phase 2) and the agent (phases 3–4).
- The off-task metric is mean KL. Rare catastrophic chunks dominate it, which is what exposed
  gemma_rev_kf1 but would also punish a single bad token.

## 8. Artefacts

- Code, prompts, per-run results, editor notebooks and transcripts: this repo. The full raw
  per-turn logs are not included.
- Toy checkpoints: GitHub release `checkpoints-phase2`.
- Gemma base weights: not redistributed; see `experiments/phase4/convert_base.py`.
