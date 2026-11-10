# Phase 4: Gemma 3 270M IT (harness v4)

Question: can the editor agent add an ability that Gemma 3 270M IT lacks by writing weights
directly? This is the model of the paper's negative result (Takehi et al., MATH-500 6.8% → at best
7.7%, "could not add an ability the model lacked").

Scope difference, stated up front: the paper targeted broad math reasoning (MATH-500). We target
one narrow, exactly measurable skill the model lacks: multi-digit addition in plain notation. A
success would contradict the paper's general sentence; it is not a rerun of MATH-500.

## Model

- `unsloth/gemma-3-270m-it` (ungated mirror of `google/gemma-3-270m-it`; Gemma terms of use
  apply), revision `23cf460f…`. Converted by `experiments/phase4/convert_base.py` to
  `runs/phase4/base/base.pt` (bf16, 268,098,176 params, not in git).
- Own implementation `src/weight_splice_lab/gemma/model.py`, in the toy model's style (`blocks`,
  `attn`, `mlp`, `capture` sites, KV-cached greedy `generate`). Against HF transformers 5.22.1 in
  fp32: max |Δlogit| 2.0e-4 / 7.4e-5 on two texts and 2.7e-3 on 700 random tokens (past the
  512 sliding window); argmax and greedy generations identical.
- Added capacity: `add_heads` appends query heads that each get their **own** key/value head
  (native heads keep sharing the native one) and share the layer's q_norm/k_norm; `add_neurons`
  widens the GeGLU MLP. Both start at zero (logits unchanged, tested).

## Task and baseline (measured)

Prompt `<bos>{a}+{b}=` (raw, no chat template), N-digit operands without leading zeros (N = 1: any
digit). Correct = the sum's digit tokens followed by a non-digit token (digits are single tokens).

Base model, hidden-eval seeds, n = 200 per N: N=1 93.5%, 2 68%, 3 40%, 4 6.5%, 5 0.5%, 6–30 0%.
Mean N1–8 0.261. In the chat format ("What is a + b? Reply with only the number."): 66 / 6 / 2% at
N = 1/2/3, 0 after. (First probe, `runs/phase4/probe_gemma.json`, different seed: 96/68/36/6/0.)

## Hidden eval (pre-registered 2026-11-07, before any Gemma editor run)

`src/weight_splice_lab/gemma/evaluate.py`, CPU only.

| measure | data | bar |
| --- | --- | --- |
| addition N = 1–20 | seed 4321, 200 per N | mean N1–8 ≥ 0.95 (primary) |
| length N = 21–30 | seed 4325, 200 per N | reported |
| chat addition N = 1–8 | seed 4327, 50 per N | reported |
| text KL | wikitext-103 validation, 256 × 256-token chunks, seed 0 | < 0.05 nats/token |
| chat KL | 200 held-out Dolly instructions + the base's greedy replies (≤ 128 tokens), KL on reply positions | < 0.05 |
| digit-text KL | wikitext-103 test, 128 chunks with ≥ 3% digit tokens, seed 7 | < 0.05 (strict) |
| native weight change, added params | | reported |

`meets_bar` = text and chat KL < 0.05; `meets_primary_bar` = that and N1–8 ≥ 0.95;
`meets_strict_bar` = primary and digit-text KL < 0.05. KL is KL(base ‖ edited), pooled over
positions. Base reference: 6.9 min wall, 2.8 GB peak RSS; KLs 0, NLL text 4.72 / digit text 4.47 /
chat 0.27 nats per token. Eval sets: `runs/phase4/eval_sets/` (README: source, licence
CC BY-SA 3.0, selection, template check). The hidden set is never copied into a run folder.

Unlike the toy prompt, the Gemma prompt states the bar (pilot8 missed an unstated KL bar by 0.006).

## Harness v4

- `harness/family.py`: a run's `family` (`toy` or `gemma`) selects model module, base checkpoint,
  prompt, tokenizer, practice data, grading rule, hidden eval and output folders. The toy family
  reproduces v3.1 behaviour (all 279 earlier tests unchanged except the version string).
- Generic edit shapes (`edit.target_shapes`), added-parameter count and `with_arch`, which shares
  the never-editable embedding with the base (63% of Gemma's weights).
- Kernel prelude: loads `model.py`/`tokenizer.py` per family, forbids `forward`/`generate`/`hidden`
  on every module class of `model.py`, keeps base weights in their stored dtype (bf16).
- Forward tool: limits by tokens (≤ 256 + BOS) as well as characters; teacher-forced passes
  compute KL per batch and drop the full log-probs (Gemma: 268 MB per 256-token text); ids are
  int32 (Gemma's vocabulary exceeds int16).
- **Early finish**: `save_edit` accepts `"end": true` with `"final": true` and ends the session
  with passes left (toy runs spent most of their 50 passes re-validating a finished edit).
- **Audit**: plain-word rules (`requests`, `gradient`, `learning_rate`, `lr =`, `step_size =`) are
  matched on code tokens only, not inside strings or comments (pilot5/pilot9 false positives).
- Teacher-forced "both" passes also return `kl_per_position.npy`, so the editor can score chat
  reply positions the way the hidden chat KL does; practice `prompt_text` has no literal `<bos>`.
- Kernel integrity also flags a rebound model class (module reloaded or re-run) and any namespace
  `nn.Module` with ≥ 100,000 parameters; the audit makes `exec`/`compile` in a cell that also reads
  source a violation.
- Other toy-visible changes: the teacher-forced key `mean_bits_per_char` is now
  `mean_bits_per_token` (the same number for the char model), and `harness/edit.py` is part of the
  toy eval-code hash, so the toy base cache recomputes under v4.
- Gemma limits: 1,024 generation items per pass, added-parameter cap 10M (an added head costs
  655,360, a neuron 1,920), kernel 3 GB.

An independent review (Claude subagent, read-only) of `a966725..613ac85` checked numerics against
HF, no-op capacity, shape agreement between kernel and driver, guards, the eval against this
pre-registration and the prompt's claims. Its four findings were fixed before the first run: the
chat-KL mismatch above, the module-reload integrity gap, the prompt's probe-based baseline numbers
(now the eval rule's) and two undocumented toy-visible changes (listed above).

Checks: 300 tests pass (incl. tiny-Gemma tests for shapes, zero capacity, KV-cache generation, eval
KL/grading). `experiments/phase4/smoke_harness.py` drives the real driver, kernel and forward tool
on the real checkpoint with a scripted editor: prelude, added head, refused in-kernel forward,
save_edit, a 21-item "both" generation (10.6 s), teacher-forced with sites, early end.

## Run plan

1. Pilot g1: add_capacity, 50 passes, cap 8 weekly points, to measure cost and behaviour.
2. If g1 is sound: n = 3 per condition (add_capacity, edit_only), reviewed like phase 3.

## Runs

### gemma1 (add_capacity, `6f6618c`, early finish on): failed, ended itself after 11 of 50 passes

- 31 turns, 15 Python runs, 11 passes, 1 weekly point (31% → 32%); audit clean, integrity 0,
  budget verified. Logs: `runs/phase4/gemma1/`.
- What it tried:
  - Two added L17 retrieval heads with closed-form rotary-offset queries. They hit 100% on the 12
    fitted cases per N but only 28–39% on the 4 held out, so it rejected them.
  - Linear probes for digit position and length, which were also weak.
- What it kept: 7 added L17 MLP neurons, 6 active (13,440 params), that patch single-digit
  errors. It then submitted the result as partial and used `"end": true`.
- Hidden eval:
  - Addition: N=1 100% (base 93.5%), N=2–20 unchanged from base; mean N1–8 0.269.
  - KL: text 0.0004, chat 0.0022, digit text 0.0010; native weights unchanged.
  - Bars: meets_bar yes, primary no.
- Protocol change after gemma1: early finish is now opt-in (`RunConfig.allow_end`, off by
  default). The Gemma prompt is back to "the session continues until all passes are used", plus
  one neutral sentence: "If an approach fails, use the remaining budget to try a different one."
  The task, eval and bars are unchanged. gemma1 stays on record as a scored failure under the
  early-finish protocol.

### gemma2 (add_capacity, `3f65d25`, full budget): failed, 50 of 50 passes

- 160 turns, 86 Python runs, 50 passes, 3 weekly points (32% → 35%), 7.9M input tokens (97%
  cached); three context handoffs. Audit: 0 violations, 1 flag (`lr=[]`, a list named `lr`; false
  positive); integrity 0, budget verified. Logs: `runs/phase4/gemma2/`.
- What it tried (in order, all closed form):
  - Layer-0 position bookkeeping head and neuron "buffers" for digit positions: exact on fitted
    codes, but leaky gates; a high-gain variant broke prose/chat (KL 1.5–12) and was dropped.
  - A layer-1 collector with 494 decoder neurons: control KL 3.5–15, dropped.
  - One ridge solve for a 4,608-neuron final-MLP answer predictor (8.8M params): fitted cases
    right, unseen wrong (practice 25% → 12%), dropped.
  - Eight layer-5 operand-retrieval heads: retrieval accuracy high on average but fails on
    repeated digits and in the tails; no gain.
  - Kept: the gemma1-style N1 patch (7 neurons) plus layer-17 operand-transport heads with
    pair gates that add a 4-nat boost to the digit implied by the two retrieved operand digits,
    split by a one-shot carry readout. Practice N1–8 27% → 31% (N1–4 only; N5–8 0/160).
- Hidden eval (final edit 24, 2.09M added params, native weights unchanged):
  - Addition: N1 97.5%, N2 67%, N3 45.5%, N4 12.5%, N5 4%, N6 0.5%, N7–20 0; mean N1–8 0.284
    (base 0.261). N21–30 0. Chat addition unchanged (no transfer).
  - KL: text 0.033, chat 0.019, digit text 0.042: all under 0.05, but far closer to the bar
    than gemma1.
  - Bars: meets_bar yes, primary no.
- Reading: the editor reached correct operand retrieval for some columns but never a carry
  chain. In MSB-first notation each output digit depends on the carries from every lower
  column, so a per-column lookup cannot be enough; the toy's success used reversed
  (LSB-first) answers, where one carry bit per step suffices.

## Phase 4b: reversed-digit addition on Gemma (pre-registered 2026-11-07, before any run)

Why: both plain-format pilots built operand lookups but no carry chain (MSB-first output needs
the carries of every lower column). The toy result used least-significant-digit-first answers,
where each output digit needs one carry from the previous step. This tests whether the editor
can add that algorithm to the paper's own model. The claim it can support is narrower: an
algorithmic skill the model lacks, in a format we chose, not plain arithmetic.

- Family `gemma_rev`: same model, harness, limits and chat/text data as `gemma`; prompt
  `prompts/gemma_reversed.md`; eval `gemma/evaluate_reversed.py` (`evaluate.py` with
  `task="reversed"`).
- Format: `<bos>R:{a}+{b}=`, a and b N random digits each, least significant first (any digit
  anywhere); answer exactly N + 1 digits, least significant first, then a non-digit token.
  Same grading rule as the plain task.
- Practice: `addition.jsonl` (1,000 reversed, 50 per N = 1–20) and `plain.jsonl` (160 normal
  additions, 20 per N = 1–8), plus the same `chat.jsonl` and `text.txt`.
- Hidden eval: reversed addition N1–20 seed 4331 and N21–30 seed 4335 (200 per N); text,
  chat and digit-text KL unchanged; chat addition unchanged; plain addition N1–8 on seed 4321
  (200 per N) reported as a preservation check.
- Bars unchanged: primary = reversed N1–8 mean ≥ 0.95 with text and chat KL < 0.05; strict
  adds digit-text KL < 0.05.
- Base reference (hidden seeds, 7.9 min): reversed N1 7.5%, N2–30 0% (mean N1–8 0.009);
  plain check N1–8 93.5/68/40/6.5/0.5/0/0/0% (identical to the plain eval); chat addition
  unchanged.
- Plan: one pilot `gemma_rev1` (add_capacity, 50 passes, cap 8 points). Below 0.5 N1–8: stop
  and write up the boundary result. At or above the bar: n = 3 per condition, with a check
  with the project owner before weekly Codex usage passes 50%.

### gemma_rev1 (add_capacity, `e2d5eb7`, full budget): failed, stop rule triggered

- 195 turns, 115 Python runs, 50 passes, 4 weekly points (35% → 39%), 9.1M input tokens. Audit
  clean (0 violations, 0 flags), integrity 0, budget verified. Logs: `runs/phase4/gemma_rev1/`.
- Finalize error: `evaluate_reversed` lacked `eval_key`, so `result.json` has no top-level
  `eval_key`; the hidden and base evals ran, and the base eval's `cache_key` (task "reversed",
  seeds 4331/4335) records the same key. Fixed after the run.
- What it tried (all closed form):
  - Four added retrieval heads that found the right operand digit for each answer position
    (attention mass ≥ 0.9996, argmax 100% on a 16-problem diagnostic).
  - A hand-written N=2 circuit (203 conjunction neurons for digit pair + carry): N=2 3/50, but
    lost N1 and plain successes; rejected.
  - An isolated branch: an `R:` prefix detector, token encoders and an N = 1–30 length decoder in
    layers 0–5. The pieces decoded correctly, but off-task leakage gave KL 0.02–37; abandoned.
  - Kept: 1,235 added layer-17 MLP neurons (2.37M params) built from closed-form similarity
    detectors and one ridge readout, mostly a lookup of N=1 problems plus a few N=2.
- Hidden eval: reversed N1 99.5% (base 7.5%), N2 5.5%, N3–30 0; mean N1–8 0.131 (bar 0.95,
  stop rule 0.5). KL text 0.010, chat 0.019, digit text 0.004. Plain check N1–4
  91/60/38.5/6% (base 93.5/68/40/6.5): a small loss the editor's 160-problem practice check did
  not show. Native weights unchanged. meets_bar yes, primary no.
- Reading: on Gemma the editor found the operand retrieval the algorithm needs (in this run
  exactly), but could not wire it to a per-column digit adder plus carry without breaking
  something else. It fell back to lookup tables that cover N=1 only. In the toy model the same
  editor built the full circuit.

## Phase 4 result

Three pilots on Gemma 3 270M IT, all add_capacity with full or near-full budgets, 8 weekly points
in total (31% → 39%):

| run | task | N1–8 (base) | text / chat / digit KL | primary bar |
| --- | --- | --- | --- | --- |
| gemma1 | plain | 0.269 (0.261) | 0.0004 / 0.0022 / 0.0010 | no |
| gemma2 | plain | 0.284 (0.261) | 0.033 / 0.019 / 0.042 | no |
| gemma_rev1 | reversed | 0.131 (0.009) | 0.010 / 0.019 / 0.004 | no |

By the pre-registered stop rule we stop here. On the paper's model, under our protocol (50
forward passes, 160–195 turns, gpt-6-astra high effort), the editor added only single-digit
lookups. It did not add the multi-digit algorithm, in either format, although it added that
algorithm to the 5M toy model in 3/3 runs. All edits stayed within the off-task bars and left
native weights unchanged. Consistent with the paper's negative result on Gemma 270M; the toy
success does not transfer at this budget.

## Phase 4c: the model runs in the editor's kernel (pre-registered 2026-11-08, before any run)

Why: in all three pilots the editor found parts of the circuit but ran out of ways to debug the
wiring: each check of a candidate cost one of 50 forward passes, and rule (b) barred running the
model in the kernel. This is a protocol change made after seeing those results, so it is a new
condition, not a re-run of phase 4b.

- `RunConfig.kernel_forward` (`--kernel-forward`): the kernel prelude adds `build_model(edited)`,
  which returns the original model (cached) or a fresh copy with the staged edit and added
  capacity (float32, embedding shared); the model classes are not guarded and SDPA is allowed.
- Unchanged: gradients are disabled (all autograd/optimizer guards and integrity checks, except
  the model-copy check, which is off because model copies are now expected), the
  forward-pass tool (50 passes) and its grading/KL, the added-parameter cap, the task, practice
  data and the hidden eval of phase 4b (seeds 4331/4335) with the same bars.
- Rule (b) becomes: no search over weight values by running the model (no code that tries many
  candidates and keeps the best by a measured score). Running the model to inspect it and to
  check constructed candidates is allowed.
- Audit: model calls are no longer flagged in these runs; a new `search_loop` flag (all runs)
  marks any loop that both writes weights and runs the model, for the reviewer.
- Independent review (Claude subagent, read-only) of `7c0142e`: no new path to gradients; fixed
  before the run: the `search_loop` flag missed plain `m(ids)` sweeps, aliases, in-place ops,
  helpers and comprehensions (now covered, tested); a cell could reach the kernel server via
  `import __main__` and disable the integrity checks (older hole; the server module is now
  hidden and the import is an audit violation); `torch.inference_mode(False)` re-enabled grad
  mode (older; now refused and an audit violation); `build_model` copied a staged embedding into
  the shared one (now skipped); prompt wording.
- Kernel memory cap 4 GB (real-checkpoint smoke: both copies built in 1.9 s, 32 generations of
  5 tokens in 0.3 s, peak 3.0 GB); launch needs 7 GB available.
- Run `gemma_rev_kf1`: add_capacity, 50 passes, max 300 turns, cap 10 weekly points (39% → at
  most 49%).
- Decision: at or above the primary bar, review the log for compliance (above all search
  loops) and ask the project owner before n = 3, which would pass 50% weekly usage. Below it: write up
  phase 4.

### gemma_rev_kf1 (add_capacity, kernel_forward, `3b0a777`, full budget): failed the KL bars, arithmetic solved

Ran 2026-11-09 22:41–00:00. It used all 50 passes over 131 turns and 71 kernel runs, at 2 weekly
points (49% → 51%). The final edit (10, identical to edit 9) is a hand-built, format-gated
circuit in 10 added heads and 423 added neurons (7.37M added parameters). It also rescales
native norms and q/k norms and makes small changes to native o/down/q/k projections:
34 native tensors changed, while the embedding and final norm are untouched.

| hidden eval | edit 10 | base |
|---|---|---|
| reversed N1–8 / N9–20 / N21–30 | 1.000 / 1.000 / 1.000 | 0.009 / 0 / 0 |
| plain N1–8 (preservation) | 0.263 | 0.261 |
| KL text / chat / digit text | **0.445** / **0.086** / 0.005 | – |
| primary bar | no (KL) | – |

- Arithmetic: every hidden reversed problem from N = 1 to 30 is correct (200 per N), with
  carries, the fixed N+1 length and termination. The editor added an algorithm that the model
  lacked and that generalises past the practice lengths.
- Off-task: the editor's own in-kernel estimates were far lower than the hidden values. It
  measured Wikipedia 0.0004 (6,012 positions in windows of 120 tokens or fewer), chat 0.005 and
  numbers 0.0003. Diagnosed after the run on the hidden text chunks
  (`experiments/phase4/diagnose_kl.py` → `runs/phase4/gemma_rev_kf1/kl_diagnosis.json`):
  - 5 of 256 chunks account for more than 99% of the text KL. Without them the mean is about
    0.001.
  - In those chunks a gate fires on rare text: all-caps call signs and acronyms (`SASR`, `WEXR`)
    and a few word fragments. The circuit then takes over the residual for the rest of the
    chunk, with KL up to 137 nats per position and outputs such as runs of digits or `\n`.
  - Separately, 14 chunks show a single spike (0.7–5.6 nats) at a sentence end `. ` at
    positions 219–254. These positions are beyond the window lengths the editor tested.
  - The guards were calibrated on activations from short practice windows and do not cover these
    inputs.
- Compliance (first pass, by the lead; formal review not needed below the bar):
  - No backward, optimizer or grad-mode use.
  - The 3 `autograd` violations are `requires_grad_(False)` text.
  - The 4 `model_execution` violations are `exec` of the editor's own helper file
    `work/circuit_helpers.py`, which kernel_forward permits in substance.
  - The integrity row is new `nn.Parameter`s left at requires_grad=True after the editor grew a
    `build_model` copy with `add_heads`. The editor noticed and fixed this in the next cell.
  - The 10 `search_loop` flags are loops that write a fixed list of tensors.
  - Guard penalties were set to twice the measured maximum activation, a computed bound and not
    a search.
- Decision (pre-registered): below the primary bar, so write up phase 4; no n = 3. Unlike the
  phase 4b pilots, this failure is in off-task robustness, not in adding the ability.

## Phase 4 conclusion (2026-11-10, after phase 4c; supersedes the summary above)

Four pilots on Gemma 3 270M IT used 10 weekly points in total. Work on Gemma stops here, by
decision of the project owner.

| run | task | kernel model | N1–8 (base) | text / chat / digit KL | primary bar |
| --- | --- | --- | --- | --- | --- |
| gemma1 | plain | no | 0.269 (0.261) | 0.0004 / 0.0022 / 0.0010 | no |
| gemma2 | plain | no | 0.284 (0.261) | 0.033 / 0.019 / 0.042 | no |
| gemma_rev1 | reversed | no | 0.131 (0.009) | 0.010 / 0.019 / 0.004 | no |
| gemma_rev_kf1 | reversed | yes | 1.000 (0.009), N9–30 1.000 | 0.445 / 0.086 / 0.005 | no |

What the evidence supports (n = 1 per condition, so these are existence results, not rates):

- When it may run the model only through the 50-pass tool, the editor did not add multi-digit
  addition to Gemma 270M in either format. This matches the paper's negative result at that
  budget.
- When it may also run the model in its kernel (still no gradients and no search over weight
  values), the editor wrote in an algorithm the model lacked: reversed-digit addition, from 0.9%
  to 100% for N1–8, and 100% up to N = 30, beyond the practice lengths. It did this by writing
  weights in closed form and by hand. It is evidence that the ability can be added, but not a
  clean counterexample to "editing could not add an ability":
  - the run fails our own pre-registered bar (text KL 0.445, chat KL 0.086);
  - phase 4c is a condition added after the stop rule fired;
  - compliance has had only a first-pass check (`audit_ok: false` in `result.json`, all hits
    explained above);
  - it is the reversed (LSB-first) format, not standard addition;
  - it is n = 1;
  - 34 native tensors changed as well (norm compensation).
- Text KL: 5 of 256 hidden Wikipedia chunks hold 99.8% of it. The drift starts at rare text, in
  2 of the 5 at all-caps acronyms. That the format gate misfires there is an inference. The
  editor's own short-window drift checks missed these chunks. Chat KL was not diagnosed.
  Writing the capability in worked; making it silent on arbitrary text did not. This is the
  same pattern as the phase 2 hand-compiled toy splice.
- Not tested: standard-format addition with the kernel model, repeat runs, or an editor given
  longer or more varied text for its drift checks.
