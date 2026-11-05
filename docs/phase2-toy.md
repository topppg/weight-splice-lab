# Phase 2: toy testbed and hand-compiled reference splice

Goal: a small transformer that has a native digit interface but cannot add, plus a splice written
analytically (no gradients on the target) that gives it multi-digit addition. The splice is the
existence proof and ceiling for the Phase 3 agent; a LoRA baseline gives the data-cost comparison.

Code: `src/weight_splice_lab/toy/` (`model.py`, `data.py`, `train.py`, `metrics.py`, `splice.py`,
`lora.py`), scripts in `experiments/phase2/`, tests in `tests/toy/`, artefacts in `runs/phase2/`.

## 1. Target model

| | |
|---|---|
| Architecture | decoder-only, pre-norm, RMSNorm, RoPE (rotate-half, base 1e4), SwiGLU, no biases, MHA |
| Size | 6 layers, d_model 256, 8 heads x 32, d_mlp 768, 5.14M params |
| Embeddings | tied |
| Tokenizer | char-level: `<pad> <bos> <unk>`, `\n`, printable ASCII (99 tokens); every digit is one token |
| Context | BOS-prefixed, 256 chars |

Why tied embeddings: Qwen2.5-0.5B, Qwen3.5-0.8B and Gemma 3 270M all tie input and output
embeddings, so the splice's read interface (embedding geometry) and write interface (unembedding)
are the same matrix, which is the constraint the Phase 4 splices will have. Untied would make the
toy easier in a way that doesn't transfer.

What doesn't match Phase 4 targets: there's no GQA (adding a head in a GQA model also means adding
or sharing a KV head) and no QK-norm (Qwen3 and Gemma 3 have it; see section 6).

### Training mixture

Each 256-char training sequence is either a random wikitext-103 window (50%) or a pack of native
digit tasks (50%), one per line:

- `C:<x>=<x>`: copy a 1-12 digit string
- `R:<x>=<reverse x>`: reverse
- `M:<x>,<y>=<max>`: compare two integers

So the model gets digits, `=`, `,`, `:` and a "read digits from earlier positions and emit them"
interface, but never sees addition. `+` only shows up in wikitext prose.

### Target task

`A:<a little-endian>+<b little-endian>=<a+b little-endian, N+1 digits>\n`, where `a` and `b` are
both N-digit strings (leading zeros allowed). Example (N=3, 893+952=1845): prompt `A:398+259=`,
answer `5481` then `\n` (`data.addition_example`). The base model never sees `A:` or digit
addition.

Why reversed operands and output: a causal model emits the units digit first, and the carry flows
the same way as generation. More importantly for a compiled circuit, with all-little-endian
operands the query at output step i needs `b_i` at relative offset N and `a_i` at offset 2N+1,
constant across i. With normal-order operands the offset is 2i+1 (a mirror), and RoPE can only
express fixed relative offsets. The carry into digit i can be computed locally from
`(a_{i-1}, b_{i-1}, s_{i-1})`, since `s_{i-1}` is the token currently being processed:
`c_i = [a_{i-1}+b_{i-1} >= 10] OR ([a_{i-1}+b_{i-1} = 9] AND [s_{i-1} = 0])`.

## 2. Splice API

`model.py` lets a splice add capacity, and every addition starts as an exact no-op:

- `add_heads(model, layer, n)`: appends `n` heads (q/k/v rows and o_proj columns, all zero).
  Since o_proj columns are zero, the output is bit-identical whatever q/k/v hold. Each layer can
  have its own head count.
- `add_neurons(model, layer, n)`: appends SwiGLU neurons with gate/up rows and down columns at zero.
- `insert_layer(model, index)`: inserts a pre-norm block whose matrices are all zero, so it's an
  identity.
- `save_model` / `load_model` round-trip models with non-uniform head and neuron counts.

`tests/toy/test_model.py` checks with `torch.equal` (not allclose) that the outputs are unchanged
after each addition, including when the new q/k/v or gate/up rows are random, and that a write to
o_proj does change them.

`metrics.py` measures regressions: held-out wikitext validation bits-per-char (256 x 256-char
chunks), KL(base || edited) on those chunks, and exact-match accuracy on the native tasks
(copy/reverse/max, lengths 1-12, 100 each). Target accuracy is exact match on the N+1 answer
digits, with an optional stricter version that also requires a `\n` terminator.

## 3. The hand-compiled splice (`splice.py`)

All splice weights are written in closed form by `build_splice(model, SpliceSpec, NativeActs)`
into capacity added with the API above. No native weight is changed. Notation: ν = median native
residual norm, d = 256, `E` = the tied embedding, `g_f` = the final RMSNorm gain.

### Interfaces

- **Private feature basis.** Splice features live in directions that are (a) in the null space of
  the unembedding `E·diag(g_f)` (SVD; d − V = 157 dims), so they never move logits directly, and
  (b) among the lowest-variance directions of native normalized residuals inside that null space
  (eigendecomposition). Small, noise-sensitive features get the quietest directions.
- **Reads.** A feature written with raw size U along `u` is read by an added row `u/g` (g = the
  RMSNorm gain of that site), so the read is `u·x/rms(x)`.
- **Token identity** comes from the token embedding. A layer-0 offset-0 "marker" head reads exact
  least-squares probes of the embedding (digit one-hot, `slot` = digit/`+`/`=`/`:`, `plus`) and
  writes them as small markers.
- **Write.** The answer token t is written along `w_t = pinv(E·diag(g_f))·(e_t − c_t·1)`: logits
  one-hot on t up to a uniform shift, with min-norm `c_t`.

### Circuit (rung `add`, lengths 1..N_max)

| Layer | Component | Function |
|---|---|---|
| L0 attn | N_max length-detector heads | at `=`, look N+1 back (RoPE query = key rotated by −(N+1)); value = is`+`; writes `Lraw_N` and the shared `E` (size U_c = 2ν) |
| L0 attn | marker head (offset 0) | digit one-hot, slot, plus markers (U_m = 0.3ν) at every such token |
| L1 attn | broadcast head | keyed on `E`, copies `Lraw`/`E` from the addition's `=` to all later positions (a negative `E` query term keeps `=` from copying itself) |
| L1 MLP | length cleanup, 2·N_max neurons | saturating step `silu(b(l−0.3E)) − silu(b(l−0.6E))`, up = l: writes clean `Lc_N` and the in-addition feature `A` with identical magnitude |
| L1 MLP | text energy, 2·(d−k) neurons | `T = Σ z_i²` over all native directions orthogonal to the private basis (`silu(bz)z + silu(−bz)(−z) ≈ b z²`); about 1 in text, about 0.1 inside an addition |
| L2 attn | 4 read heads (a_i, b_i, a_{i−1}, b_{i−1}) | RoPE positional attention with offset `Δ(N)` (2N+1, N, 2N+2, N+1), query multiplexed by `Lc_N`; values = digit markers (+ plus marker) |
| L2 MLP | one-hot cleanup, 2 per dim | steps against `A`: clean one-hots `r~j`, plus the constants `one~` and `zero~` |
| L3 MLP | carry, 2·55 neurons | `c_i = [a'+b' ≥ 10] ∨ ([a'+b' = 9] ∧ [s_{i−1} = 0])`, as narrow steps centred between the on and off sums |
| L4 MLP | lookup, 203 neurons | `(x, y, c) → (x+y+c) mod 10`, plus last digit (`a_i` is `+` → output the carry) and end-of-answer (`a_{i−1}` is `+` → `\n`), written to a private answer one-hot `ans` |
| L5 attn | emit head | attends to itself iff `A` is present (offset-0 filter keyed/queried by `A`), else to BOS; copies `ans_k` into `U_out·w_k` |

Rungs reuse subsets: `copy`/`successor` use one read head and a 10-neuron lookup; `add_nocarry`
drops the carry stage (ablation).

### The mechanisms that made it work

Each was found by debugging a failure (sections 5 and 7):

1. **Positional addressing with RoPE.** A query equal to the key template rotated by −Δ scores
   `Σ_f cos(θ_f(Δ_actual − Δ))` over four high-frequency pairs. The peak is 4, but the largest
   off-peak value within 160 positions is 3.36: only a 16% margin.
2. **Bias trick.** Keys put the slot marker both in the positional pairs and in a near-static bias
   pair, and the query subtracts `A·3.68`. The target is then the only position with a positive
   score whatever the key magnitudes. That matters because RMSNorm makes key magnitudes vary per
   position by more than the 16% filter margin.
3. **Multiplexing.** One head per role serves all lengths: the query is `Σ_N Lc_N·rope(−Δ_N)`,
   linear in the one-hot length code. The length code itself costs one detector head per N,
   because a RoPE model has no positional information in the residual stream: only attention
   patterns can measure the distance between `+` and `=`.
4. **Homogeneity.** Bias-free SwiGLU is degree-2 homogeneous in its normalized input, and RMSNorm
   rescales every read by 1/rms. So there is no linear "constant", and a threshold is only
   meaningful between features with the same RMSNorm/SwiGLU history (`one~` for cleaned
   features, `A` for raw ones). Mixing histories turned rms-estimate errors into threshold errors.
5. **Large features.** Large length/in-addition features (2ν) make the RMS at addition positions
   predictable (set by the splice, not by the native norm), so calibration constants computed from
   native norm statistics hold.
6. **Sinks and exponential gating.** Off-task silence comes from attention sinks: BOS, whose value
   is projected out of every added value read (exactly 0). Softmax gives exponential suppression;
   SwiGLU can only give a leak proportional to the noise floor.

## 4. Allowed-operations boundary

There is no gradient descent anywhere in the splice, and no fitting to target answers. Every
closed-form operation is logged in `SpliceReport.ops` (copied into `runs/phase2/splice_results.json`):

| # | Operation | Data | Purpose |
|---|---|---|---|
| 1 | SVD of `E·diag(g_final)` | own weights | unembedding null space for private features |
| 2 | eigendecomposition of native residual covariance (in that null space) | native activations (text + copy/reverse/max, 40k positions) | private basis ordered by noise |
| 3 | median residual norms per layer | native activations | magnitude calibration |
| 4 | ridge least-squares probes: const, BOS, `=`, `+`, slot at L0/L1/L2; digit one-hot at L0 (exact on the embedding) | native activations, including wikitext windows centred on rare `+`, `=`, `:` | read interfaces |
| 5 | second-moment eigendecomposition at L1 (⊥ private basis) | native activations | text-energy directions |
| 6 | projection of the BOS normalized residual out of every added value read | own activation at BOS (context-independent) | exact-zero sink |
| 7 | projection of the fixed template-tag residual (`A` right after BOS, L2 input) out of read keys | **one input-only activation of the target prompt prefix** | its deterministic slot-direction component otherwise won the bias-trick comparison |
| 8 | pseudo-inverse of `E·diag(g_final)` | own weights | answer write directions |

Constants (β's, scales, thresholds) are hand-chosen: about 15 scalars, listed in
`SpliceReport.constants`.

**Disclosure.** The design was reached by about 40 debug iterations in which I inspected the
splice on addition prompts: teacher-forced attention patterns, feature projections, score
decompositions, and accuracy at a few lengths. That is human design iteration on the target, not
optimization, but it is iteration, and it includes choosing a few constants (the sink strength,
the text-gate strength, the read scale, the ordering of private directions) after seeing target
accuracy and text KL. Item 7 uses one activation of the target's prompt prefix (no answer). A
strict reading of the "no iterative optimization on the target" rule should count the
constant choices; the Phase 3 harness should log the agent's equivalent choices the same way.

## 5. Results

### Base model

| | |
|---|---|
| Training | 3600 steps × 64 × 256 chars (59M chars), AdamW, cosine 2e-3, 1202 s on MPS |
| Peak memory | 1.27 GB MPS (micro-batch 16; batch 64 in one shot needed > 4 GB) |
| Held-out wikitext | 1.70 bits/char |
| Native tasks (len 1-12) | copy 1.00, reverse 1.00, max 1.00 |
| Target `add`, N=1..8 | 0.10, 0.00, 0.005, 0, 0, 0, 0, 0 (N=1 at chance) |
| Ladder baselines in the `A:` format | successor 0.00 at every N; copy 0.01-0.99 (the model partly reuses its native copy skill in the new format) |

### Splice: the accuracy and regression trade-off

`runs/phase2/splice_results.json`, probe seed 99, 200 prompts per length, greedy exact match on
the answer digits (`add` with the `\n` terminator is within 1 point of this everywhere except
gated seed 2). Regressions: wikitext bpc and KL(base‖spliced) over 256 × 256 chars, and native
tasks over 1200 examples each. **Acceptance bar: KL < 0.05 and every native task ≥ 0.98.**

**Ungated** (`read_scale=1.0, text_gate=0`): the capability ceiling.

| Rung | N_max | accuracy by N | bpc | KL | copy / rev / max |
|---|---|---|---|---|---|
| copy-a-digit | 8 | 1.00 at N=1..8; 0 at N=9,10 (no detector) | 1.81 | 0.079 | 0.56 / 0.33 / 0.50 |
| successor | 8 | 1.00 at N=1..8; 0 beyond | 1.81 | 0.079 | 0.56 / 0.33 / 0.50 |
| single-digit add | 1 | 1.00 | 2.29 | 0.42 | 0.25 / 0.11 / 0.19 |
| multi-digit add with carry | 8 | **1.00 at N=1..8** (with terminator too); 0 at 9, 10 | 2.62 | 0.65 | 0.17 / 0.06 / 0.09 |
| multi-digit add with carry | 20 | 1.00, 0.995, 0.97 (N=4), 0.96 (8), 0.95 (10), 1.00 (16), **1.00 (20)**; 0 at 21 | 3.11 | 0.99 | 0.10 / 0.08 / 0.06 |
| ablation: no carry stage | 8 | 0.61, 0.35, 0.11, 0.12, 0.08, 0.03, 0.02, 0.005 (≈ P(no carry) = 0.55^N) | 1.98 | 0.20 | 0.65 / 0.40 / 0.49 |

**Gated** (default `SpliceSpec`: `read_scale=0.15, text_gate=4`, strong sinks): off-task silence
first.

| Rung | N_max | accuracy by N | bpc | KL | copy / rev / max |
|---|---|---|---|---|---|
| copy-a-digit | 8 | 1.00, 1.00, 1.00, 0.99, 0.79, 0.85, 0.91, 0.79 | 1.713 | **0.007** | 0.93 / 0.69 / 0.74 |
| successor | 8 | 1.00, 1.00, 1.00, 1.00, 0.73, 0.92, 0.92, 0.81 | 1.713 | **0.007** | 0.93 / 0.69 / 0.74 |
| single-digit add | 1 | 0.84 | 1.735 | **0.025** | 0.97 / 0.70 / 0.73 |
| add with carry, seed 99 | 8 | 0.93, 0.97, 0.97, 0.99, 0.74, 0.12, 0.01, 0.05 | 1.762 | **0.043** | 0.96 / 0.67 / 0.74 |
| add with carry, seed 1 | 8 | 0.70, 0.92, 0.78, 0.85, 0.10, 0.80, 0.51, 0.34 | 1.753 | **0.037** | 0.94 / 0.47 / 0.72 |
| add with carry, seed 2 | 8 | 0.60, 0.84, 0.59, 0.77, 0.70, 0.22, 0.79, 0.29 | 1.757 | **0.039** | 0.84 / 0.42 / 0.56 |
| add with carry | 20 | 0.72, 0.99, 0.90 (N=4), 0.04 (8), 0.00 (10), 0.03 (16), 0.04 (20) | 1.811 | 0.079 | 0.96 / 0.68 / 0.72 |
| markers only (L0 marker head; everything else zeroed) | – | 0 | 1.711 | 0.007 | 0.98 / 0.70 / 0.73 |

"Seed" is the seed of the native sample used for the probes and the private basis (64 mixture
sequences plus 96 rare-token windows). Same design, same constants.

**Verdict against the bar.** No configuration meets it:
- The gated variant meets the text criterion (KL 0.007-0.043 for N_max ≤ 8).
- No variant keeps native digit tasks ≥ 0.98. The markers-only row shows the always-on layer-0
  digit/slot markers (0.3ν at every digit) alone take reverse to 0.70 and max to 0.73. The
  read interface itself, not the leaks, breaks the native digit circuits.
- Gating costs accuracy (and stability) beyond N ≈ 4.
- The ungated circuit is fully correct up to N=20 but wrecks the native interface.

### Length limit and parameter cost

| config | added heads | added neurons | private dims | params allocated | params nonzero |
|---|---|---|---|---|---|
| copy / successor, N_max 8 | 12 | 432 | 65 | 725k | 396k |
| add, N_max 1 | 8 | 685 | 115 | 788k | 586k |
| add, N_max 8 | 15 | 671 | 129 | 1.01M | 607k |
| add, N_max 20 | 27 | 647 | 153 | 1.38M | 644k |
| base model | 48 | 4608 | – | 5.14M | – |

The neuron count depends on N_max because the text-energy feature uses 2·(d − #private dims)
neurons; the logic itself is 2·N_max + 88 + 110 + 203.

- **Allocated vs nonzero.** Most allocated head parameters are zero: a detector uses about 12 of
  its 4 × 32 head rows/columns.
- **Hard length limit.** Each N costs a detector head plus 2 private dims, so the private budget
  (d − V = 157 dims) caps this circuit at N_max = 22. An earlier, leaner version (no text energy,
  no answer one-hot) ran N_max = 28 at 98-100%.
- **RoPE is not the bottleneck.** The 4-pair positional filter keeps a 16% margin out to 160
  positions.
- **Accuracy beyond N_max is exactly 0**, because the splice is inert without a matching detector.

## 6. LoRA baseline (`experiments/phase2/run_lora.py`, `runs/phase2/lora_results.json`)

Setup:
- LoRA rank 8 (α = 16) on every q/k/v/o/gate/up/down: 246k trainable parameters, about 25% fewer
  than the ungated N_max=8 splice's 607k nonzero.
- The base model is frozen. Training data: `budget` random addition examples with N uniform in
  1..8. Loss on the answer and `\n` only.
- 3000 steps × 64 (AdamW 3e-3, linear decay), so small budgets are many epochs.
- Evaluation is the same as for the splice.

| examples | acc N=1 | N=2 | N=3..8 | N=9, 10 (unseen) | bpc | KL | copy / rev / max |
|---|---|---|---|---|---|---|---|
| 100 | 0.10 | 0.01 | 0 | 0 | 1.80 | 0.069 | 0.09 / 0.02 / 0.14 |
| 300 | 0.23 | 0.03 | 0 | 0 | 2.07 | 0.26 | 0.07 / 0.02 / 0.08 |
| 1,000 | 0.77 | 0.04 | ≤ 0.01 | 0 | 3.86 | 1.50 | 0.01 / 0.01 / 0.02 |
| **3,000** | 1.00 | 1.00 | **1.00** | 0 | 3.83 | 1.49 | 0.00 / 0.00 / 0.01 |
| 10,000 | 1.00 | 1.00 | 1.00 (N=7: 0.06) | 0 | 5.26 | 2.48 | 0.00 / 0.00 / 0.01 |
| 30,000 | 1.00 | 1.00 | 1.00 (N=8: 0.09) | 0 | 3.53 | 1.28 | 0.01 / 0.00 / 0.00 |
| 100,000 | 1.00 | 1.00 | 1.00 | 0 | 3.97 | 1.58 | 0.00 / 0.00 / 0.00 |

Takeaways:
- **Data cost.** LoRA needs 1k-3k labelled examples to reach the ungated splice's ceiling (100% at
  N=1..8). With ≤ 1k examples it learns only N=1.
- **Neither generalises past its range.** LoRA trained on N ≤ 8 gets 0% at N=9, 10, and the splice
  gets 0% past N_max. The splice reaches N=20 by adding detectors (a few thousand nonzero weights
  each), with no data at all.
- **Regression.** Trained on addition only, LoRA forgets the native interface entirely (KL ≈ 1.5,
  native tasks ≈ 0). That is worse than the ungated splice (KL 0.65) and far worse than the gated
  splice (KL 0.04).
- **Update:** the replay runs are done; results and the hardened-eval rescore are in
  `docs/eval-hardened.md` (native replay restores native tasks ≥ 0.98, text KL stays 0.44-0.69).
- The fair regression baseline is LoRA with native-task replay; `--replay` is implemented. Its run
  was SIGKILLed by the memory watchdog because of system-wide swap pressure (53% swap from other
  applications, `runs/memwatch.log` 14:55), not this job's footprint (0.06 GB MPS), and per the
  safety rules it was not retried. Run
  `uv run python experiments/phase2/run_lora.py --replay --budgets 1000,3000,10000` when the
  machine is idle.

## 7. Why off-task silence is hard here (the main finding)

The capability can be added: the ungated splice is correct on N ≤ 20. Adding it *without
touching anything else* is the hard part:

- **There is no quiet subspace.** The lowest-variance direction of the native normalized residual
  has std 0.25 (mean 0.6); within the unembedding null space the floor is 0.30. A trained d=256
  model uses all of its residual stream. The splice needs 65-153 dims, so its features always sit
  on top of native activity and native layers always read them.
- **RMSNorm couples everything.** Inside an addition the splice's large features dominate the RMS,
  shrinking native signal about 4.8x. That is what makes in-addition calibration predictable, and
  it also means any read calibrated there is amplified about 4.8x on text. Text noise along a
  feature direction is therefore about σ·√k/16 of the feature (k = active large features) whatever
  the feature's size: roughly 3-10%.
- **Bias-free SwiGLU cannot gate.** Its output is degree-2 homogeneous, with no constant input, so
  "fire only if X" leaks in proportion to the noise floor times the gate gain. Every MLP stage
  leaks a little, and the answer write into the unembedding amplifies the leak into logit damage.
  The fixes that worked:
  1. Route the output through an attention head (softmax gives exponential suppression).
  2. A high-SNR "text energy" feature T: a sum of squares over about 100 native directions, so its
     noise averages out, whereas any single native probe, such as `const`, was ±23% with
     outliers at 0.
  3. Keep anything that can fire on text at marker scale.
- **The read interface itself regresses native digit tasks.** Token identity is clean only at the
  embedding (layer-0 probes are exact; layer-2 probes have uneven in-context gain, 0.45-1.04), so
  digit markers have to be written at every digit, everywhere. At 0.3ν they cost little text KL
  (0.007) but break the model's own exact digit tasks (reverse 1.00 → 0.70). At 0.5ν they cost
  KL 0.05. Smaller markers fall into the noise floor.
- **Fragility.** With about 130 of 157 null directions in use, which feature lands in a direction
  that carries heavy native interference at addition positions depends on the probe sample. The
  gated variant's accuracy swings from 0.1 to 0.99 at fixed N across probe seeds. Reordering
  features over directions changed failure modes. A deterministic interference from the fixed
  template tag (`A` after BOS) had to be projected out (op 7).

## 8. Implications for the Phase 3 agent harness

The reference splice needed, and the agent will need, these tools:

- **Capacity API** with exact no-op init (`add_heads`, `add_neurons`, `insert_layer`) and a
  parameter / private-dimension ledger. Budget accounting matters: the private subspace ran out
  before RoPE precision did.
- **Activation capture** per site (`resid_pre/mid`, attention patterns, MLP hidden), plus helpers
  to project residuals onto a named feature basis. Almost every bug was found by printing feature
  values at the 2-6 answer positions, layer by layer.
- **Closed-form toolkit**:
  - SVD/pseudo-inverse of the unembedding
  - covariance eigendecomposition (quiet directions)
  - ridge probes with a constraint subspace
  - a RoPE helper (`rope_query(key, offset)`) and a filter-margin checker
  - BOS/fixed-position projection
  All of these are logged automatically, as `SpliceReport.ops` does, so the allowed-ops boundary
  is auditable.
- **Score decomposition for heads**: per-key scores split into positional / bias / content RoPE
  pairs. That's how the `A`-token and slot-noise failures were found.
- **Fast regression gate**: text KL on 32 chunks (CPU, seconds) plus leave-one-out and cumulative
  build-up ablations by component. The leak was distributed, and only the build-up table pinned it
  down (section 7).
- **Probe-seed robustness check** as a standard eval: the same design varied from 0.1 to 0.99.
- **A forward-pass budget well above a few hundred**: this reference took about 40 design
  iterations and thousands of diagnostic forward passes. A harness that measures only end-to-end
  target accuracy would give the agent too little signal; per-position, per-feature diagnostics
  are needed.
- **An explicit acceptance criterion** (KL and native tasks) checked by the harness, not
  self-reported. "Accuracy 100%" was reached early, and every later iteration was about silence.

What should transfer to Phase 4 (Qwen3.5-0.8B / Gemma 3 270M):
- **QK-norm.** Qwen3 and Gemma 3 have it; it bounds attention logits, so the sharp positional
  heads here (scores in the hundreds) need large q/k-norm gains, or a different addressing scheme.
- **GQA.** New heads must share or add KV heads.
- **No unembedding null space.** Their vocabularies (about 150-260k tokens) are far larger than
  d (1024 and 640), so the unembedding has full column rank and there is **no null space** at all.
  Private features must instead be erased before the last layer, or live in directions the
  unembedding barely reads.
- **Prefer reusing native features.** Native features such as digit identity, which Qwen already
  computes, should be preferred over embedding-level markers, because writing always-on markers
  is what broke the native digit tasks here.

## 9. Compute and peak memory per run

Every GPU job ran alone, inside `weight_splice_lab.gpu.gpu_session` (exclusive lock, memwatch
running, per-process MPS cap of about 3.6-5 GB depending on free memory). MPS figures are
`torch.mps.driver_allocated_memory()` sampled at checkpoints; RSS is the process's
`ru_maxrss`. Unified memory: RSS includes the CPU side.

| Run | Script | Wall time | Peak MPS | Peak RSS |
|---|---|---|---|---|
| base training (3600 steps) + eval | `experiments/phase2/train_base.py` | 20 min train + 2 min eval | 1.27 GB | not recorded |
| splice evaluation (15 configs) | `experiments/phase2/run_splice.py` | about 22 min | 0.10 GB (after collection; 1.18 GB during activation capture) | 3.05 GB |
| LoRA, 7 budgets | `experiments/phase2/run_lora.py` | about 22 min | 0.06 GB | 0.72 GB |
| LoRA + replay, 3 budgets | `experiments/phase2/run_lora.py --replay` | killed at start of training by memwatch (system swap 53%) | – | – |

Incidents:
- A first full-batch training attempt (batch 64 in one pass) hit the MPS cap at 4.0 GB. The
  allocator refused cleanly, and I switched to micro-batches of 16.
- An early splice evaluation hit the MPS cap during generation. Fixed with `inference_mode`,
  64-prompt generation chunks and `empty_cache` between rungs.
- CPU-only analysis scripts (activation capture plus model deep copies) peaked at 3.9-4.5 GB RSS.
  That is above the 2 GB guidance for CPU work; future CPU diagnostics should capture fewer sites
  or fewer sequences.

## 10. Reproduce

```bash
uv run pytest tests/toy                          # no-op capacity, tokenizer, RoPE filter, splice smoke
uv run python experiments/phase2/train_base.py   # runs/phase2/base.pt (+ .json)
uv run python experiments/phase2/run_splice.py   # runs/phase2/splice_results.json, spliced_*.pt
uv run python experiments/phase2/run_lora.py     # runs/phase2/lora_results.json
```

All three scripts need the memwatch watchdog running (`gpu_session` enforces this).
`runs/phase2/spliced_gated_add_nmax8_seed99.pt` and `spliced_ungated_add_nmax20_seed99.pt` load
with `load_model` and are the Phase 3 references (silent but imperfect; perfect but invasive).
