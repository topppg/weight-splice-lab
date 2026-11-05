# Phase 1: reproducing the Qwen3.5-0.8B looping fix

Status (2026-11-05): logs inventoried; the harness spec is captured; all three published edits
rebuilt bit for bit; u re-derived from scratch, matching the published one. The grader port agrees
with the paper on 200/200 samples. **The local MLX eval has not run.** `gpu_session` refused it
because only ~5.5 GB of RAM was free and the run needs ≥ 6.3–7.4 GB (see "Blocked"). In its place,
the paper's own published generations were re-scored at reduced token caps, which gives the
reduced-scale reference numbers a local run should land near.

## Setup

- Logs: `external/lm_understand-logs` (shallow clone of RikiyaT/lm_understand-logs, 369 MB on disk, gitignored).
- Model: `models/qwen3.5-0.8b` = `Qwen/Qwen3.5-0.8B` at revision `2fc06364715b967f1860aea9cf38778875588b17`,
  the exact revision the edits were made on (the base-tensor sha256 checks pass). It is the full VLM release
  (`Qwen3_5ForConditionalGeneration`, one 1.75 GB safetensors file, `model.language_model.*` names plus
  `model.visual.*` and `mtp.*`). The paper used a text-only export with the same tensor names; its
  `chat_template.jinja` is identical to the release's.
- mlx-lm 0.32.2 supports the architecture (`mlx_lm/models/qwen3_5.py`, which drops visual/mtp weights).
  transformers 5.22 loads it on CPU, and its weights match the file.
- math-verify 0.9.0, the version the paper used.

## Code

| File | What |
|---|---|
| `src/weight_splice_lab/edits.py` | bf16 bit helpers, a minimal safetensors reader, `load_published_edit` / `apply_tensor_edit` / `rebuild_published_edit` (port of the logs' `apply_edit.py`, hash-checked), `ablate_direction` (W − α·u(uᵀW), optionally on chosen columns), `recover_rank_one`, `count_changed` |
| `src/weight_splice_lab/eval.py` | pure code: protocol constants, MATH-500 loader, a port of the paper's answer extraction and math-verify check (`code/mathqa.py`), loop signals (verbatim loop end, tail 8-gram duplication, whitespace tail), `Record` JSONL I/O, `summarize` |
| `src/weight_splice_lab/mlx_gen.py` | MLX generation: vLLM-order sampler (penalty → temperature → top-k → top-p), vLLM-style presence penalty (output tokens only, whole output), loads base + in-memory edited tensors (checked equal after load), batched generation with `BatchGenerator` |
| `experiments/phase1/reconstruct_edits.py` | rebuilds p10c/p25c/p50c, verifies hashes and counts, analyses each edit; `--recompute-u` re-derives u from Astra's texts (CPU, bf16, ~2.3 GB RSS) |
| `experiments/phase1/paper_outputs.py` | re-scores the paper's published first-sample texts at caps 4k/8k/16k/32k, checks the grader port against the paper's grades, totals the paper's eval compute |
| `experiments/phase1/run_eval.py` | the local eval (`--probe` ignores EOS to measure peak memory at the full length). Runs inside `gpu_session`, resumable JSONL output in `runs/phase1/` |
| `experiments/phase1/bench_throughput.py` | throughput and peak memory vs batch size, inside `gpu_session` (EOS ignored, so every sequence runs to `--max-tokens`); writes `runs/phase1/throughput_capped.json` (not included) |
| `tests/test_edits.py`, `tests/test_eval.py` | 23 tests (bf16 rounding vs torch, ablation algebra, sparse/low-rank rebuild, box extraction, math-verify cases, loop signals, summaries) |

`uv run --all-extras pytest tests/test_edits.py tests/test_eval.py` passes, and ruff is clean on all phase-1 files.
Use `--all-extras`. A plain `uv sync` / `uv run` without it drops mlx-lm and math-verify from the shared venv.

## What the logs contain (exp8_math)

- `runs/{p10c,p25c,p50c}/`: every turn (`turns/tNNN/{prompt.txt,answer.md,events.jsonl,run.json}`), the first
  message (`turns/t001/prompt.txt`) and its final form with the "EXPLAIN EVERY STEP" paragraph (`message.txt`),
  `tool_calls.jsonl` (every Python run: code and output cut to 6,000 chars; every forward request and its summary),
  `requests/fwd_k.{json.claimed,done.json}`, `explanations.md`, `final_answer.md`, `state.json`, `driver.log`,
  `work_notes/` (Astra's notebook, final report, frozen manifest with the 64 neuron ids per layer, splits).
  p50c also has `readable/`, a rendered export.
- **The edits are published** in `runs/<run>/edit/` as `manifest.json` plus `edit.npz`, exact bf16 diffs against the base:
  - `sparse` (p25c, p50c): flat indices (int32) and new bf16 bit patterns (uint16).
  - `lowrank` (p10c): float64 rank-1 factors `a` (rows×1) and `b` (1×cols), applied as bf16(f32(W + a·b)), plus "fixup" entries.
  - **u itself is not published** as a tensor. It is recoverable exactly up to sign from the p50c diff (rank 1 per tensor), and it can be recomputed from the logged texts (below).
- `eval/<unedited|p10c|p25c|p50c>/`: `merged.json` (the post's numbers), and per shard `scores.json`
  (acc, cut, `tokens_sum`, wall seconds `s`), `*_per_problem.json` (0/1 per sample for all problems), and
  `*_first_samples.jsonl.gz`. The full first-sample text is published for only the first 20 problems of each shard.
  For MATH-500 (10 shards, `rows[i::10]`) that covers indices 0–199. GPQA texts are withheld.
- `code/`: `evaluate.py` (vLLM eval), `mathqa.py` (grader), `profiles.py` (protocols), `build_ckpt.py`,
  `apply_edit.py`, `make_edit_diff.py`, `merge.py`, `practice35.py`, `st_numpy.py`.
- `practice/`: GSM8K train 7,469, MATH train 7,275, AIME 714, exactly as Astra saw them. `notices/`: 001 and 002
  reached these runs, 003 and 004 did not. `base_model/`: the text-only config, generation config and template.
- Not included: the forward-pass server and harness driver code (only `harness/sandbox.py` from the Go
  experiment, `codex_config.toml`, `instructions.md`), Astra's work folders, and IFEval/MC graders.

## How u was computed (p50c, from tool_calls.jsonl runs 2–9, 12, 16, 37, 61, 97)

1. Pass 1 generated 96 practice answers: 11/96 correct, 85 cut at 32,768 tokens, cut tails ~95% duplicate 8-grams.
2. Astra wrote **4 matched pairs of short thinking texts by hand**. Each pair shares a prefix and then splits into a
   productive continuation or a repetitive one ("But is 20 the total? No, it is the remaining time." ×3). The four
   problems: Rafael's weekly pay, a polynomial remainder, an arithmetic sequence, necklaces. They were sent as raw
   text `<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n<think>\n{text}`, with no instruction
   suffix. The first attempt (pass 2) used `chats`, which inserted an empty think block, and was redone as pass 3 (`fwd_4`).
3. A teacher-forced read returned the residual stream after every layer.
   **u_l = normalize( mean over the 4 pairs of [mean of h_{l+1} over the last 36 tokens of the repetitive text −
   the same for the productive text] )**, with h_{l+1} = `hidden_states[l+1]` (0 = embeddings). For down_proj of
   layer 15 that is hidden_16, for layer 19 hidden_20 (Astra's "residual outputs 16/20"). One u per layer; |cos(u15, u19)| = 0.85.
4. Sites: an attribution of the growth of the repetitive−productive contrast per block put layers 15 and 19
   (MLP part 0.26 and 0.25 of block growth 0.47 and 0.53) highest. Edits went from 1 neuron (L19 #11), to rank-1
   at L19, to rank-1 at L15+L19 with α = 1, 2, 3, 4.
5. Neuron subset: score_j = (u·W0[:, j]) × (mean over pairs of the repetitive − productive down_proj input j, last
   24 tokens of `fwd_4`), with u oriented toward "repetitive". Top 64 by score per layer.
6. Final: **W = W0 − 6·u ⊗ (mask64 ⊙ (uᵀW0))** on `layers.{15,19}.mlp.down_proj.weight`. Practice confirmation
   on 1,152 held-out problems: 10.6% → 60.4% (+49.8 pp, SE 1.5).

The other runs: **p10c** is a rank-1 W − 1.5·u(uᵀW) on 30 matrices (`linear_attn.out_proj`/`self_attn.o_proj` plus
`mlp.down_proj`, layers 8–22). Implied α = 1.5000 on all 30. The directions differ between matrices (min |cos| 0.41).
**p25c** zeroes columns 1792–2047 of `layers.15.self_attn.o_proj.weight` (= head 7, 262,144 entries).

## Edit reconstruction (verified)

`uv run --all-extras python experiments/phase1/reconstruct_edits.py --recompute-u` → `runs/phase1/edits_report.json`

| Run | Tensors | bf16 entries changed | Hashes |
|---|---|---|---|
| p10c | 30 | 72,986,982 | base + edited sha256 OK |
| p25c | 1 | 262,144 | OK |
| p50c | 2 | **130,188** (65,260 + 64,928) | OK |

p50c: exactly 64 changed columns per tensor, identical to the frozen manifest's neuron ids, so 2 × 64 × 1,024 =
**131,072 touched weights**. 884 of them round back to the same bf16 value. The SVD of the diff on those columns is
rank 1 (99.9995% of the energy), and the least-squares α = 6.0003 / 6.0002.

**From scratch:** the 8 logged texts were re-run through transformers (CPU, bf16). The token lengths match the
log exactly ([124, 123, 131, 136, 107, 146, 98, 135]). The recomputed u has |cos| = 0.99998 (L15) and 0.99997
(L19) with the published u. The top-64 neuron selection overlaps the published one in 64/64 (L15) and 63/64 (L19).
Rebuilding W from the recomputed u, α = 6 and the published neurons differs from the published tensor only by bf16
rounding: max |Δ| 4.9e-4 (L15) and 2.9e-3 (L19), with ~41k of 3.67M entries off by about one ulp.

## Harness spec (to rebuild in Phase 3)

- **Editor call:** Codex CLI, `codex exec` (first turn) / `codex exec resume <thread>` (later turns), prompt on stdin,
  `--json`, `-m gpt-6-astra -c model_reasoning_effort="max" -c model_context_window=872000
  -c model_auto_compact_token_limit=1000000`, plus the instructions file holding one line: "Answer the user's request."
  `harness/codex_config.toml` switches off every Codex built-in tool (shell, unified exec, web search, view_image,
  agents, plugins, browser, computer use, hooks …; `sandbox_mode = "read-only"`, `approval_policy = "never"`).
  Every turn's event stream was audited to contain no tool items (`run.json.audit.tool_free`).
- **Protocol per turn:** Astra replies with exactly one ```` ```python ```` or ```` ```forward ```` (JSON) block.
  The next message is the result: `OUTPUT of Python run k of at most 300:` plus the output in a code block, or
  `Forward pass k of 50 done (<type>); results in /work/...` with a JSON summary and `Forward passes left: N.`
  A reply without a block does not end the session before the budget is spent. After the last pass, Astra gives a
  final answer without code and the contents of `/work/edited` are submitted.
- **Kernel:** a persistent Python kernel (variables survive), CPU only, no internet. `/model` (ro), `/practice` (ro),
  `/work` (rw). numpy, scipy, sklearn, torch (CPU), pandas, sympy, networkx, safetensors, tokenizers. Predefined:
  `NAMES`, `load(name)` (float32 torch, current edit else original), `load_original`, `save_edit`, `drop_edit`,
  `edited()`, `tok`. Gradients disabled. 900 s and 96 GB per execution, output cut at 10,000 chars, 300 runs,
  and the editor is told when the kernel restarts. (The published `sandbox.py` is the Go-experiment variant: bubblewrap
  `--unshare-all`, clean env, ro venv, rw `/work`, rlimit AS, per-run timeout.)
- **save_edit refuses** `model.language_model.norm.weight`, `embed_tokens.weight`, `lm_head.weight`, and any edit that
  only multiplies a whole tensor by a constant. Same names and shapes are required; values must be finite.
- **Forward-pass tool**, with a budget of 10/25/50 passes. A failed request is free.
  - `generate`: original | edited | both, ≤ 512 generations (≤ 256 per model with "both"), optional custom prompts,
    `n` samples, `max_tokens` ≤ 32,768, eval sampling. The seed is fixed per problem (shared by both models). The
    summary gives accuracy overall and by source, mean tokens, cut share, failure counts (wrong_answer,
    no_answer_finished, cut_at_max_tokens), and for "both" the paired difference with SE and rescues/losses. Texts
    go to `/work/gen_k*.jsonl`. Optional activations: the residual at chosen layers (and down_proj inputs) at the
    last prompt token plus the mean over the first 8,192 answer tokens.
  - `teacher_forced`: ≤ 1,024 texts, ≤ 32,768 tokens each, ≤ 524,288 total. Raw `texts` or `chats`
    (+`enable_thinking`). Returns token logprobs, top-k (≤ 50), hidden states, mlp_act, and attention probabilities
    (full-attention layers only, ≤ 2 texts of ≤ 2,048 tokens), as right-padded `.npy` in `/work/fwd_k/` (≤ 6 GB).
- **Rules in the prompt:** no gradient descent or iterative fitting; no running or simulating the network in the
  kernel; no internet. In scope: edits justified by an understood failure mechanism, including repetition. Out of
  scope: global sharpening/flattening, and skipping or closing the thinking early regardless of the problem.
- **Driver behaviours:** a fresh conversation every 40 turns or after a restart (the "handoff": full task text +
  "THIS IS A RESTARTED CONVERSATION" + counters + current edited tensors + `/work` listing + the last 8,000 chars of
  `/work/notes.md`). From turn 29 in p50c it required "EXPLAIN EVERY STEP" (3–8 sentences before each block), and a
  block with no explanation was sent back unrun, at most twice in a row. When the provider's content filter refused
  a message, Astra got a canned note (don't print raw vocabulary tokens or long "Wait, …" transcripts). Notices 001
  (a pass costs 1 at any size) and 002 (tool corrections, `/work/eval_prompt_format.txt`, note the edit before
  each pass) were appended to turn messages.
- **Final eval** (`code/evaluate.py`, protocol `think35`): the submission is cast to bf16. vLLM `LLM(dtype=bf16,
  max_model_len=32768+2048, seed=0)`, chat template with `enable_thinking=True`, user = problem + "\n" + "Please
  reason step by step, and put your final answer within \boxed{}.". SamplingParams(n, temperature 0.6, top_p 0.95,
  top_k 20, presence_penalty 1.5, max_tokens 32768, seed 0). The grade is the last `\boxed{}`/`\fbox{}` after the
  last `</think>` (brace matched; `\boxed 5` form allowed), checked with math-verify
  `verify(parse("\boxed{gold}"), parse("\boxed{pred}"))`. The score is the mean over all samples, and
  `cut_share` is finish_reason == "length".

## Local reproduction: what is done and what is not

### Grader port

Re-grading the paper's published first samples with `eval.py` gives the paper's own 0/1 on **200/200** samples
for each of the 4 models (`grader_agreement_at_32768`).

### Paper outputs re-scored at reduced caps (reference for a local run)

`experiments/phase1/paper_outputs.py` re-tokenizes each published first sample (vLLM, 32k cap) and truncates it
at the cap. A sample counts as cut if it is ≥ cap tokens, and the truncated text is graded. Columns: accuracy /
cut share / share of cut answers ending in a verbatim loop / mean tail 8-gram duplicate share of cut answers.
MATH-500 indices 0–199, 1 sample (`runs/phase1/paper_outputs_200.json`; first 50 in `paper_outputs.json`).

| Cap | Base | p10c | p25c | p50c |
|---|---|---|---|---|
| 4,096 | 3.5 / 98 / 14 / .18 | 2.5 / 98 / 2 / .10 | 3.5 / 98.5 / 3 / .08 | 8.0 / 92 / 0 / .02 |
| 8,192 | 6.5 / 94.5 / 63 / .68 | 5.5 / 95.5 / 27 / .42 | 7.0 / 93.5 / 32 / .39 | 14.5 / 85.5 / 0 / .07 |
| 16,384 | 7.5 / 93.5 / 92 / .92 | 13 / 87 / 67 / .80 | 12 / 88.5 / 81 / .82 | 27.5 / 71 / 1 / .19 |
| 32,768 | 7.5 / 85.5 / 96 / .96 | 16.5 / 76 / 89 / .97 | 13 / 80.5 / 94 / .94 | 46 / 31 / 11 / .41 |

For reference, the paper's full 4-sample MATH-500 numbers: base 7.9% acc / 93.0% cut, p50c 48.1% / 33.1%.
p50c's 32k cut answers include whitespace degeneration: 16% of them end in ≥ 500 chars of pure whitespace (one
sample is 3.46M chars of spaces). Astra flagged this as "whitespace_tail_cut".

What this implies for the planned 50-problem, 4,096-token run:
- **Accuracy will barely separate the models at 4k.** Expect roughly 0–4% for base and 6–8% for p50c, both
  ≥ 92% cut. The edit's gain needs long thinking: p50c's median correct answer is ~9k tokens on practice problems.
- **The loop signature separates the models at every cap.** At 4k, base cut answers already show 14% verbatim
  loops and 0.18 tail duplication against 0% and 0.02 for p50c. At 8k the gap is 63% vs 0% and 0.68 vs 0.07.
  Base loops start late, mostly after 4k tokens, so **8,192 tokens is the more informative local cap** if memory
  allows.
- Two base outputs inspected (problems `intermediate_algebra/607`, `number_theory/516`): genuine verbatim loops.
  One repeats "Wait, $35^{1.5} = 35 \sqrt{35}$." to the cap; the other circles on "Wait, let me check the string
  again. "10101001110"" from ~20% of the text onward. The p50c answer to 607 finishes with `\boxed{13535}`
  (correct). Its answer to 516 degenerates into whitespace.

### Local MLX eval: blocked

- Throughput (measured before the GPU rules, no memory caps, `runs/phase1/throughput.json`, 1,024 new tokens,
  presence-penalty processor on): batch 1 → 110 tok/s (1.65 GB peak), 16 → 553 tok/s (3.14 GB), 32 → 829 tok/s
  (3.99 GB), 64 → 824 tok/s (6.54 GB).
- Peak memory needed. Estimated from those points: ~1.6 GB of model, ~0.10 GB per sequence of logits/sampling/state,
  and 12.3 KB/token of KV (6 full-attention layers × K,V × 2 heads × 256 × bf16). At **4,096 tokens**:

  | Batch | Est. peak | `min_gb` (peak / 0.8) | RAM available needed |
  |---|---|---|---|
  | 1 | ~1.7 GB | 2.2 | 6.2 GB |
  | 2 | ~1.9 GB | 2.4 | 6.4 GB |
  | 4 | ~2.15 GB | 2.7 | 6.7 GB |
  | 8 | ~2.7 GB | 3.4 | 7.4 GB |

  At 8,192 tokens, add ~50 MB per sequence (batch 8 → ~3.1 GB peak, `min_gb` 3.9, 7.9 GB available).
- I tried the batch-2 peak probe (`run_eval.py --variant base --n 2 --max-tokens 4096 --batch-size 2 --min-gb 2.3
  --probe`). `gpu_session` refused it: "only 5.8 GB available; need 6.3 GB". Available memory stayed at 5.5–5.8 GB
  for the rest of the session. I did not work around the refusal.
- **To run** once ≥ ~7.5 GB is free, with no other GPU job:
  1. Probe: `uv run --all-extras python experiments/phase1/run_eval.py --variant base --n 2 --max-tokens 4096 --batch-size 2 --min-gb 2.4 --probe` and read `peak_gb`.
  2. Choose the batch so that the extrapolated peak is ≤ 0.8 × budget.
  3. Run `run_eval.py --variant base --n 50 --max-tokens 4096 --batch-size 8 --min-gb 3.4`, then the same with `--variant p50c`.

  Expected throughput is ~350–400 tok/s at batch 8, so roughly 10 min per model at 4k (≤ 200k tokens). The output
  `runs/phase1/eval_<variant>_n50_t4096_b8/{records.jsonl,summary.json}` includes `peak_gb`, `model_gb` and
  `budget_gb`, so the per-run peak memory goes into this doc once it has run. Compare it with the 4k row above
  (indices 0–49 are in the published set).
- **Safety fix in `mlx_gen.generate_batch`:** mlx-lm's `BatchGenerator.__init__` calls
  `maybe_set_recommended_wired_limit()`, which raises the Metal wired limit to the device maximum
  (`max_recommended_working_set_size`). That silently overrides `gpu_session`'s cap. `stream_generate` does the same
  through `wired_limit()`. This probably contributed to the earlier freeze. `generate_batch` now replaces that hook
  with a no-op, so the session's limit stands. Nothing calls `set_wired_limit` itself. Do not use `mlx_lm.generate` /
  `stream_generate` / `batch_generate` directly in this repo without the same guard.

## Deviations from the paper's protocol (in the local pipeline)

- MLX instead of vLLM: different kernels and RNG, so runs are not seed-reproducible against the paper. The sampler
  and penalty follow vLLM's order and semantics (top-p applied after temperature and top-k; the presence penalty
  covers the whole output and never the prompt). mlx-lm's built-in versions differ: they apply top-p before
  temperature, and the presence penalty covers prompt + the last 20 tokens. Both stop tokens are honoured:
  248044 `<|endoftext|>` (the export's generation_config eos) and 248046 `<|im_end|>` (the tokenizer eos).
- The full VLM checkpoint is used instead of the text-only export (same tensors; mlx-lm drops visual/mtp).
- Reduced scale: first 50 MATH-500 problems, 1 sample, 4,096 tokens, against 500 × 4 × 32,768 in the paper. The cap
  changes what "cut" means, so compare with the re-scored table above, not with 48.1% / 33.1%.
- The token count for a stop is len(output) + 1 (counting EOS, as vLLM does). The paper-output re-scoring
  re-tokenizes decoded text, so lengths can be off by a few tokens.

## H100 GPU-hour estimate (full reported eval, vLLM)

The paper's own eval (from `scores.json`): MATH-500 × 4 samples = 61.7M generated tokens for base, 60.2M for p10c,
59.3M for p25c and 44.1M for p50c. Base is ≈ 2,000 × 32k because almost every sample hits the cap. All 10
benchmarks for the 4 models total 1.30B tokens and 103 job-hours on their (unnamed, shared) GPUs. Per-job
throughput ranged from 3.0k to 12.4k tok/s.

Throughput assumption: decoding a 0.8B hybrid model at long context is bound by KV-cache reads. At ~16k average
context that is 12.3 KB/token × 16k ≈ 200 MB per generated token, so ≲ 17k tok/s at 3.35 TB/s. I assume
**~10k tok/s per H100** (range 6k–15k), with enough concurrency (~150 × 32k sequences fit easily in 80 GB).

| Scope | Tokens | H100-hours (10k tok/s; range) |
|---|---|---|
| MATH-500 × 4, base + p50c | 106M | **~3 h** (2–5) |
| MATH-500 × 4, all 4 models | 225M | ~6.5 h (4–10.5) |
| Every reported benchmark, 4 models (MATH-500, GSM8K, AMC23, Minerva, OlympiadBench, AIME, MMLU-Pro 1k, GPQA, IFEval) | 1.30B | ~36 h (24–60) |

Add ~10–15% for model load, CUDA-graph capture and stragglers. At ~$2–3 per H100-hour the main table
(MATH-500, base + p50c) costs under $15.

## Open issues

1. The local eval is blocked on free RAM (needs ≥ 6.4 GB available for the batch-2 probe, ~7.4 GB for batch 8 at
   4k). The per-run peak memory for this doc comes only from that run.
2. A 4k cap will barely show the accuracy gain. Prefer 8k (≈ 7.9 GB available needed at batch 8), or treat the
   loop metrics as the primary local check.
3. The MLX sampler has no CPU unit test, by design (tests must not touch the GPU). It should be sanity-checked
   in the probe run: the edit should reduce tail duplication against base.
4. The throughput numbers above predate the GPU rules (uncapped, wired limit raised by mlx-lm). Re-measure them with `bench_throughput.py --min-gb ...` once memory allows.
5. The GPU type behind the paper's eval is not stated in the logs, so the H100 rate is an estimate. It should be
   measured on the first rented run.
6. p10c's 30 directions are not one shared u (min |cos| 0.41). How Astra computed them per matrix was not traced
   here, since Phase 1 targets p50c.
