# Phase 4 eval sets

Built 2026-11-07 by `experiments/phase4/build_eval_sets.py`.

## Source and licence

- Instructions: [`databricks/databricks-dolly-15k`](https://huggingface.co/datasets/databricks/databricks-dolly-15k), file `databricks-dolly-15k.jsonl`,
  revision `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`. Licence: CC BY-SA 3.0 (Databricks, Inc.). These files are a
  derivative (selected instructions plus generated responses) and carry the same licence.
- Responses: greedy outputs of the phase-4 base model `runs/phase4/base/base.pt`
  (`unsloth/gemma-3-270m-it` at revision `23cf460f6bb16954176b3ddcc8d4f250501458a9`, sha256 `3fb9e8716c97002e1587209d3eb47760d1e6701c1a69561511f5ff2fb65e8f25`), run with
  this repository's own Gemma implementation on CPU.

## Selection

- Rows with an empty `context` and a stripped instruction of 1 to 300
  characters; duplicate instruction texts kept once (first row). 10261 of
  15011 rows are eligible.
- Shuffled with `random.Random(0)`; the first 200 are `chat_practice.jsonl`
  (shown to the editing agent) and the next 200 are `chat_hidden.jsonl` (hidden eval).
- `id` is the 0-based line number in the source file.

## Format

Prompt: `<bos><start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n`,
checked string- and token-identical to `apply_chat_template(..., add_generation_prompt=True)` of
the HF tokenizer for all 400 items. Response: greedy, at most 128 new tokens,
cut after the first `<end_of_turn>` (kept). Batched by prompt length (16 per batch).

Fields: `id`, `instruction`, `prompt_ids`, `response_ids`, `prompt_text`, `response_text`.

| set | items | response tokens | ended with `<end_of_turn>` |
| --- | ---: | ---: | ---: |
| practice | 200 | 14594 | 117 |
| hidden | 200 | 14597 | 117 |

Generation took 355.0 s.
