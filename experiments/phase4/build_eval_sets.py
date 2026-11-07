import datetime
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from weight_splice_lab.gemma.evaluate import EVAL_SETS, MIN_FREE_GB, chat_ids, chat_text
from weight_splice_lab.gemma.model import load_model
from weight_splice_lab.gemma.task import BASE_CKPT, HF_MODEL, tokenizer
from weight_splice_lab.gpu import available_gb

DOLLY = "databricks/databricks-dolly-15k"
DOLLY_FILE = "databricks-dolly-15k.jsonl"
DOLLY_REVISION = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
HF_MODEL_REVISION = "23cf460f6bb16954176b3ddcc8d4f250501458a9"
SHUFFLE_SEED = 0
MAX_INSTRUCTION_CHARS = 300
N_PER_SET = 200
MAX_NEW = 128
GEN_BATCH = 16
END_OF_TURN = 106


def instructions() -> tuple[list[tuple[int, str]], dict[str, object]]:
    path = hf_hub_download(DOLLY, DOLLY_FILE, repo_type="dataset", revision=DOLLY_REVISION)
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    pool: list[tuple[int, str]] = []
    seen: set[str] = set()
    for i, row in enumerate(rows):
        # the Gemma template trims message content, so the stored instruction is trimmed too
        text = row["instruction"].strip()
        if row["context"] == "" and 0 < len(text) <= MAX_INSTRUCTION_CHARS and text not in seen:
            seen.add(text)
            pool.append((i, text))
    random.Random(SHUFFLE_SEED).shuffle(pool)
    return pool, {"rows": len(rows), "eligible_unique": len(pool)}


def check_template(texts: list[str]) -> None:
    hf = AutoTokenizer.from_pretrained(HF_MODEL, revision=HF_MODEL_REVISION)
    tok = tokenizer()
    for text in texts:
        messages = [{"role": "user", "content": text}]
        ref = hf.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if ref != chat_text(text):
            raise SystemExit(f"chat template mismatch for {text!r}:\n{ref!r}")
        if hf(ref, add_special_tokens=False)["input_ids"] != chat_ids(tok, text):
            raise SystemExit(f"token id mismatch for {text!r}")


def cut(ids: list[int]) -> list[int]:
    return ids[: ids.index(END_OF_TURN) + 1] if END_OF_TURN in ids else ids


@torch.inference_mode()
def responses(prompts: list[list[int]]) -> list[list[int]]:
    model = load_model(BASE_CKPT, device="cpu")
    groups: dict[int, list[int]] = defaultdict(list)
    for i, p in enumerate(prompts):
        groups[len(p)].append(i)
    out: list[list[int]] = [[] for _ in prompts]
    for plen, idxs in sorted(groups.items()):
        for j in range(0, len(idxs), GEN_BATCH):
            chunk = idxs[j : j + GEN_BATCH]
            gen = model.generate(torch.tensor([prompts[k] for k in chunk]), MAX_NEW)
            for k, ids in zip(chunk, gen[:, plen:].tolist(), strict=True):
                out[k] = cut(ids)
    return out


def write_set(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def readme(stats: dict[str, object]) -> str:
    return f"""# Phase 4 eval sets

Built {datetime.date.today().isoformat()} by `experiments/phase4/build_eval_sets.py`.

## Source and licence

- Instructions: [`{DOLLY}`](https://huggingface.co/datasets/{DOLLY}), file `{DOLLY_FILE}`,
  revision `{DOLLY_REVISION}`. Licence: CC BY-SA 3.0 (Databricks, Inc.). These files are a
  derivative (selected instructions plus generated responses) and carry the same licence.
- Responses: greedy outputs of the phase-4 base model `runs/phase4/base/base.pt`
  (`{HF_MODEL}` at revision `{HF_MODEL_REVISION}`, sha256 `{stats["base_sha256"]}`), run with
  this repository's own Gemma implementation on CPU.

## Selection

- Rows with an empty `context` and a stripped instruction of 1 to {MAX_INSTRUCTION_CHARS}
  characters; duplicate instruction texts kept once (first row). {stats["eligible_unique"]} of
  {stats["rows"]} rows are eligible.
- Shuffled with `random.Random({SHUFFLE_SEED})`; the first {N_PER_SET} are `chat_practice.jsonl`
  (shown to the editing agent) and the next {N_PER_SET} are `chat_hidden.jsonl` (hidden eval).
- `id` is the 0-based line number in the source file.

## Format

Prompt: `<bos><start_of_turn>user\\n{{instruction}}<end_of_turn>\\n<start_of_turn>model\\n`,
checked string- and token-identical to `apply_chat_template(..., add_generation_prompt=True)` of
the HF tokenizer for all {2 * N_PER_SET} items. Response: greedy, at most {MAX_NEW} new tokens,
cut after the first `<end_of_turn>` (kept). Batched by prompt length ({GEN_BATCH} per batch).

Fields: `id`, `instruction`, `prompt_ids`, `response_ids`, `prompt_text`, `response_text`.

| set | items | response tokens | ended with `<end_of_turn>` |
| --- | ---: | ---: | ---: |
| practice | {N_PER_SET} | {stats["practice_tokens"]} | {stats["practice_ended"]} |
| hidden | {N_PER_SET} | {stats["hidden_tokens"]} | {stats["hidden_ended"]} |

Generation took {stats["gen_seconds"]} s.
"""


def main() -> None:
    if available_gb() < MIN_FREE_GB:
        raise SystemExit(f"only {available_gb():.1f} GB available; need {MIN_FREE_GB} GB")
    torch.set_num_threads(4)
    pool, stats = instructions()
    chosen = pool[: 2 * N_PER_SET]
    check_template([text for _, text in chosen])
    tok = tokenizer()
    prompts = [chat_ids(tok, text) for _, text in chosen]
    t0 = time.time()
    gens = responses(prompts)
    rows: list[dict[str, object]] = [
        {"id": i, "instruction": text, "prompt_ids": p, "response_ids": r,
         "prompt_text": chat_text(text), "response_text": tok.decode(r)}
        for (i, text), p, r in zip(chosen, prompts, gens, strict=True)
    ]  # fmt: skip
    EVAL_SETS.mkdir(parents=True, exist_ok=True)
    for name, part in (("practice", slice(0, N_PER_SET)), ("hidden", slice(N_PER_SET, None))):
        write_set(EVAL_SETS / f"chat_{name}.jsonl", rows[part])
        stats[f"{name}_tokens"] = sum(len(r) for r in gens[part])
        stats[f"{name}_ended"] = sum(r[-1] == END_OF_TURN for r in gens[part])
    with BASE_CKPT.open("rb") as f:
        stats["base_sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
    stats["gen_seconds"] = round(time.time() - t0, 1)
    (EVAL_SETS / "README.md").write_text(readme(stats))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
