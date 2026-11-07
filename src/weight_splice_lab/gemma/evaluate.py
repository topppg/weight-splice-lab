import hashlib
import json
import math
import random
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch import Tensor

from weight_splice_lab.gemma.model import GemmaLM, count_params, load_model
from weight_splice_lab.gemma.task import (
    BASE_DIR,
    ROOT,
    addition_example,
    answer_tokens,
    grade_ids,
    reversed_example,
    tokenizer,
)
from weight_splice_lab.gemma.tokenizer import BOS, GemmaTokenizer
from weight_splice_lab.gpu import available_gb
from weight_splice_lab.harness.edit import apply_edit, load_edit
from weight_splice_lab.toy.data import WIKITEXT_DIR

# Pre-registered hidden-eval settings for phase 4 (Gemma 3 270M IT).
LENGTHS = list(range(1, 21))
LONG_LENGTHS = list(range(21, 31))
CHAT_LENGTHS = list(range(1, 9))
CHAT_N_PER = 50
ADDITION_SEED = 4321
LONG_SEED = 4325
CHAT_ADDITION_SEED = 4327
TEXT_SEED = 0
DIGIT_TEXT_SEED = 7
# reversed task (pre-registered 2026-11-07 after gemma1/gemma2): its own addition seeds; plain
# addition N1-8 on ADDITION_SEED is reported as a preservation check
REV_SEED = 4331
REV_LONG_SEED = 4335
PLAIN_CHECK_LENGTHS = list(range(1, 9))
TASKS = {"plain": addition_example, "reversed": reversed_example}
SEQ_LEN = 256
N_TEXT_CHUNKS = 256
N_DIGIT_CHUNKS = 128
DIGIT_FRAC = 0.03
KL_BAR = 0.05
ADDITION_BAR = 0.95
GEN_BATCH = 64
LOGIT_ROWS = 64
MIN_FREE_GB = 4.0

CHAT_TEMPLATE = "<bos><start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n"
CHAT_ADDITION = "What is {a} + {b}? Reply with only the number."
EVAL_SETS = ROOT / "runs" / "phase4" / "eval_sets"
CHAT_HIDDEN = EVAL_SETS / "chat_hidden.jsonl"
TOKENIZER_JSON = BASE_DIR / "tokenizer.json"
WIKITEXT_FILES = {"validation": "validation-00000-of-00001", "test": "test-00000-of-00001"}
PKG = Path(__file__).resolve().parents[1]
EVAL_SOURCES = (
    "gemma/evaluate.py",
    "gemma/model.py",
    "gemma/task.py",
    "gemma/tokenizer.py",
    "harness/edit.py",
)

# token ids and the first position whose next-token prediction is scored
Scored = tuple[list[int], int]


@dataclass(frozen=True)
class EvalData:
    tok: GemmaTokenizer
    text: list[Scored]
    digit_text: list[Scored]
    chat: list[Scored]


class Generator(Protocol):
    def generate(self, idx: Tensor, n_new: int) -> Tensor: ...


def chat_text(message: str) -> str:
    return CHAT_TEMPLATE.format(message)


def chat_ids(tok: GemmaTokenizer, message: str) -> list[int]:
    return tok.encode(chat_text(message), bos=False)


def wikitext_ids(tok: GemmaTokenizer, split: str) -> np.ndarray:
    lines = pq.read_table(WIKITEXT_DIR / f"{WIKITEXT_FILES[split]}.parquet").column("text")
    return np.array(tok.encode("".join(lines.to_pylist()), bos=False), dtype=np.int64)


def random_chunks(ids: np.ndarray, n: int, seed: int) -> list[Scored]:
    body = SEQ_LEN - 1
    starts = np.random.default_rng(seed).integers(0, len(ids) - body, size=n)
    return [([BOS, *ids[s : s + body].tolist()], 0) for s in starts]


def digit_chunks(ids: np.ndarray, digit_ids: frozenset[int], n: int, seed: int) -> list[Scored]:
    body = SEQ_LEN - 1
    count = np.concatenate([[0], np.cumsum(np.isin(ids, list(digit_ids)))])
    in_window = count[body:] - count[:-body]
    valid = np.flatnonzero(in_window >= DIGIT_FRAC * body)
    starts = np.random.default_rng(seed).choice(valid, size=n, replace=False)
    return [([BOS, *ids[s : s + body].tolist()], 0) for s in starts]


def load_chat(path: Path) -> list[Scored]:
    items = [json.loads(line) for line in path.read_text().splitlines()]
    return [(it["prompt_ids"] + it["response_ids"], len(it["prompt_ids"]) - 1) for it in items]


def load_data() -> EvalData:
    tok = tokenizer()
    return EvalData(
        tok=tok,
        text=random_chunks(wikitext_ids(tok, "validation"), N_TEXT_CHUNKS, TEXT_SEED),
        digit_text=digit_chunks(
            wikitext_ids(tok, "test"), tok.digit_ids, N_DIGIT_CHUNKS, DIGIT_TEXT_SEED
        ),
        chat=load_chat(CHAT_HIDDEN),
    )


@torch.inference_mode()
def teacher_forced(model: GemmaLM, base: GemmaLM, items: list[Scored]) -> dict[str, float]:
    # one sequence at a time and logits for LOGIT_ROWS positions at a time: a full 256-token
    # logit tensor over Gemma's 262k vocabulary is 268 MB per model
    kl = nll = nll_base = 0.0
    count = 0
    for ids, first in items:
        x = torch.tensor([ids[:-1]])
        target = torch.tensor(ids[first + 1 :])
        h, hb = model.hidden(x)[0, first:], base.hidden(x)[0, first:]
        for i in range(0, len(target), LOGIT_ROWS):
            rows, t = slice(i, i + LOGIT_ROWS), target[i : i + LOGIT_ROWS, None]
            logp = (h[rows] @ model.embed.weight.T).log_softmax(-1)
            logq = (hb[rows] @ base.embed.weight.T).log_softmax(-1)
            kl += F.kl_div(logp, logq, reduction="sum", log_target=True).item()
            nll -= logp.gather(1, t).sum().item()
            nll_base -= logq.gather(1, t).sum().item()
        count += len(target)
    return {"kl": kl / count, "nll": nll / count, "nll_base": nll_base / count, "positions": count}


@torch.inference_mode()
def generate_ids(
    model: Generator, prompts: list[list[int]], n_new: list[int], batch_size: int = GEN_BATCH
) -> list[list[int]]:
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, (p, n) in enumerate(zip(prompts, n_new, strict=True)):
        groups[(len(p), n)].append(i)
    out: list[list[int]] = [[] for _ in prompts]
    for (plen, n), idxs in groups.items():
        for j in range(0, len(idxs), batch_size):
            chunk = idxs[j : j + batch_size]
            gen = model.generate(torch.tensor([prompts[k] for k in chunk]), n)[:, plen:].tolist()
            for k, ids in zip(chunk, gen, strict=True):
                out[k] = ids
    return out


def addition_accuracy(
    model: Generator,
    tok: GemmaTokenizer,
    lengths: list[int],
    n_per: int,
    seed: int,
    chat: bool = False,
    task: str = "plain",
) -> dict[int, float]:
    rng = random.Random(seed)
    result: dict[int, float] = {}
    for n in lengths:
        examples = [TASKS[task](rng, n) for _ in range(n_per)]
        if chat:
            pairs = [ex.prompt.removesuffix("=").split("+") for ex in examples]
            prompts = [chat_ids(tok, CHAT_ADDITION.format(a=a, b=b)) for a, b in pairs]
        else:
            prompts = [tok.encode(ex.prompt) for ex in examples]
        n_new = [len(answer_tokens(tok, ex.answer)) + 1 for ex in examples]
        gens = generate_ids(model, prompts, n_new)
        hits = [grade_ids(tok, g, ex.answer) for g, ex in zip(gens, examples, strict=True)]
        result[n] = sum(hits) / len(hits)
    return result


def native_weight_change(model: GemmaLM, base: GemmaLM) -> dict[str, object]:
    # added heads/neurons are appended after the native rows/columns, so the native part of each
    # tensor is its leading block
    state = model.state_dict()
    per_tensor: dict[str, float] = {}
    delta_sq = base_sq = 0.0
    for name, w0 in base.state_dict().items():
        w = state[name][tuple(slice(0, s) for s in w0.shape)]
        d = (w - w0).pow(2).sum().item()
        b = w0.pow(2).sum().item()
        delta_sq += d
        base_sq += b
        if d > 0:
            per_tensor[name] = math.sqrt(d / b) if b > 0 else math.inf
    return {
        "total_rel": math.sqrt(delta_sq / base_sq),
        "max_rel": max(per_tensor.values(), default=0.0),
        "changed_tensors": len(per_tensor),
        "n_tensors": len(base.state_dict()),
        "per_tensor_rel": per_tensor,
    }


def added_param_report(model: GemmaLM, base: GemmaLM) -> dict[str, int]:
    # a unit is "active" when its output weights (o_proj columns / down_proj column) are non-zero
    units: list[tuple[Tensor, list[Tensor]]] = []
    for blk, ref in zip(model.blocks, base.blocks, strict=True):
        attn, mlp, hd = blk.attn, blk.mlp, blk.attn.head_dim
        for j, h in enumerate(range(ref.attn.n_heads, attn.n_heads)):
            q = slice(h * hd, (h + 1) * hd)
            kv = slice((attn.n_native_kv + j) * hd, (attn.n_native_kv + j + 1) * hd)
            ins = [attn.q_proj.weight[q], attn.k_proj.weight[kv], attn.v_proj.weight[kv]]
            units.append((attn.o_proj.weight[:, q], ins))
        for j in range(ref.mlp.gate_proj.out_features, mlp.gate_proj.out_features):
            ins = [mlp.gate_proj.weight[j], mlp.up_proj.weight[j]]
            units.append((mlp.down_proj.weight[:, j], ins))
    report = dict.fromkeys(("allocated", "nonzero", "inactive", "active", "nonzero_active"), 0)
    for out, ins in units:
        size = out.numel() + sum(t.numel() for t in ins)
        nz = int(out.count_nonzero()) + sum(int(t.count_nonzero()) for t in ins)
        live = bool(out.any())
        report["allocated"] += size
        report["nonzero"] += nz
        report["active" if live else "inactive"] += size
        report["nonzero_active"] += nz if live else 0
    return report


def by_n(acc: dict[int, float]) -> dict[str, object]:
    return {"by_n": {str(n): a for n, a in acc.items()}, "mean": sum(acc.values()) / len(acc)}


def timed[T](seconds: dict[str, float], name: str, fn: Callable[[], T]) -> T:
    t0 = time.time()
    out = fn()
    seconds[name] = round(time.time() - t0, 1)
    print(f"[eval] {name}: {seconds[name]} s", flush=True)
    return out


def evaluate_model(
    model: GemmaLM,
    base: GemmaLM,
    n_per: int,
    data: EvalData | None = None,
    task: str = "plain",
) -> dict[str, object]:
    t0 = time.time()
    model.eval()
    sec: dict[str, float] = {}
    data = data or timed(sec, "load_data", load_data)
    tok = data.tok
    seed, long_seed = (ADDITION_SEED, LONG_SEED) if task == "plain" else (REV_SEED, REV_LONG_SEED)
    acc = timed(
        sec, "addition", lambda: addition_accuracy(model, tok, LENGTHS, n_per, seed, task=task)
    )
    long = timed(
        sec,
        "length_n21_30",
        lambda: addition_accuracy(model, tok, LONG_LENGTHS, n_per, long_seed, task=task),
    )
    extra: dict[str, object] = {}
    if task != "plain":
        plain = timed(
            sec,
            "plain_addition",
            lambda: addition_accuracy(model, tok, PLAIN_CHECK_LENGTHS, n_per, ADDITION_SEED),
        )
        extra["plain_addition_n1_8"] = by_n(plain)
    chat_add = timed(
        sec,
        "chat_addition",
        lambda: addition_accuracy(model, tok, CHAT_LENGTHS, CHAT_N_PER, CHAT_ADDITION_SEED, True),
    )
    text = timed(sec, "text_kl", lambda: teacher_forced(model, base, data.text))
    digit = timed(sec, "digit_text_kl", lambda: teacher_forced(model, base, data.digit_text))
    chat = timed(sec, "chat_kl", lambda: teacher_forced(model, base, data.chat))
    mean_1_8 = sum(acc[n] for n in range(1, 9)) / 8
    meets_bar = text["kl"] < KL_BAR and chat["kl"] < KL_BAR
    primary = meets_bar and mean_1_8 >= ADDITION_BAR
    return {
        "task": task,
        "addition_by_n": {str(n): a for n, a in acc.items()},
        "addition_mean_n1_8": mean_1_8,
        "addition_mean_n9_20": sum(acc[n] for n in range(9, 21)) / 12,
        "length_n21_30": by_n(long),
        "text_kl": text["kl"],
        "text_nll": text["nll"],
        "text_nll_base": text["nll_base"],
        "digit_text_kl": digit["kl"],
        "digit_text_nll": digit["nll"],
        "digit_text_nll_base": digit["nll_base"],
        "chat_kl": chat["kl"],
        "chat_nll": chat["nll"],
        "chat_nll_base": chat["nll_base"],
        "chat_addition": by_n(chat_add),
        **extra,
        "positions": {"text": text["positions"], "digit_text": digit["positions"],
                      "chat": chat["positions"]},
        "native_weight_change": native_weight_change(model, base),
        "added_params": added_param_report(model, base),
        "thresholds": {"text_kl_lt": KL_BAR, "chat_kl_lt": KL_BAR, "digit_text_kl_lt": KL_BAR,
                       "addition_mean_n1_8_ge": ADDITION_BAR},
        "meets_bar": meets_bar,
        "meets_primary_bar": primary,
        "meets_strict_bar": primary and digit["kl"] < KL_BAR,
        "params": count_params(model),
        "n_per_length": n_per,
        "section_seconds": sec,
        "seconds": round(time.time() - t0, 1),
    }  # fmt: skip


def evaluate_edit(
    base_ckpt: Path,
    edit_path: Path | None,
    n_per: int = 200,
    threads: int = 4,
    task: str = "plain",
) -> dict[str, object]:
    if available_gb() < MIN_FREE_GB:
        raise RuntimeError(f"only {available_gb():.1f} GB available; need {MIN_FREE_GB} GB")
    torch.set_num_threads(threads)
    base = load_model(base_ckpt, device="cpu")
    model = base
    if edit_path is not None:
        model = apply_edit(base, load_edit(edit_path)).to("cpu").eval()
        if not torch.equal(model.embed.weight, base.embed.weight):
            raise ValueError("the edit changed the embedding, which is never editable")
        model.embed = base.embed
    result = evaluate_model(model, base, n_per, task=task)
    result["edit"] = str(edit_path) if edit_path is not None else None
    return result


def _file_sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def eval_code_hash() -> str:
    h = hashlib.sha256(b"".join(n.encode() + b"\0" + (PKG / n).read_bytes() for n in EVAL_SOURCES))
    return h.hexdigest()


def eval_key(base_ckpt: Path, n_per: int, task: str = "plain") -> dict[str, object]:
    seeds = {"addition": ADDITION_SEED, "long": LONG_SEED, "chat_addition": CHAT_ADDITION_SEED,
             "text": TEXT_SEED, "digit_text": DIGIT_TEXT_SEED}  # fmt: skip
    if task != "plain":
        seeds |= {"task_addition": REV_SEED, "task_long": REV_LONG_SEED}
    return {
        "checkpoint_sha256": _file_sha256(base_ckpt),
        "tokenizer_sha256": _file_sha256(TOKENIZER_JSON),
        "chat_hidden_sha256": _file_sha256(CHAT_HIDDEN),
        "eval_code_sha256": eval_code_hash(),
        "dataset": f"{WIKITEXT_DIR.parent.name}/{WIKITEXT_DIR.name}",
        "seeds": seeds,
        **({"task": task} if task != "plain" else {}),
        "thresholds": {"kl": KL_BAR, "addition": ADDITION_BAR, "digit_frac": DIGIT_FRAC},
        "n_per": n_per,
    }  # fmt: skip


def base_reference(
    base_ckpt: Path, cache_dir: Path, n_per: int = 200, task: str = "plain"
) -> dict[str, object]:
    # cached per (checkpoint, tokenizer, chat set, eval code, data, seeds, task, thresholds, n)
    key = eval_key(base_ckpt, n_per, task)
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    cache = cache_dir / f"base_eval_n{n_per}_{digest}.json"
    if cache.exists():
        cached = json.loads(cache.read_text())
        if cached.get("cache_key") == key:
            return cached
    result = evaluate_edit(base_ckpt, None, n_per, task=task) | {"cache_key": key}
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=1))
    tmp.replace(cache)
    return result
