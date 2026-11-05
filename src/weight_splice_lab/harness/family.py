import importlib
import inspect
import json
import random
import shutil
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Protocol

import pyarrow.parquet as pq
from torch import nn

from weight_splice_lab.gemma import model as gemma_model
from weight_splice_lab.gemma import task as gemma_task
from weight_splice_lab.gemma import tokenizer as gemma_tokenizer
from weight_splice_lab.toy import data as toy_data
from weight_splice_lab.toy import metrics as toy_metrics
from weight_splice_lab.toy import model as toy_model

ROOT = Path(__file__).resolve().parents[3]
PRACTICE_SEED = 20261105
ADDITION_LENGTHS = range(1, 21)
ADDITION_PER_N = 50
NATIVE_PER_LEN = 20
TEXT_CHARS = 200_000

Rows = list[dict[str, object]]


class Tokenizer(Protocol):
    # the interface the forward tool needs from either tokenizer
    def encode(self, text: str) -> list[int]: ...
    def show(self, ids: list[int]) -> str: ...


@dataclass(frozen=True)
class Family:
    name: str
    model_module: ModuleType
    lm_class: str
    kernel_alias: str
    base_ckpt: Path
    prompt_file: Path
    runs_dir: Path
    eval_module: str
    forward_items: int
    max_added_params: int
    kernel_max_rss_gb: float
    write_model_extras: Callable[[Path], None]
    practice: Callable[[], dict[str, Rows]]
    practice_text: Callable[[], str]
    tokenizer: Callable[[], Tokenizer]
    grade: Callable[[Tokenizer, list[int], str, str], bool]

    def load_model(self, path: Path) -> nn.Module:
        return self.model_module.load_model(path).eval()

    def evaluator(self) -> ModuleType:
        # imported lazily: the eval modules pull in their datasets
        return importlib.import_module(self.eval_module)

    @property
    def base_cache_dir(self) -> Path:
        return self.runs_dir / ("base_eval_cache" if self.name == "toy" else "eval_cache")

    def sources(self) -> list[Path]:
        pkg = ROOT / "src" / "weight_splice_lab"
        return [*(pkg / self.model_module.__name__.split(".")[1]).glob("*.py"), self.prompt_file]


# ---- toy (phase 3): char-level 5M model, reversed-digit addition ----


class _ToyTokenizer:
    def encode(self, text: str) -> list[int]:
        return toy_data.TOKENIZER.encode(text)

    def show(self, ids: list[int]) -> str:
        # one character per token; special tokens stay visible so positions line up
        return "".join(toy_data.TOKENIZER.itos[i] for i in ids)


def _toy_extras(model_dir: Path) -> None:
    head = (
        "import numpy as np\n\n"
        f"PAD, BOS, UNK = {toy_data.PAD}, {toy_data.BOS}, {toy_data.UNK}\n"
        f"CHARS = {toy_data.CHARS!r}\n\n\n"
    )
    (model_dir / "tokenizer.py").write_text(head + inspect.getsource(toy_data.CharTokenizer))


def _toy_practice() -> dict[str, Rows]:
    rng = random.Random(PRACTICE_SEED)
    addition = []
    for n in ADDITION_LENGTHS:
        for i in range(ADDITION_PER_N):
            ex = toy_data.addition_example(rng, n)
            addition.append({"id": f"add_n{n:02d}_{i:03d}", "task": "addition", "n_digits": n,
                             "prompt": ex.prompt, "answer": ex.answer})  # fmt: skip
    native = []
    for name, task in toy_data.NATIVE_TASKS.items():
        for n in range(1, toy_data.NATIVE_MAX_LEN + 1):
            for i in range(NATIVE_PER_LEN):
                ex = task(rng, n)
                native.append({"id": f"{name}_n{n:02d}_{i:03d}", "task": name, "n_digits": n,
                               "prompt": ex.prompt, "answer": ex.answer})  # fmt: skip
    return {"addition.jsonl": addition, "native.jsonl": native}


def _toy_text() -> str:
    ids = toy_data.load_wikitext("train", max_chars=2_000_000)
    return toy_data.TOKENIZER.decode(ids[-TEXT_CHARS:].tolist())


def _toy_grade(tok: Tokenizer, ids: list[int], task: str, answer: str) -> bool:
    # addition = the N+1 answer tokens, native tasks = answer + newline
    return toy_metrics.grade_ids(ids, answer, task != "addition")


TOY = Family(
    name="toy",
    model_module=toy_model,
    lm_class="ToyLM",
    kernel_alias="toy_model",
    base_ckpt=ROOT / "runs" / "phase2" / "base.pt",
    prompt_file=ROOT / "prompts" / "toy_addition.md",
    runs_dir=ROOT / "runs" / "phase3",
    eval_module="weight_splice_lab.harness.evaluate",
    forward_items=2048,
    max_added_params=1_500_000,
    kernel_max_rss_gb=2.0,
    write_model_extras=_toy_extras,
    practice=_toy_practice,
    practice_text=_toy_text,
    tokenizer=_ToyTokenizer,
    grade=_toy_grade,
)


# ---- gemma (phase 4): Gemma 3 270M IT, plain-notation addition ----

CHAT_PRACTICE = ROOT / "runs" / "phase4" / "eval_sets" / "chat_practice.jsonl"


class _GemmaTokenizer:
    def __init__(self) -> None:
        self.tok = gemma_task.tokenizer()

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text)

    def show(self, ids: list[int]) -> str:
        return self.tok.decode(ids)


def _gemma_extras(model_dir: Path) -> None:
    shutil.copy(inspect.getsourcefile(gemma_tokenizer) or "", model_dir / "tokenizer.py")
    shutil.copy(gemma_task.BASE_DIR / "tokenizer.json", model_dir / "tokenizer.json")
    (model_dir / "tokenizer.json").chmod(0o644)


def _addition_rows(
    example: Callable[[random.Random, int], gemma_task.Example],
    lengths: range,
    per_n: int,
    task: str,
    prefix: str,
) -> Rows:
    rng = random.Random(PRACTICE_SEED)
    rows: Rows = []
    for n in lengths:
        for i in range(per_n):
            ex = example(rng, n)
            rows.append({"id": f"{prefix}_n{n:02d}_{i:03d}", "task": task, "n_digits": n,
                         "prompt": ex.prompt, "answer": ex.answer})  # fmt: skip
    return rows


def _chat_practice() -> Rows:
    keep = ("id", "instruction", "prompt_text", "response_text", "prompt_ids", "response_ids")
    chat = []
    for line in CHAT_PRACTICE.read_text().splitlines():
        row = {k: v for k, v in json.loads(line).items() if k in keep}
        # the forward tool adds <bos> itself
        row["prompt_text"] = str(row["prompt_text"]).removeprefix("<bos>")
        chat.append(row)
    return chat


def _gemma_practice() -> dict[str, Rows]:
    rows = _addition_rows(
        gemma_task.addition_example, ADDITION_LENGTHS, ADDITION_PER_N, "addition", "add"
    )
    return {"addition.jsonl": rows, "chat.jsonl": _chat_practice()}


def _gemma_rev_practice() -> dict[str, Rows]:
    rev = _addition_rows(
        gemma_task.reversed_example, ADDITION_LENGTHS, ADDITION_PER_N, "addition", "add"
    )
    plain = _addition_rows(gemma_task.addition_example, range(1, 9), 20, "plain", "plain")
    return {"addition.jsonl": rev, "plain.jsonl": plain, "chat.jsonl": _chat_practice()}


def _gemma_text() -> str:
    table = pq.read_table(toy_data.WIKITEXT_DIR / "train-00000-of-00002.parquet")
    pieces: list[str] = []
    total = 0
    for line in table.column("text").to_pylist():
        pieces.append(line)
        total += len(line)
        if total >= 2_000_000:
            break
    return "".join(pieces)[-TEXT_CHARS:]


def _gemma_grade(tok: Tokenizer, ids: list[int], task: str, answer: str) -> bool:
    assert isinstance(tok, _GemmaTokenizer) and task in ("addition", "plain")
    return gemma_task.grade_ids(tok.tok, ids, answer)


GEMMA = Family(
    name="gemma",
    model_module=gemma_model,
    lm_class="GemmaLM",
    kernel_alias="gemma_model",
    base_ckpt=gemma_task.BASE_CKPT,
    prompt_file=ROOT / "prompts" / "gemma_addition.md",
    runs_dir=ROOT / "runs" / "phase4",
    eval_module="weight_splice_lab.gemma.evaluate",
    forward_items=1024,
    max_added_params=10_000_000,
    kernel_max_rss_gb=3.0,
    write_model_extras=_gemma_extras,
    practice=_gemma_practice,
    practice_text=_gemma_text,
    tokenizer=_GemmaTokenizer,
    grade=_gemma_grade,
)

# ---- gemma_rev (phase 4b): the same model, reversed-digit addition (the toy's format) ----

GEMMA_REV = replace(
    GEMMA,
    name="gemma_rev",
    prompt_file=ROOT / "prompts" / "gemma_reversed.md",
    eval_module="weight_splice_lab.gemma.evaluate_reversed",
    practice=_gemma_rev_practice,
)

FAMILIES = {f.name: f for f in (TOY, GEMMA, GEMMA_REV)}
