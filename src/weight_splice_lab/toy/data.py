import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import Tensor

WIKITEXT_DIR = (
    Path.home()
    / ".cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots"
    / "b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-103-raw-v1"
)

PAD, BOS, UNK = 0, 1, 2
CHARS = "\n" + "".join(chr(c) for c in range(32, 127))


class CharTokenizer:
    def __init__(self) -> None:
        self.itos = ["<pad>", "<bos>", "<unk>", *CHARS]
        self.stoi = {c: i for i, c in enumerate(self.itos) if i > UNK}
        self.lut = np.full(0x110000, UNK, dtype=np.uint8)
        for c, i in self.stoi.items():
            self.lut[ord(c)] = i

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str, bos: bool = True) -> list[int]:
        ids = [self.stoi.get(c, UNK) for c in text]
        return [BOS, *ids] if bos else ids

    def encode_array(self, text: str) -> np.ndarray:
        codes = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
        return self.lut[codes]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] if i > UNK else "" for i in ids)

    def id(self, char: str) -> int:
        return self.stoi[char]

    def digit_ids(self) -> list[int]:
        return [self.stoi[str(d)] for d in range(10)]


TOKENIZER = CharTokenizer()


def load_wikitext(split: str, max_chars: int | None = None) -> np.ndarray:
    name = {
        "train": "train-00000-of-00002",
        "validation": "validation-00000-of-00001",
        "test": "test-00000-of-00001",
    }[split]
    table = pq.read_table(WIKITEXT_DIR / f"{name}.parquet")
    pieces: list[str] = []
    total = 0
    for line in table.column("text").to_pylist():
        pieces.append(line)
        total += len(line)
        if max_chars is not None and total >= max_chars:
            break
    return TOKENIZER.encode_array("".join(pieces))


def rand_digits(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("0123456789") for _ in range(n))


def rand_number(rng: random.Random, n: int) -> str:
    return rng.choice("123456789") + rand_digits(rng, n - 1) if n > 1 else rand_digits(rng, 1)


@dataclass(frozen=True)
class Example:
    prompt: str
    answer: str


def copy_task(rng: random.Random, n: int) -> Example:
    x = rand_digits(rng, n)
    return Example(f"C:{x}=", x)


def reverse_task(rng: random.Random, n: int) -> Example:
    x = rand_digits(rng, n)
    return Example(f"R:{x}=", x[::-1])


def max_task(rng: random.Random, n: int) -> Example:
    x, y = rand_number(rng, rng.randint(1, n)), rand_number(rng, n)
    if rng.random() < 0.5:
        x, y = y, x
    return Example(f"M:{x},{y}=", max(x, y, key=int))


NATIVE_TASKS: dict[str, Callable[[random.Random, int], Example]] = {
    "copy": copy_task,
    "reverse": reverse_task,
    "max": max_task,
}
NATIVE_MAX_LEN = 12


# Target format: little-endian operands and little-endian sum, fixed width N+1 output.
def addition_example(rng: random.Random, n: int) -> Example:
    a, b = rand_digits(rng, n), rand_digits(rng, n)
    s = str(int(a[::-1]) + int(b[::-1])).zfill(n + 1)[::-1]
    return Example(f"A:{a}+{b}=", s)


def ladder_example(rung: str, rng: random.Random, n: int) -> Example:
    a, b = rand_digits(rng, n), rand_digits(rng, n)
    prompt = f"A:{a}+{b}="
    match rung:
        case "copy":
            return Example(prompt, a)
        case "successor":
            return Example(prompt, "".join(str((int(c) + 1) % 10) for c in a))
        case "add":
            return Example(prompt, str(int(a[::-1]) + int(b[::-1])).zfill(n + 1)[::-1])
    raise ValueError(rung)


def native_pack(rng: random.Random, n_chars: int) -> str:
    lines: list[str] = []
    total = 0
    while total < n_chars:
        task = rng.choice(list(NATIVE_TASKS.values()))
        ex = task(rng, rng.randint(1, NATIVE_MAX_LEN))
        line = ex.prompt + ex.answer + "\n"
        lines.append(line)
        total += len(line)
    return "".join(lines)[:n_chars]


def native_lines(rng: random.Random, k: int) -> str:
    tasks = list(NATIVE_TASKS.values())
    lines = [rng.choice(tasks)(rng, rng.randint(1, NATIVE_MAX_LEN)) for _ in range(k)]
    return "".join(ex.prompt + ex.answer + "\n" for ex in lines)


def text_before_newline(text: np.ndarray, rng: random.Random, length: int) -> str:
    newline = TOKENIZER.id("\n")
    while True:
        end = rng.randrange(length, len(text))
        if text[end - 1] == newline:
            return TOKENIZER.decode(text[end - length : end].tolist())


class MixtureSampler:
    def __init__(self, text: np.ndarray, seq_len: int, text_frac: float, seed: int) -> None:
        self.text = text
        self.seq_len = seq_len
        self.text_frac = text_frac
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def batch(self, batch_size: int) -> Tensor:
        rows = np.empty((batch_size, self.seq_len + 1), dtype=np.int64)
        rows[:, 0] = BOS
        n_text = round(batch_size * self.text_frac)
        starts = self.np_rng.integers(0, len(self.text) - self.seq_len, size=n_text)
        for r, s in enumerate(starts):
            rows[r, 1:] = self.text[s : s + self.seq_len]
        for r in range(n_text, batch_size):
            rows[r, 1:] = TOKENIZER.encode(native_pack(self.rng, self.seq_len), bos=False)
        return torch.from_numpy(rows)


def text_eval_chunks(text: np.ndarray, seq_len: int, n_chunks: int, seed: int) -> Tensor:
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(text) - seq_len, size=n_chunks)
    rows = np.empty((n_chunks, seq_len + 1), dtype=np.int64)
    rows[:, 0] = BOS
    for r, s in enumerate(starts):
        rows[r, 1:] = text[s : s + seq_len]
    return torch.from_numpy(rows)


def focus_windows(text: np.ndarray, char: str, n: int, seq_len: int, seed: int) -> Tensor:
    hits = np.flatnonzero(text == TOKENIZER.id(char))
    hits = hits[(hits > seq_len // 2) & (hits < len(text) - seq_len)]
    picks = np.random.default_rng(seed).choice(hits, size=min(n, len(hits)), replace=False)
    rows = np.empty((len(picks), seq_len + 1), dtype=np.int64)
    rows[:, 0] = BOS
    for r, h in enumerate(picks):
        start = h - seq_len // 2
        rows[r, 1:] = text[start : start + seq_len]
    return torch.from_numpy(rows)


def prefixed_text_chunks(
    text: np.ndarray, seq_len: int, n_chunks: int, seed: int, prefix: str
) -> Tensor:
    rng = np.random.default_rng(seed)
    digits = set(TOKENIZER.digit_ids())
    head = TOKENIZER.encode(prefix, bos=False)
    width = seq_len - len(head)
    rows = np.empty((n_chunks, seq_len + 1), dtype=np.int64)
    rows[:, 0] = BOS
    rows[:, 1 : 1 + len(head)] = head
    r = 0
    while r < n_chunks:
        s = int(rng.integers(0, len(text) - width))
        if int(text[s]) in digits:
            continue
        rows[r, 1 + len(head) :] = text[s : s + width]
        r += 1
    return torch.from_numpy(rows)
