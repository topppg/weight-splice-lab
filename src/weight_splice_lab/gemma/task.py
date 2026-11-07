import random
from dataclasses import dataclass
from pathlib import Path

from weight_splice_lab.gemma.tokenizer import GemmaTokenizer, load

ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = ROOT / "runs" / "phase4" / "base"
BASE_CKPT = BASE_DIR / "base.pt"
HF_MODEL = "unsloth/gemma-3-270m-it"


@dataclass(frozen=True)
class Example:
    prompt: str
    answer: str


def tokenizer() -> GemmaTokenizer:
    return load(BASE_DIR / "tokenizer.json")


def operand(rng: random.Random, n: int) -> int:
    return rng.randint(0, 9) if n == 1 else rng.randint(10 ** (n - 1), 10**n - 1)


def addition_example(rng: random.Random, n: int) -> Example:
    # plain decimal notation, most significant digit first, no leading zeros
    a, b = operand(rng, n), operand(rng, n)
    return Example(f"{a}+{b}=", str(a + b))


def reversed_example(rng: random.Random, n: int) -> Example:
    # least significant digit first, as in the toy task: N random digits per operand (any digit,
    # so zeros may end them) and the sum as exactly N + 1 digits, zero-padded at the end
    a = "".join(str(rng.randint(0, 9)) for _ in range(n))
    b = "".join(str(rng.randint(0, 9)) for _ in range(n))
    s = str(int(a[::-1]) + int(b[::-1])).zfill(n + 1)[::-1]
    return Example(f"R:{a}+{b}=", s)


def answer_tokens(tok: GemmaTokenizer, answer: str) -> list[int]:
    return tok.encode(answer, bos=False)


# The one grading rule for practice and hidden eval: the answer's digit tokens, then a token that
# is not a digit (so 1234 is wrong when the answer is 123). Generate len(answer) + 1 tokens.
def grade_ids(tok: GemmaTokenizer, generated: list[int], answer: str) -> bool:
    target = answer_tokens(tok, answer)
    n = len(target)
    return generated[:n] == target and len(generated) > n and generated[n] not in tok.digit_ids
