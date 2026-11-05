import functools
import json
import re
import threading
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
# Qwen3.5: <|endoftext|> (generation_config eos) and <|im_end|> (tokenizer eos); vLLM stops on both
STOP_TOKEN_IDS = (248044, 248046)


@dataclass(frozen=True)
class Sampling:
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    presence_penalty: float = 1.5
    max_tokens: int = 32768


@dataclass(frozen=True)
class Problem:
    id: str
    problem: str
    answer: str


def build_user_message(problem: str) -> str:
    return problem + "\n" + INSTRUCTION


def load_math500() -> list[Problem]:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return [Problem(id=r["unique_id"], problem=r["problem"], answer=r["answer"]) for r in ds]


# ---- answer extraction, a port of the paper's code/mathqa.py (extract / is_correct)

_BOX_FORMS = re.compile(r"\\boxed|\\fbox")


def answer_text(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text


def _box_at(text: str, idx: int) -> str | None:
    key = "\\boxed" if text.startswith("\\boxed", idx) else "\\fbox"
    rest = text[idx + len(key) :]
    r = rest.lstrip()
    if r.startswith("{"):
        depth = 0
        for k, ch in enumerate(r):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return r[1:k].strip() or None
        return None
    # '\boxed 5' without braces: read up to the next '$' or line end
    if key == "\\boxed" and rest.startswith(" "):
        return rest.strip().split("$")[0].split("\n")[0].strip() or None
    return None


def last_boxed(text: str) -> str | None:
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None
    return _box_at(text, idx)


def extract_answer(text: str) -> str | None:
    return last_boxed(answer_text(text))


def _timeout() -> int | None:
    # math-verify's timeouts use signal.alarm, which only works on the main thread
    return 5 if threading.current_thread() is threading.main_thread() else None


@functools.lru_cache(maxsize=1 << 16)
def _parsed(s: str, timeout: int | None) -> list[object]:
    from math_verify import parse

    return parse("\\boxed{" + s + "}", parsing_timeout=timeout)


def is_correct(pred: str | None, gold: str | None) -> bool:
    if pred is None or gold is None:
        return False
    from math_verify import verify

    to = _timeout()
    g, p = _parsed(gold, to), _parsed(pred, to)
    if not g or not p:
        return False
    try:
        return bool(verify(g, p, timeout_seconds=to))
    except Exception:  # noqa: BLE001 - math-verify raises arbitrary sympy errors; treat as not equal
        return False


# ---- loop signals


def tail_duplicate_share(text: str, n: int = 8, window: int = 200) -> float:
    # share of n-word windows among the last `window` words whose phrase occurs more than once there
    w = text.split()[-window:]
    grams = [tuple(w[i : i + n]) for i in range(len(w) - n + 1)]
    if not grams:
        return 0.0
    c = Counter(grams)
    return sum(c[g] > 1 for g in grams) / len(grams)


def verbatim_loop_end(
    text: str, max_period: int = 200, min_repeats: int = 3, min_span: int = 40
) -> bool:
    # the text (minus its last, possibly cut word) ends with one block of words repeated >= 3 times over >= 40 words
    w = text.split()[:-1]
    n = len(w)
    for p in range(1, max_period + 1):
        reps = max(min_repeats, -(-min_span // p))
        if reps * p > n:
            return False
        blk = w[n - p :]
        if all(w[n - (j + 1) * p : n - j * p] == blk for j in range(1, reps)):
            return True
    return False


# ---- records


@dataclass(frozen=True)
class Record:
    id: str
    sample: int
    answer: str
    pred: str | None
    correct: bool
    tokens: int
    finish: str
    closed_think: bool
    tail_dup8: float
    verbatim_loop: bool
    text: str

    @property
    def cut(self) -> bool:
        return self.finish == "length"


def grade(problem: Problem, sample: int, text: str, tokens: int, finish: str) -> Record:
    pred = extract_answer(text)
    return Record(
        id=problem.id,
        sample=sample,
        answer=problem.answer,
        pred=pred,
        correct=is_correct(pred, problem.answer),
        tokens=tokens,
        finish=finish,
        closed_think="</think>" in text,
        tail_dup8=tail_duplicate_share(text),
        verbatim_loop=verbatim_loop_end(text),
        text=text,
    )


def read_records(path: Path) -> list[Record]:
    if not path.exists():
        return []
    return [Record(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def append_record(path: Path, rec: Record) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")


def summarize(records: list[Record]) -> dict[str, float | int]:
    n = len(records)
    if n == 0:
        return {"n": 0}
    cut = [r for r in records if r.cut]
    finished = [r for r in records if not r.cut]
    return {
        "n": n,
        "acc": sum(r.correct for r in records) / n,
        "cut_share": len(cut) / n,
        "mean_tokens": sum(r.tokens for r in records) / n,
        "finished_wrong": sum(not r.correct for r in finished),
        "finished_no_answer": sum(r.pred is None for r in finished),
        "cut_verbatim_loop_share": sum(r.verbatim_loop for r in cut) / len(cut) if cut else 0.0,
        "cut_mean_tail_dup8": sum(r.tail_dup8 for r in cut) / len(cut) if cut else 0.0,
        # the edited model sometimes degenerates into pure whitespace instead of a verbal loop
        "cut_whitespace_tail_share": sum(not r.text[-500:].strip() for r in cut) / len(cut)
        if cut
        else 0.0,
    }
