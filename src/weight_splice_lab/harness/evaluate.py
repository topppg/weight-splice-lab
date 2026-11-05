import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from weight_splice_lab.harness.edit import apply_edit, load_edit
from weight_splice_lab.toy import metrics
from weight_splice_lab.toy.data import (
    WIKITEXT_DIR,
    load_wikitext,
    native_lines,
    prefixed_text_chunks,
    text_before_newline,
    text_eval_chunks,
)
from weight_splice_lab.toy.model import ToyLM, count_params, load_model

LENGTHS = list(range(1, 21))
KL_BAR = 0.05
NATIVE_BAR = 0.98
ADDITION_BAR = 0.95

# Robustness block, pre-registered 2026-11-05 before pilots 2-3 (docs/eval-hardened.md).
PRIMARY_LENGTHS = list(range(1, 9))
LONG_LENGTHS = list(range(21, 31))
END_OF_ANSWER = "\n"
PREFIXED_KL_BAR = 0.05
TERMINATION_BAR = 0.95
NOT_FIRST_LINE_BAR = 0.95
ADDITION_SEED = 4321
PREFIX_TEXT_SEED = 7
NOT_FIRST_LINE_SEED = 4323
LONG_SEED = 4325
TEXT_PREFIX_LENGTHS = (32, 64, 96, 128)
NATIVE_SEED = 1234
TEXT_SEED = 0
PKG = Path(__file__).resolve().parents[1]
EVAL_SOURCES = (
    "harness/evaluate.py",
    "harness/edit.py",
    "toy/metrics.py",
    "toy/data.py",
    "toy/model.py",
)

# CPU only: metrics.free_cache would otherwise touch the MPS allocator outside gpu_session.
metrics.free_cache = lambda: None


def held_out_chunks() -> torch.Tensor:
    return text_eval_chunks(load_wikitext("validation"), 256, 256, seed=TEXT_SEED)


def held_out_test_text() -> np.ndarray:
    return load_wikitext("test")


def by_n(acc: dict[int, float]) -> dict[str, object]:
    return {"by_n": {str(n): a for n, a in acc.items()}, "mean": sum(acc.values()) / len(acc)}


def robustness(model: ToyLM, base: ToyLM, test_text: np.ndarray, n_per: int) -> dict[str, object]:
    chunks = prefixed_text_chunks(test_text, 256, 256, PREFIX_TEXT_SEED, "A:")
    kl = metrics.kl_to_reference(model, base, chunks)
    term = metrics.termination(model, PRIMARY_LENGTHS, n_per, ADDITION_SEED, END_OF_ANSWER)
    term_mean = sum(term["terminated"].values()) / len(PRIMARY_LENGTHS)

    def after_native(rng: random.Random) -> str:
        return native_lines(rng, rng.randint(1, 3))

    def after_text(rng: random.Random) -> str:
        return text_before_newline(test_text, rng, rng.choice(TEXT_PREFIX_LENGTHS))

    not_first = {
        name: metrics.prefixed_target_accuracy(
            model, PRIMARY_LENGTHS, n_per, NOT_FIRST_LINE_SEED + i, prefix
        )
        for i, (name, prefix) in enumerate(
            (("after_native_lines", after_native), ("after_text", after_text))
        )
    }
    min_mean = min(sum(acc.values()) / len(acc) for acc in not_first.values())
    long = metrics.target_accuracy(model, LONG_LENGTHS, n_per=n_per, seed=LONG_SEED)
    return {
        "thresholds": {
            "a_prefixed_text_kl_lt": PREFIXED_KL_BAR,
            "b_termination_ge": TERMINATION_BAR,
            "c_not_first_line_ge": NOT_FIRST_LINE_BAR,
        },
        "a_prefixed_text": {
            "kl": kl,
            "bpc": metrics.text_bpc(model, chunks),
            "bpc_base": metrics.text_bpc(base, chunks),
            "n_chunks": len(chunks),
            "pass": kl < PREFIXED_KL_BAR,
        },
        "b_termination": {
            "end_token": END_OF_ANSWER,
            "terminated": by_n(term["terminated"]),
            "strict_accuracy": by_n(term["strict"]),
            "pass": term_mean >= TERMINATION_BAR,
        },
        "c_not_first_line": {name: by_n(acc) for name, acc in not_first.items()}
        | {"min_mean": min_mean, "pass": min_mean >= NOT_FIRST_LINE_BAR},
        "d_length_n21_30": by_n(long),
        "e_native_weight_change": metrics.native_weight_change(model, base),
        "f_added_params": metrics.added_param_report(model, base),
    }


def robust_pass(rob: dict[str, object]) -> bool:
    keys = ("a_prefixed_text", "b_termination", "c_not_first_line")
    return all(isinstance(r := rob[k], dict) and r["pass"] is True for k in keys)


def evaluate_model(
    model: ToyLM,
    base: ToyLM,
    chunks: torch.Tensor,
    n_per: int,
    test_text: np.ndarray | None = None,
) -> dict[str, object]:
    t0 = time.time()
    model.eval()
    acc = metrics.target_accuracy(model, LENGTHS, n_per=n_per)
    reg = metrics.regression_report(model, base, chunks)
    native = {k[5:]: v for k, v in reg.items() if k.startswith("task_")}
    mean_1_8 = sum(acc[n] for n in range(1, 9)) / 8
    meets_bar = reg["text_kl"] < KL_BAR and min(native.values()) >= NATIVE_BAR
    result: dict[str, object] = {
        "addition_by_n": {str(n): a for n, a in acc.items()},
        "addition_mean_n1_8": mean_1_8,
        "addition_mean_n9_20": sum(acc[n] for n in range(9, 21)) / 12,
        "text_bpc": reg["text_bpc"],
        "text_kl": reg["text_kl"],
        "native": native,
        "meets_bar": meets_bar,
        "params": count_params(model),
        "n_per_length": n_per,
    }
    if test_text is not None:
        rob = robustness(model, base, test_text, n_per)
        primary = meets_bar and mean_1_8 >= ADDITION_BAR
        result["meets_primary_bar"] = primary
        result["meets_strict_bar"] = primary and robust_pass(rob)
        result["robustness"] = rob
    result["seconds"] = round(time.time() - t0, 1)
    return result


def evaluate_edit(
    base_ckpt: Path, edit_path: Path | None, n_per: int = 200, threads: int = 4
) -> dict[str, object]:
    torch.set_num_threads(threads)
    base = load_model(base_ckpt).eval()
    model = apply_edit(base, load_edit(edit_path)) if edit_path is not None else base
    result = evaluate_model(model, base, held_out_chunks(), n_per, held_out_test_text())
    result["edit"] = str(edit_path) if edit_path is not None else None
    return result


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def eval_code_hash() -> str:
    return _sha256(b"".join(n.encode() + b"\0" + (PKG / n).read_bytes() for n in EVAL_SOURCES))


def eval_key(base_ckpt: Path, n_per: int) -> dict[str, object]:
    return {
        "checkpoint_sha256": _sha256(base_ckpt.read_bytes()),
        "eval_code_sha256": eval_code_hash(),
        "dataset": f"{WIKITEXT_DIR.parent.name}/{WIKITEXT_DIR.name}",
        "seeds": {"addition": ADDITION_SEED, "native": NATIVE_SEED, "text": TEXT_SEED,
                  "prefix_text": PREFIX_TEXT_SEED, "not_first_line": NOT_FIRST_LINE_SEED,
                  "long": LONG_SEED},
        "thresholds": {"kl": KL_BAR, "native": NATIVE_BAR, "addition": ADDITION_BAR,
                       "prefixed_kl": PREFIXED_KL_BAR, "termination": TERMINATION_BAR,
                       "not_first_line": NOT_FIRST_LINE_BAR},
        "n_per": n_per,
    }  # fmt: skip


def base_reference(base_ckpt: Path, cache_dir: Path, n_per: int = 200) -> dict[str, object]:
    # cached per (checkpoint, eval code, data, seeds, thresholds, n); any change recomputes
    key = eval_key(base_ckpt, n_per)
    digest = _sha256(json.dumps(key, sort_keys=True).encode())[:16]
    cache = cache_dir / f"base_eval_n{n_per}_{digest}.json"
    if cache.exists():
        cached = json.loads(cache.read_text())
        if cached.get("cache_key") == key:
            return cached
    result = evaluate_edit(base_ckpt, None, n_per) | {"cache_key": key}
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=1))
    tmp.replace(cache)
    return result
