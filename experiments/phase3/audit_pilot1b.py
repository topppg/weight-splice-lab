"""Independent audit of run pilot1b: data-isolation checks on the logs and a CPU re-evaluation of the final edit.

Reads only (never writes) the live run directory; all outputs go to --out.
"""

import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from weight_splice_lab.harness import evaluate as harness_eval
from weight_splice_lab.harness.edit import Edit, apply_edit, load_edit
from weight_splice_lab.toy import metrics
from weight_splice_lab.toy.data import (
    TOKENIZER,
    WIKITEXT_DIR,
    addition_example,
    load_wikitext,
    text_eval_chunks,
)
from weight_splice_lab.toy.model import ToyLM, load_model

metrics.free_cache = lambda: None  # CPU only, never touch the MPS allocator

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "runs/phase2/base.pt"
LOGS = REPO / "runs/phase3/pilot1b"
RUN = Path.home() / "wsl-runs/pilot1b"
FRESH_SEED = 90210  # differs from the hidden eval (4321) and practice (20261105) seeds
NATIVE_SEED = 5150  # hidden eval uses 1234
TEXT_SEED = 7
WINDOW = 64


def load_test_text() -> np.ndarray:
    table = pq.read_table(WIKITEXT_DIR / "test-00000-of-00001.parquet")
    return TOKENIZER.encode_array("".join(table.column("text").to_pylist()))


def shingles(text: str, w: int = WINDOW) -> set[str]:
    return {text[i : i + w] for i in range(len(text) - w + 1)}


def decode_chunks(chunks: torch.Tensor) -> list[str]:
    return [TOKENIZER.decode(row.tolist()) for row in chunks]


def overlap(chunks: list[str], ref: set[str], w: int = WINDOW) -> dict[str, int]:
    hit_chunks, hit_windows, total = 0, 0, 0
    for c in chunks:
        hits = sum(c[i : i + w] in ref for i in range(len(c) - w + 1))
        total += max(len(c) - w + 1, 0)
        hit_windows += hits
        hit_chunks += hits > 0
    return {"chunks": len(chunks), "chunks_with_hit": hit_chunks, "windows": total,
            "windows_hit": hit_windows}  # fmt: skip


DIGIT_TASK = re.compile(r"^[ACRM]:[0-9+,=]*\n?$")


def isolation_checks(practice: Path) -> dict[str, object]:
    text = (practice / "text.txt").read_text()
    prac_sh = shingles(text)
    val_chunks = decode_chunks(text_eval_chunks(load_wikitext("validation"), 256, 256, seed=0))
    val_text = TOKENIZER.decode(load_wikitext("validation").tolist())
    out: dict[str, object] = {
        "practice_text_vs_hidden_eval_chunks": overlap(val_chunks, prac_sh),
        "practice_text_vs_reeval_test_chunks": overlap(
            decode_chunks(text_eval_chunks(load_test_text(), 256, 1024, seed=TEXT_SEED)), prac_sh
        ),
    }
    # Every text or prompt the editor sent through the forward tool.
    rows = [json.loads(x) for x in (LOGS / "tool_calls.jsonl").read_text().splitlines()]
    sent: list[str] = []
    for r in rows:
        if r["tool"] == "forward":
            req = r["request"]
            sent += req.get("texts", []) + req.get("prompts", [])
    kinds = {"digit_task": 0, "substring_of_practice_text": 0, "other": 0}
    others: list[str] = []
    for s in sent:
        if DIGIT_TASK.match(s):
            kinds["digit_task"] += 1
        elif s in text:
            kinds["substring_of_practice_text"] += 1
        else:
            kinds["other"] += 1
            others.append(s)
    val_sh = shingles(val_text, 32)
    english = [s for s in sent if not DIGIT_TASK.match(s)]
    out["editor_forward_inputs"] = kinds | {
        "other_examples": sorted(set(others))[:10],
        "english_inputs_with_32char_window_in_validation_split": sum(
            any(s[i : i + 32] in val_sh for i in range(len(s) - 31)) for s in english
        ),
    }
    # Hidden eval problems that coincide with practice problems (same N, same prompt).
    practice_prompts = {json.loads(x)["prompt"] for x in (practice / "addition.jsonl").open()}
    rng = random.Random(4321)
    coincide: dict[str, int] = {}
    for n in range(1, 21):
        exs = [addition_example(rng, n) for _ in range(200)]
        coincide[str(n)] = sum(e.prompt in practice_prompts for e in exs)
    out["hidden_eval_addition_problems_in_practice_set"] = {
        k: v for k, v in coincide.items() if v
    } | {"total": sum(coincide.values())}
    return out


def unit_report(base: ToyLM, edit: Edit) -> dict[str, object]:
    t = edit.tensors
    d, hd = base.cfg.d_model, base.cfg.head_dim
    per_layer, inactive_params, active_params, nonzero_active = [], 0, 0, 0
    for layer, (nh, nm) in enumerate(zip(edit.arch["n_heads"], edit.arch["d_mlp"], strict=True)):
        o = t[f"blocks.{layer}.attn.o_proj.weight"]
        heads = [h for h in range(base.cfg.n_heads, nh)]
        dead_h = [h for h in heads if not o[:, h * hd : (h + 1) * hd].any()]
        down = t.get(f"blocks.{layer}.mlp.down_proj.weight")
        neurons = list(range(base.cfg.d_mlp, nm))
        dead_n = [j for j in neurons if down is None or not down[:, j].any()]
        inactive = len(dead_h) * 4 * d * hd + len(dead_n) * 3 * d
        total = len(heads) * 4 * d * hd + len(neurons) * 3 * d
        inactive_params += inactive
        active_params += total - inactive
        nz = 0
        for h in sorted(set(heads) - set(dead_h)):
            sl = slice(h * hd, (h + 1) * hd)
            nz += int(o[:, sl].count_nonzero())
            for p in "qkv":
                nz += int(t[f"blocks.{layer}.attn.{p}_proj.weight"][sl].count_nonzero())
        for j in sorted(set(neurons) - set(dead_n)):
            nz += int(down[:, j].count_nonzero())  # type: ignore[index]
            for p in ("gate", "up"):
                nz += int(t[f"blocks.{layer}.mlp.{p}_proj.weight"][j].count_nonzero())
        nonzero_active += nz
        per_layer.append({"layer": layer, "added_heads": len(heads), "zero_output_heads": len(dead_h),
                          "added_neurons": len(neurons), "zero_output_neurons": len(dead_n),
                          "added_params": total, "inactive_params": inactive,
                          "nonzero_entries_in_active_units": nz})  # fmt: skip
    return {"per_layer": per_layer, "inactive_params": inactive_params,
            "active_params": active_params, "nonzero_entries_in_active_units": nonzero_active}  # fmt: skip


def same_edit(a: Edit, b: Edit) -> bool:
    return (
        a.arch == b.arch
        and a.tensors.keys() == b.tensors.keys()
        and all(torch.equal(a.tensors[k], b.tensors[k]) for k in a.tensors)
    )


@torch.inference_mode()
def kl_bpc(model: ToyLM, base: ToyLM, chunks: torch.Tensor) -> dict[str, float]:
    kls: list[float] = []
    bits_e = bits_b = 0.0
    count = 0
    for i in range(0, len(chunks), 16):
        batch = chunks[i : i + 16]
        lp = model(batch[:, :-1]).log_softmax(-1)
        lq = base(batch[:, :-1]).log_softmax(-1)
        kls += (lq.exp() * (lq - lp)).sum(-1).mean(-1).tolist()
        tgt = batch[:, 1:, None]
        bits_e -= lp.gather(-1, tgt).sum().item()
        bits_b -= lq.gather(-1, tgt).sum().item()
        count += tgt.numel()
    k = np.array(kls)
    return {"kl": float(k.mean()), "kl_chunk_p50": float(np.median(k)),
            "kl_chunk_p99": float(np.quantile(k, 0.99)), "kl_chunk_max": float(k.max()),
            "bpc_edited": bits_e / count / np.log(2), "bpc_base": bits_b / count / np.log(2)}  # fmt: skip


@torch.inference_mode()
def prefixed_addition(model: ToyLM, prefix: str, n: int, k: int, seed: int) -> float:
    rng = random.Random(seed)
    exs = [addition_example(rng, n) for _ in range(k)]
    ids = torch.tensor([TOKENIZER.encode(prefix + e.prompt) for e in exs])
    out = model.generate(ids, n + 1)[:, ids.shape[1] :].tolist()
    return sum(TOKENIZER.decode(o) == e.answer for o, e in zip(out, exs, strict=True)) / k


def reeval(base: ToyLM, model: ToyLM) -> dict[str, object]:
    res: dict[str, object] = {}
    t0 = time.time()
    res["hidden_eval_replica"] = harness_eval.evaluate_model(
        model, base, harness_eval.held_out_chunks(), 200
    )
    print("replica", round(time.time() - t0), "s", flush=True)
    acc = metrics.target_accuracy(model, list(range(1, 21)), n_per=200, seed=FRESH_SEED)
    res["addition_fresh_n1_20"] = {str(n): a for n, a in acc.items()}
    acc_t = metrics.target_accuracy(model, list(range(1, 21)), n_per=200, seed=FRESH_SEED,
                                    with_terminator=True)  # fmt: skip
    res["addition_fresh_n1_20_with_newline"] = {str(n): a for n, a in acc_t.items()}
    gen = metrics.target_accuracy(model, list(range(21, 31)), n_per=200, seed=FRESH_SEED + 1)
    res["addition_n21_30"] = {str(n): a for n, a in gen.items()}
    gen_b = metrics.target_accuracy(base, list(range(21, 31)), n_per=200, seed=FRESH_SEED + 1)
    res["addition_n21_30_base"] = {str(n): a for n, a in gen_b.items()}
    print("addition", round(time.time() - t0), "s", flush=True)
    res["native_fresh"] = metrics.native_task_accuracy(model, seed=NATIVE_SEED)
    res["native_fresh_base"] = metrics.native_task_accuracy(base, seed=NATIVE_SEED)
    test = load_test_text()
    chunks = text_eval_chunks(test, 256, 1024, seed=TEXT_SEED)
    res["wikitext_test_1024x256"] = kl_bpc(model, base, chunks)
    a_chunks = chunks[:256].clone()
    a_chunks[:, 1:3] = torch.tensor([TOKENIZER.id("A"), TOKENIZER.id(":")])
    res["wikitext_test_256_prefixed_A_colon"] = kl_bpc(model, base, a_chunks)
    print("text", round(time.time() - t0), "s", flush=True)
    res["addition_after_native_line"] = {
        str(n): prefixed_addition(model, "C:123=123\n", n, 100, FRESH_SEED + n) for n in (1, 5, 10)
    }
    res["addition_after_space"] = {
        str(n): prefixed_addition(model, " ", n, 100, FRESH_SEED + n) for n in (1, 5, 10)
    }
    samples = []
    for p in SAMPLE_PROMPTS:
        ids = torch.tensor([TOKENIZER.encode(p)])
        row = {"prompt": p}
        for name, m in (("base", base), ("edited", model)):
            row[name] = TOKENIZER.decode(m.generate(ids, 30)[0, ids.shape[1] :].tolist())
        samples.append(row)
    res["generation_samples"] = samples
    return res


SAMPLE_PROMPTS = [
    "A:398+259=",
    "A:12345678+87654321=",
    "A: The film was released in",
    "A:123",
    "Q: The film was released in",
    "C:123=123\nA:398+259=",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edit", type=Path, default=RUN / "harness/edits/edit_16/edit.pt")
    ap.add_argument("--prev-edit", type=Path, default=RUN / "harness/edits/edit_15/edit.pt")
    ap.add_argument("--practice", type=Path, default=RUN / "sandbox/practice")
    ap.add_argument("--out", type=Path, default=LOGS / "independent_audit.json")
    ap.add_argument("--static-only", action="store_true", help="skip the re-evaluation")
    a = ap.parse_args()
    torch.set_num_threads(4)
    base = load_model(BASE).eval()
    edit = load_edit(a.edit)
    model = apply_edit(base, edit)
    result: dict[str, object] = {
        "edit": str(a.edit),
        "seeds": {"addition": FRESH_SEED, "native": NATIVE_SEED, "text": TEXT_SEED},
        "edit16_equals_edit15": same_edit(edit, load_edit(a.prev_edit)),
        "units": unit_report(base, edit),
        "isolation": isolation_checks(a.practice),
    }
    print(json.dumps(result, indent=1), flush=True)
    if a.static_only:
        return
    result["reeval"] = reeval(base, model)
    a.out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result["reeval"], indent=1))


if __name__ == "__main__":
    main()
