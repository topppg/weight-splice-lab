import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from weight_splice_lab.gpu import available_gb

MODEL = "unsloth/gemma-3-270m-it"
OUT = Path(__file__).resolve().parents[2] / "runs/phase4/probe_gemma.json"


def operands(rng: random.Random, n: int) -> tuple[int, int]:
    lo, hi = 10 ** (n - 1) if n > 1 else 0, 10**n - 1
    return rng.randint(lo, hi), rng.randint(lo, hi)


def prompt(tok: AutoTokenizer, fmt: str, a: int, b: int) -> str:
    if fmt == "raw":
        return f"{a}+{b}="
    msg = f"What is {a} + {b}? Reply with only the number."
    return tok.apply_chat_template(
        [{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True
    )


def parse(text: str) -> str:
    digits = ""
    for ch in text.strip().replace(",", ""):
        if not ch.isdigit():
            break
        digits += ch
    return digits


@torch.no_grad()
def accuracy(model, tok, fmt: str, n: int, count: int, seed: int, device: str) -> float:
    rng = random.Random(seed * 100 + n)
    pairs = [operands(rng, n) for _ in range(count)]
    texts = [prompt(tok, fmt, a, b) for a, b in pairs]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=fmt == "raw").to(device)
    out = model.generate(**enc, max_new_tokens=n + 6, do_sample=False)
    gen = tok.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
    return sum(parse(g) == str(a + b) for g, (a, b) in zip(gen, pairs, strict=True)) / count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=20)
    ap.add_argument("--count", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    if available_gb() < 4.0:
        raise SystemExit(f"only {available_gb():.1f} GB available")
    torch.set_num_threads(4)
    device = "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device).eval()
    print("digit tokens:", tok.tokenize("12345+678="), flush=True)
    results: dict[str, dict[str, float]] = {}
    for fmt in ("chat", "raw"):
        results[fmt] = {}
        for n in range(1, a.max_n + 1):
            t0 = time.time()
            acc = accuracy(model, tok, fmt, n, a.count, a.seed, device)
            results[fmt][str(n)] = acc
            print(f"{fmt} N={n}: {acc:.2f} ({time.time() - t0:.1f} s)", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"model": MODEL, "count": a.count, "results": results}, indent=1))


if __name__ == "__main__":
    main()
