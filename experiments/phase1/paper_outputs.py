import argparse
import gzip
import json
from pathlib import Path

from transformers import AutoTokenizer

from weight_splice_lab.eval import Problem, grade, load_math500, summarize

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "external/lm_understand-logs/exp8_math/eval"
MODELS = ("unedited", "p10c", "p25c", "p50c")


def eval_cost(model: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for scores in sorted((EVAL / model / "parts").glob("*/scores.json")):
        s = json.loads(scores.read_text())
        for bench, r in s.items():
            if bench.startswith("_"):
                continue
            b = out.setdefault(
                bench, {"seconds": 0, "tokens": 0, "samples": 0, "cut": 0, "jobs": 0}
            )
            b["seconds"] += r["s"]
            b["tokens"] += r["tokens_sum"]
            b["samples"] += r["total"]
            b["cut"] += r["cut"]
            b["jobs"] += 1
    return out


def first_samples(model: str) -> dict[str, str]:
    texts: dict[str, str] = {}
    for f in sorted((EVAL / model / "parts").glob("math500_*/math500_first_samples.jsonl.gz")):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                d = json.loads(line)
                texts[d["id"]] = d["text"]
    return texts


def per_problem(model: str) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for f in sorted((EVAL / model / "parts").glob("math500_*/math500_per_problem.json")):
        for r in json.loads(f.read_text()):
            out[r["id"]] = r["correct"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--caps", default="4096,8192,16384,32768")
    ap.add_argument("--out", default=str(ROOT / "runs/phase1/paper_outputs.json"))
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(ROOT / "models/qwen3.5-0.8b")
    problems = load_math500()
    caps = [int(c) for c in a.caps.split(",")]
    report: dict[str, object] = {}
    for model in MODELS:
        texts = first_samples(model)
        pp = per_problem(model)
        # the published texts cover MATH-500 indices 0..199 (first 20 problems of each of 10 shards)
        subset: list[Problem] = [p for p in problems[: a.n] if p.id in texts]
        agree = 0
        by_cap: dict[str, object] = {}
        for cap in caps:
            records = []
            for p in subset:
                ids = tok.encode(texts[p.id], add_special_tokens=False)
                cut = len(ids) >= cap
                text = tok.decode(ids[:cap]) if cut else texts[p.id]
                rec = grade(
                    p, 0, text, tokens=min(len(ids), cap), finish="length" if cut else "stop"
                )
                records.append(rec)
                if cap == 32768:
                    agree += rec.correct == bool(pp[p.id][0])
            by_cap[str(cap)] = {k: v for k, v in summarize(records).items()}
        report[model] = {
            "subset_n": len(subset),
            "first_sample_by_cap": by_cap,
            "grader_agreement_at_32768": f"{agree}/{len(subset)}",
            "subset_4sample_acc_at_32768": sum(sum(pp[p.id]) for p in subset) / (4 * len(subset)),
            "eval_cost": eval_cost(model),
        }
        print(
            model,
            json.dumps(report[model]["first_sample_by_cap"]),
            report[model]["grader_agreement_at_32768"],
        )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
