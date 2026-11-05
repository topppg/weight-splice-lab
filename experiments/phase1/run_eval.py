import argparse
import json
import time
from pathlib import Path

from weight_splice_lab.edits import rebuild_published_edit
from weight_splice_lab.eval import (
    Sampling,
    append_record,
    grade,
    is_correct,
    load_math500,
    read_records,
    summarize,
)
from weight_splice_lab.gpu import gpu_session

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models/qwen3.5-0.8b"
EXP8 = ROOT / "external/lm_understand-logs/exp8_math"
VARIANTS = ("base", "p10c", "p25c", "p50c")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VARIANTS, required=True)
    ap.add_argument("--n", type=int, default=50, help="first n MATH-500 problems")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--min-gb", type=float, required=True, help="GPU budget the run needs")
    ap.add_argument(
        "--probe", action="store_true", help="ignore EOS, generate max_tokens; report peak"
    )
    a = ap.parse_args()

    # imported before the GPU session: pure CPU work, and math-verify must not be loaded lazily mid-run
    is_correct("1", "1")
    problems = load_math500()[: a.n]
    edited = (
        None
        if a.variant == "base"
        else rebuild_published_edit(MODEL, EXP8 / f"runs/{a.variant}/edit")
    )
    s = Sampling(max_tokens=a.max_tokens)
    tag = "probe" if a.probe else "eval"
    out = ROOT / f"runs/phase1/{tag}_{a.variant}_n{a.n}_t{a.max_tokens}_b{a.batch_size}"
    out.mkdir(parents=True, exist_ok=True)
    rec_path = out / "records.jsonl"
    done = {r.id for r in read_records(rec_path)}
    todo = [p for p in problems if p.id not in done]

    with gpu_session(min_gb=a.min_gb) as budget_gb:
        import mlx.core as mx

        from weight_splice_lab.mlx_gen import generate_batch, load_mlx_model, prompt_tokens

        mx.random.seed(0)
        model, tok = load_mlx_model(MODEL, edited)
        model_gb = mx.get_active_memory() / 1e9
        prompts = [prompt_tokens(tok, p) for p in todo]
        t_start = time.time()

        def on_done(i: int, tokens: list[int], finish: str) -> None:
            text = tok.decode(tokens)  # type: ignore[attr-defined]
            n = len(tokens) + (finish == "stop")
            append_record(rec_path, grade(todo[i], 0, text, tokens=n, finish=finish))
            print(f"[{time.time() - t_start:7.0f}s] {todo[i].id} {finish} {n} tok", flush=True)

        stats = generate_batch(model, tok, prompts, s, a.batch_size, on_done, stop=not a.probe)
        stats.update(
            peak_gb=mx.get_peak_memory() / 1e9,
            model_gb=model_gb,
            budget_gb=budget_gb,
            batch_size=a.batch_size,
            max_tokens=a.max_tokens,
            variant=a.variant,
            n_problems=len(todo),
        )

    summary = {"run": stats, "scores": summarize(read_records(rec_path)), "sampling": s.__dict__}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
