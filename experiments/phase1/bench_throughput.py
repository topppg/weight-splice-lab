import argparse
import json
from pathlib import Path

from weight_splice_lab.eval import Sampling, load_math500
from weight_splice_lab.gpu import gpu_session

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-sizes", default="1,2,4,8")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument(
        "--min-gb", type=float, required=True, help="GPU budget the largest batch needs"
    )
    ap.add_argument("--out", default=str(ROOT / "runs/phase1/throughput_capped.json"))
    a = ap.parse_args()
    problems = load_math500()
    results = []
    with gpu_session(min_gb=a.min_gb) as budget_gb:
        import mlx.core as mx

        from weight_splice_lab.mlx_gen import generate_batch, load_mlx_model, prompt_tokens

        mx.random.seed(0)
        model, tok = load_mlx_model(ROOT / "models/qwen3.5-0.8b")
        for b in [int(x) for x in a.batch_sizes.split(",")]:
            prompts = [prompt_tokens(tok, p) for p in problems[:b]]
            s = Sampling(max_tokens=a.max_tokens)
            stats = generate_batch(model, tok, prompts, s, b, lambda i, t, f: None, stop=False)
            stats.update(
                batch_size=b,
                max_tokens=a.max_tokens,
                peak_gb=mx.get_peak_memory() / 1e9,
                budget_gb=budget_gb,
            )
            print(json.dumps(stats), flush=True)
            results.append(stats)
            mx.reset_peak_memory()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
