import argparse
import copy
import json
import random
from pathlib import Path

from weight_splice_lab.gpu import gpu_session
from weight_splice_lab.toy.data import (
    NATIVE_MAX_LEN,
    NATIVE_TASKS,
    Example,
    addition_example,
    load_wikitext,
    text_before_newline,
    text_eval_chunks,
)
from weight_splice_lab.toy.lora import apply_lora, finetune, lora_param_count, merge_lora
from weight_splice_lab.toy.metrics import free_cache, regression_report, target_accuracy
from weight_splice_lab.toy.model import load_model, save_model
from weight_splice_lab.toy.train import mps_gb, peak_rss_gb

RUN_DIR = Path("runs/phase2")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=str(RUN_DIR / "base.pt"))
    parser.add_argument("--budgets", default="100,300,1000,3000,10000,30000,100000")
    parser.add_argument("--train-max-n", type=int, default=8)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--full", action="store_true", help="full fine-tune instead of LoRA")
    parser.add_argument("--replay", action="store_true", help="add as many native-task examples")
    parser.add_argument("--text-replay", type=int, default=0, help="fresh text windows per batch")
    parser.add_argument("--text-pool", action="store_true", help="fixed pool of budget windows")
    parser.add_argument("--text-len", type=int, default=64)
    parser.add_argument("--save-prefix", default="", help="save merged models as <prefix>_b<N>.pt")
    parser.add_argument("--out", default=str(RUN_DIR / "lora_results.json"))
    args = parser.parse_args()
    use_text = args.text_replay > 0 or args.text_pool
    train_text = load_wikitext("train", max_chars=5_000_000) if use_text else None

    def text_window(r: random.Random) -> Example:
        assert train_text is not None
        # loss on the whole window; encode_batch re-appends the window's final newline
        return Example("", text_before_newline(train_text, r, args.text_len)[:-1])

    val = text_eval_chunks(load_wikitext("validation"), 256, 256, seed=0)
    eval_lengths = list(range(1, args.train_max_n + 3))
    results: dict[str, object] = {"args": vars(args)}
    with gpu_session(min_gb=1.0) as budget_gb:
        print(f"[gpu] budget {budget_gb:.1f} GB")
        base = load_model(Path(args.base), "mps").eval()
        peak = mps_gb()
        for budget in [int(b) for b in args.budgets.split(",")]:
            rng = random.Random(1000 + budget)
            data = [addition_example(rng, rng.randint(1, args.train_max_n)) for _ in range(budget)]
            if args.replay:
                tasks = list(NATIVE_TASKS.values())
                data += [
                    rng.choice(tasks)(rng, rng.randint(1, NATIVE_MAX_LEN)) for _ in range(budget)
                ]
            if args.text_pool:
                data += [text_window(rng) for _ in range(budget)]
            model = copy.deepcopy(base)
            if args.full:
                params = list(model.parameters())
            else:
                params = apply_lora(model, args.rank, alpha=2.0 * args.rank)
            history = finetune(
                model,
                params,
                data,
                args.steps,
                args.batch_size,
                args.lr,
                seed=budget,
                device="mps",
                stream=text_window if args.text_replay > 0 else None,
                stream_per_batch=args.text_replay,
            )
            row = {
                "budget": budget,
                "examples": len(data),
                "fresh_text_windows": args.text_replay * args.steps,
                "trainable_params": sum(p.numel() for p in params)
                if args.full
                else lora_param_count(params),
                "final_loss": history[-1]["loss"],
                "seconds": history[-1]["elapsed"],
                "accuracy": target_accuracy(model, eval_lengths),
                "regression": regression_report(model, base, val),
            }
            peak = max(peak, mps_gb())
            results[str(budget)] = row
            if args.save_prefix and not args.full:
                merge_lora(model)
                save_model(model.cpu(), Path(f"{args.save_prefix}_b{budget}.pt"))
            print(
                budget,
                json.dumps(row["accuracy"]),
                json.dumps(row["regression"]),
                f"mps {mps_gb():.2f} GB",
                flush=True,
            )
            del model
            free_cache()
        results["peak_mps_gb"] = peak
        results["peak_rss_gb"] = peak_rss_gb()
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}; peak mps {peak:.2f} GB")


if __name__ == "__main__":
    main()
