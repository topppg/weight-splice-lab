import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from weight_splice_lab.gpu import gpu_session
from weight_splice_lab.toy.data import TOKENIZER, MixtureSampler, load_wikitext, text_eval_chunks
from weight_splice_lab.toy.metrics import native_task_accuracy, target_accuracy, text_bpc
from weight_splice_lab.toy.model import ToyConfig, ToyLM, count_params, save_model
from weight_splice_lab.toy.train import TrainConfig, mps_gb, train_base

RUN_DIR = Path("runs/phase2")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", default=str(RUN_DIR / "base.pt"))
    args = parser.parse_args()

    torch.manual_seed(0)
    cfg = ToyConfig(vocab_size=TOKENIZER.vocab_size)
    tcfg = TrainConfig(steps=args.steps, batch_size=args.batch_size, warmup=200, text_frac=0.5)
    text = load_wikitext("train", max_chars=40_000_000)
    val = text_eval_chunks(load_wikitext("validation"), 256, 256, seed=0)
    with gpu_session() as budget_gb:
        model = ToyLM(cfg).to("mps")
        print(f"params {count_params(model):,} budget {budget_gb:.1f} GB")
        sampler = MixtureSampler(text, tcfg.seq_len, tcfg.text_frac, tcfg.seed)
        history = train_base(model, sampler, tcfg, "mps")
        save_model(model, Path(args.out))
        report = {
            "params": count_params(model),
            "model_cfg": asdict(cfg),
            "train_cfg": asdict(tcfg),
            "final_train_loss": history[-1]["loss"],
            "train_seconds": history[-1]["elapsed"],
            "val_bpc": text_bpc(model, val),
            "native_tasks": native_task_accuracy(model),
            "addition": target_accuracy(model, list(range(1, 9))),
            "peak_mps_gb": max(max(r["mps_gb"] for r in history), mps_gb()),
        }
    print(json.dumps(report, indent=2))
    Path(args.out).with_suffix(".json").write_text(
        json.dumps({"report": report, "history": history}, indent=2)
    )


if __name__ == "__main__":
    main()
