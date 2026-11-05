import argparse
import json
import time
from pathlib import Path

import torch

from weight_splice_lab.gpu import available_gb
from weight_splice_lab.harness import evaluate
from weight_splice_lab.harness.edit import apply_edit, load_edit
from weight_splice_lab.toy import metrics
from weight_splice_lab.toy.data import load_wikitext, text_eval_chunks
from weight_splice_lab.toy.model import load_model
from weight_splice_lab.toy.train import peak_rss_gb

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "runs/phase2/base.pt"
OUT = REPO / "runs/phase3/reeval_fresh.json"
RUNS_ROOT = Path.home() / "wsl-runs"
# None of these seeds or the test split is used by the hidden eval (evaluate.py) for these measures.
ADDITION_SEED = 90210
NATIVE_SEED = 5150
TEXT_SEED = 11


def main() -> None:
    ap = argparse.ArgumentParser(
        description="re-evaluate final edits on fresh seeds and the test split"
    )
    ap.add_argument("run_ids", nargs="+")
    ap.add_argument("--n-per", type=int, default=200)
    ap.add_argument("--base", type=Path, default=BASE)
    a = ap.parse_args()
    torch.set_num_threads(4)
    base = load_model(a.base).eval()
    chunks = text_eval_chunks(load_wikitext("test"), 256, 256, seed=TEXT_SEED)
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    for run_id in a.run_ids:
        if available_gb() < 4.0:
            raise SystemExit(f"only {available_gb():.1f} GB available; stopping before {run_id}")
        t0 = time.time()
        hidden = json.loads((RUNS_ROOT / run_id / "harness/result.json").read_text())
        k = hidden["evaluated_edit"]
        model = apply_edit(base, load_edit(RUNS_ROOT / run_id / f"harness/edits/edit_{k}/edit.pt"))
        acc = metrics.target_accuracy(model, evaluate.LENGTHS, a.n_per, seed=ADDITION_SEED)
        native = metrics.native_task_accuracy(model, seed=NATIVE_SEED)
        results[run_id] = {
            "edit": k,
            "seeds": {"addition": ADDITION_SEED, "native": NATIVE_SEED, "text": TEXT_SEED},
            "text_split": "test",
            "n_per_length": a.n_per,
            "addition_by_n": {str(n): v for n, v in acc.items()},
            "text_kl": metrics.kl_to_reference(model, base, chunks),
            "native": native,
            "hidden_text_kl": hidden["hidden_eval"]["text_kl"],
            "hidden_addition_mean_n1_20": sum(hidden["hidden_eval"]["addition_by_n"].values()) / 20,
        }
        OUT.write_text(json.dumps(results, indent=1))
        print(f"{run_id}: {time.time() - t0:.0f} s, rss {peak_rss_gb():.2f} GB", flush=True)
        del model


if __name__ == "__main__":
    main()
