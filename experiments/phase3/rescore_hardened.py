import argparse
import json
import time
from pathlib import Path

import torch

from weight_splice_lab.gpu import available_gb
from weight_splice_lab.harness import evaluate
from weight_splice_lab.harness.edit import apply_edit, load_edit
from weight_splice_lab.toy.model import ToyLM, load_model
from weight_splice_lab.toy.train import peak_rss_gb

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "runs/phase2/base.pt"
OUT = REPO / "runs/phase3/rescore_hardened.json"
BASE_CACHE = REPO / "runs/phase3/base_eval_n200.json"
DEFAULT_MODELS = [
    f"base={BASE}",
    f"gated_splice_nmax8={REPO / 'runs/phase2/spliced_gated_add_nmax8_seed99.pt'}",
    f"ungated_splice_nmax20={REPO / 'runs/phase2/spliced_ungated_add_nmax20_seed99.pt'}",
    f"pilot1b_edit16={REPO / 'runs/phase3/pilot1b/pilot1b_final_edit16.pt'}",
]


def load_any(path: Path, base: ToyLM) -> ToyLM:
    if "state" in torch.load(path, map_location="cpu", weights_only=True):
        return load_model(path).eval()
    return apply_edit(base, load_edit(path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "models", nargs="*", default=DEFAULT_MODELS, help="name=path (.pt model or edit)"
    )
    ap.add_argument("--n-per", type=int, default=200)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    torch.set_num_threads(4)
    base = load_model(BASE).eval()
    chunks = evaluate.held_out_chunks()
    test = evaluate.held_out_test_text()
    results = json.loads(a.out.read_text()) if a.out.exists() else {}
    for spec in a.models:
        name, raw = spec.split("=", 1)
        path = Path(raw).resolve()
        if available_gb() < 4.0:
            raise SystemExit(f"only {available_gb():.1f} GB available; stopping before {name}")
        t0 = time.time()
        model = load_any(path, base)
        res = evaluate.evaluate_model(model, base, chunks, a.n_per, test)
        res["edit"] = str(path.relative_to(REPO) if path.is_relative_to(REPO) else path)
        results[name] = res
        a.out.write_text(json.dumps(results, indent=1))
        if name == "base" and a.n_per == 200:
            BASE_CACHE.write_text(json.dumps(res | {"edit": None}, indent=1))
        print(f"{name}: {time.time() - t0:.0f} s, rss {peak_rss_gb():.2f} GB", flush=True)
        del model


if __name__ == "__main__":
    main()
