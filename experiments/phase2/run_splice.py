import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path

import torch

from weight_splice_lab.gpu import gpu_session
from weight_splice_lab.toy.data import (
    MixtureSampler,
    focus_windows,
    load_wikitext,
    text_eval_chunks,
)
from weight_splice_lab.toy.metrics import (
    free_cache,
    ladder_accuracy,
    regression_report,
    target_accuracy,
)
from weight_splice_lab.toy.model import ToyLM, count_params, load_model, save_model
from weight_splice_lab.toy.splice import NativeActs, SpliceSpec, build_splice, collect_native
from weight_splice_lab.toy.train import mps_gb, peak_rss_gb

RUN_DIR = Path("runs/phase2")
UNGATED = {"read_scale": 1.0, "text_gate": 0.0}


def native_acts(base: ToyLM, text: torch.Tensor, seed: int) -> NativeActs:
    rare = torch.cat([focus_windows(text, c, 32, 256, seed=seed) for c in "+=:"])
    return collect_native(base, MixtureSampler(text, 256, 0.5, seed=seed), n_seqs=64, extra=rare)


def lengths_for(n_max: int) -> list[int]:
    if n_max <= 8:
        return list(range(1, n_max + 3))
    return sorted({1, 2, 4, 8, n_max // 2, n_max - 4, n_max, n_max + 1})


def keep_only_markers(model: ToyLM, base: ToyLM) -> None:
    # Zero every added write except the always-on marker head (the last head added to layer 0).
    for layer, (block, ref) in enumerate(zip(model.blocks, base.blocks, strict=True)):
        n0, m0 = ref.attn.n_heads * ref.attn.head_dim, ref.mlp.gate_proj.out_features
        hd = ref.attn.head_dim
        with torch.no_grad():
            o = block.attn.o_proj.weight
            end = o.shape[1] - hd if layer == 0 else o.shape[1]
            o[:, n0:end] = 0
            block.mlp.down_proj.weight[:, m0:] = 0


def evaluate(model: ToyLM, base: ToyLM, spec: SpliceSpec, val: torch.Tensor) -> dict[str, object]:
    rung = "add" if spec.rung.startswith("add") else spec.rung
    lengths = lengths_for(spec.n_max)
    result: dict[str, object] = {"accuracy": ladder_accuracy(model, rung, lengths)}
    if spec.rung == "add":
        result["accuracy_with_terminator"] = target_accuracy(model, lengths, with_terminator=True)
    result["regression"] = regression_report(model, base, val)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=str(RUN_DIR / "base.pt"))
    parser.add_argument("--out", default=str(RUN_DIR / "splice_results.json"))
    args = parser.parse_args()

    text = load_wikitext("train", max_chars=5_000_000)
    val = text_eval_chunks(load_wikitext("validation"), 256, 256, seed=0)
    configs: list[tuple[str, SpliceSpec, int]] = []
    for gating, kw in (("gated", {}), ("ungated", UNGATED)):
        for rung, n in (("copy", 8), ("successor", 8), ("add", 1), ("add", 8), ("add", 20)):
            configs.append((gating, SpliceSpec(rung, n, **kw), 99))
    configs.append(("ungated", SpliceSpec("add_nocarry", 8, **UNGATED), 99))
    configs.append(("markers_only", SpliceSpec("add", 8), 99))
    configs += [("gated", SpliceSpec("add", 8), seed) for seed in (1, 2)]

    results: dict[str, object] = {}
    with gpu_session() as budget_gb:
        print(f"[gpu] budget {budget_gb:.1f} GB", flush=True)
        base = load_model(Path(args.base), "mps").eval()
        results["base"] = {
            "params": count_params(base),
            "ladder": {
                r: ladder_accuracy(base, r, list(range(1, 11)))
                for r in ("copy", "successor", "add")
            },
            "regression": regression_report(base, base, val),
        }
        peak = mps_gb()
        acts_by_seed: dict[int, NativeActs] = {}
        for gating, spec, seed in configs:
            if seed not in acts_by_seed:
                acts_by_seed = {seed: native_acts(base, text, seed)}
                free_cache()
            model = copy.deepcopy(base)
            report = build_splice(model, spec, acts_by_seed[seed])
            if gating == "markers_only":
                keep_only_markers(model, base)
            key = f"{gating}_{spec.rung}_nmax{spec.n_max}_seed{seed}"
            row = {
                "gating": gating,
                "probe_seed": seed,
                "spec": asdict(spec),
                "params_allocated": report.params_allocated,
                "params_nonzero": report.params_nonzero,
                "private_dims": len(report.layout),
                "added": report.added,
                "constants": report.constants,
                "ops": report.ops,
            } | evaluate(model, base, spec, val)
            results[key] = row
            peak = max(peak, mps_gb())
            print(
                key,
                json.dumps(row["accuracy"]),
                json.dumps(row["regression"]),
                f"mps {mps_gb():.2f} GB",
                flush=True,
            )
            if key in ("gated_add_nmax8_seed99", "ungated_add_nmax20_seed99"):
                save_model(model, RUN_DIR / f"spliced_{key}.pt")
            del model
            free_cache()
        results["peak_mps_gb"] = peak
        results["peak_rss_gb"] = peak_rss_gb()
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}; peak mps {peak:.2f} GB, peak rss {peak_rss_gb():.2f} GB")


if __name__ == "__main__":
    main()
