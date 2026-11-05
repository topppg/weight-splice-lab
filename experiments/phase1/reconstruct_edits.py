import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from weight_splice_lab.edits import (
    ablate_direction,
    bf16_bits,
    bf16_to_f32,
    count_changed,
    load_published_edit,
    read_tensors,
    rebuild_published_edit,
    recover_rank_one,
    safetensors_file,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models/qwen3.5-0.8b"
EXP8 = ROOT / "external/lm_understand-logs/exp8_math"
RUNS = ("p10c", "p25c", "p50c")


def verify_published(report: dict[str, object]) -> None:
    for run in RUNS:
        edits = load_published_edit(EXP8 / f"runs/{run}/edit")
        rebuilt = rebuild_published_edit(MODEL, EXP8 / f"runs/{run}/edit")
        base = read_tensors(safetensors_file(MODEL), rebuilt)
        changed = {n: count_changed(base[n], rebuilt[n]) for n in rebuilt}
        assert all(changed[e.name] == e.changed_entries for e in edits)
        report[run] = {
            "tensors": len(rebuilt),
            "changed_bf16_entries": sum(changed.values()),
            "hashes_ok": True,
        }
        print(run, report[run])


def analyse_p50c(report: dict[str, object]) -> dict[int, np.ndarray]:
    rebuilt = rebuild_published_edit(MODEL, EXP8 / "runs/p50c/edit")
    base = read_tensors(safetensors_file(MODEL), rebuilt)
    frozen = json.loads((EXP8 / "runs/p50c/work_notes/frozen_weights_manifest.json").read_text())
    directions: dict[int, np.ndarray] = {}
    out: dict[str, object] = {}
    for layer in (15, 19):
        name = f"model.language_model.layers.{layer}.mlp.down_proj.weight"
        w0 = bf16_to_f32(base[name]).astype(np.float64)
        delta = bf16_to_f32(rebuilt[name]).astype(np.float64) - w0
        cols = np.flatnonzero(np.abs(delta).max(axis=0) > 0)
        u, _, share = recover_rank_one(delta[:, cols])
        # delta[:, cols] = -alpha * u (u^T W0)[cols]; least-squares alpha
        c = u @ w0[:, cols]
        alpha = float(-(u @ delta[:, cols] @ c) / (c @ c))
        directions[layer] = u
        selected = frozen[str(layer)]["selected_neurons"]
        out[f"layer{layer}"] = {
            "changed_columns": int(cols.size),
            "columns_match_frozen_manifest": cols.tolist() == sorted(selected),
            "rank1_energy_share": share,
            "implied_alpha": alpha,
            "touched_weights": int(cols.size * w0.shape[0]),
        }
        print(layer, out[f"layer{layer}"])
    out["cos_u15_u19"] = float(abs(directions[15] @ directions[19]))
    report["p50c_analysis"] = out
    return directions


def analyse_p10c(report: dict[str, object]) -> None:
    edits = load_published_edit(EXP8 / "runs/p10c/edit")
    base = read_tensors(safetensors_file(MODEL), [e.name for e in edits])
    us, alphas = [], []
    for e in edits:
        assert e.a is not None and e.b is not None
        u = e.a[:, 0] / np.linalg.norm(e.a[:, 0])
        coeff = u @ bf16_to_f32(base[e.name]).astype(np.float64)
        # a b = -alpha u (u^T W)
        v = np.linalg.norm(e.a[:, 0]) * e.b[0]
        alphas.append(float(-(v @ coeff) / (coeff @ coeff)))
        us.append(u)
    cos = np.abs(np.stack(us) @ np.stack(us).T)
    report["p10c_analysis"] = {
        "matrices": len(edits),
        "implied_alpha_min_max": [min(alphas), max(alphas)],
        "abs_cos_between_directions_min": float(cos.min()),
    }
    print(report["p10c_analysis"])


def analyse_p25c(report: dict[str, object]) -> None:
    rebuilt = rebuild_published_edit(MODEL, EXP8 / "runs/p25c/edit")
    ((name, bits),) = rebuilt.items()
    w = bf16_to_f32(bits)
    zero_cols = np.flatnonzero(np.all(w == 0, axis=0))
    report["p25c_analysis"] = {
        "tensor": name,
        "zero_columns": [int(zero_cols.min()), int(zero_cols.max())],
        "n_zero_columns": int(zero_cols.size),
        "head": int(zero_cols.min()) // 256,
    }
    print(report["p25c_analysis"])


def recompute_direction(report: dict[str, object], published: dict[int, np.ndarray]) -> None:
    # Re-derive u from Astra's four matched productive/repetitive texts (its teacher-forced pass fwd_4):
    # u_l = mean over pairs of (repetitive - productive) residual after layer l, averaged over the last 36 tokens.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    calls = [json.loads(x) for x in (EXP8 / "runs/p50c/tool_calls.jsonl").read_text().splitlines()]
    req = next(c["request"] for c in calls if c.get("kind") == "forward" and c.get("k") == 4)
    texts: list[str] = req["texts"]
    tok = AutoTokenizer.from_pretrained(MODEL)
    # bf16 on CPU keeps RAM ~2 GB; the logged pass also ran in bf16
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    model.eval()
    w15 = model.model.layers[15].mlp.down_proj.weight.detach().float().numpy()
    acts: dict[int, list[np.ndarray]] = {15: [], 19: []}
    hooks = []

    def make_hook(layer: int) -> Callable[..., None]:
        def hook(_m: torch.nn.Module, inp: tuple[torch.Tensor, ...], _out: torch.Tensor) -> None:
            acts[layer].append(inp[0][0].detach().float().numpy())

        return hook

    for layer in acts:
        down = model.model.layers[layer].mlp.down_proj
        hooks.append(down.register_forward_hook(make_hook(layer)))
    hidden: dict[int, list[np.ndarray]] = {16: [], 20: []}
    lengths = []
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt", add_special_tokens=False).input_ids
            lengths.append(int(ids.shape[1]))
            hs = model(ids, output_hidden_states=True).hidden_states
            for k, store in hidden.items():
                store.append(hs[k][0].float().numpy())
    for h in hooks:
        h.remove()
    w0 = read_tensors(
        safetensors_file(MODEL),
        [f"model.language_model.layers.{layer}.mlp.down_proj.weight" for layer in (15, 19)],
    )
    rebuilt = rebuild_published_edit(MODEL, EXP8 / "runs/p50c/edit")
    out: dict[str, object] = {"token_lengths": lengths}
    assert np.array_equal(
        w15, bf16_to_f32(w0["model.language_model.layers.15.mlp.down_proj.weight"])
    ), "transformers did not load the checkpoint weights"
    for layer, k in ((15, 16), (19, 20)):
        means = np.stack([h[n - 36 : n].mean(0) for h, n in zip(hidden[k], lengths)])
        d = (means[1::2] - means[::2]).mean(0)
        u = d / np.linalg.norm(d)
        w = bf16_to_f32(w0[f"model.language_model.layers.{layer}.mlp.down_proj.weight"])
        m = np.stack([a[n - 24 : n].mean(0) for a, n in zip(acts[layer], lengths)])
        score = (u @ w) * (m[1::2] - m[::2]).mean(0)
        top = np.argsort(-score)[:64]
        frozen = json.loads(
            (EXP8 / "runs/p50c/work_notes/frozen_weights_manifest.json").read_text()
        )
        selected = set(frozen[str(layer)]["selected_neurons"])
        # rebuild the edit from the recomputed u and the published neuron list, compare bit for bit
        name = f"model.language_model.layers.{layer}.mlp.down_proj.weight"
        mine = bf16_bits(
            ablate_direction(w.astype(np.float64), u, 6.0, sorted(selected)).astype(np.float32)
        )
        target = rebuilt[name]
        out[f"layer{layer}"] = {
            "abs_cos_with_published_u": float(abs(u @ published[layer])),
            "top64_overlap_with_published": len(selected & set(top.tolist())),
            "from_u_changed_entries": count_changed(w0[name], mine),
            "from_u_entries_equal_to_published": int(np.count_nonzero(mine == target)),
            "from_u_max_abs_diff_vs_published": float(
                np.abs(bf16_to_f32(mine) - bf16_to_f32(target)).max()
            ),
        }
        print(layer, out[f"layer{layer}"])
    report["recomputed_u"] = out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute-u", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "runs/phase1/edits_report.json"))
    a = ap.parse_args()
    report: dict[str, object] = {}
    verify_published(report)
    published = analyse_p50c(report)
    analyse_p10c(report)
    analyse_p25c(report)
    if a.recompute_u:
        recompute_direction(report, published)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
