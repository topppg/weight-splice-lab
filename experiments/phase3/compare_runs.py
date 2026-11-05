import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs/phase3"

# (label, how the weights were written, source file, key inside it or None for a run result.json)
ROWS: list[tuple[str, str, Path, str | None]] = [
    (
        "hand splice, gated (N≤8)",
        "by hand, closed form",
        RUNS / "rescore_hardened.json",
        "gated_splice_nmax8",
    ),
    (
        "hand splice, ungated (N≤20)",
        "by hand, closed form",
        RUNS / "rescore_hardened.json",
        "ungated_splice_nmax20",
    ),
    (
        "LoRA + native replay, 1k",
        "gradients",
        RUNS / "rescore_hardened.json",
        "lora_replay_native_b1000",
    ),
    (
        "LoRA + native replay, 3k",
        "gradients",
        RUNS / "rescore_hardened.json",
        "lora_replay_native_b3000",
    ),
    (
        "LoRA + native replay, 10k",
        "gradients",
        RUNS / "rescore_hardened.json",
        "lora_replay_native_b10000",
    ),
    (
        "LoRA + text replay, 1k",
        "gradients",
        RUNS / "rescore_hardened.json",
        "lora_replay_text_b1000",
    ),
    (
        "LoRA + text replay, 3k",
        "gradients",
        RUNS / "rescore_hardened.json",
        "lora_replay_text_b3000",
    ),
    (
        "LoRA + text replay, 10k",
        "gradients",
        RUNS / "rescore_hardened.json",
        "lora_replay_text_b10000",
    ),
    (
        "pilot1b (add_capacity)",
        "agent, closed form",
        RUNS / "rescore_hardened.json",
        "pilot1b_edit16",
    ),
    ("pilot4c (add_capacity)", "agent, closed form", RUNS / "pilot4c/result.json", None),
    ("pilot5 (edit_only)", "agent, closed form", RUNS / "pilot5/result.json", None),
    ("pilot6 (add_capacity)", "agent, closed form", RUNS / "pilot6/result.json", None),
    ("pilot7 (edit_only)", "agent, closed form", RUNS / "pilot7/result.json", None),
    ("pilot8 (add_capacity)", "agent, closed form", RUNS / "pilot8/result.json", None),
    ("pilot9 (edit_only)", "agent, closed form", RUNS / "pilot9/result.json", None),
]


def load(path: Path, key: str | None) -> dict:
    data = json.loads(path.read_text())
    return data[key] if key else data["hidden_eval"]


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def row(label: str, how: str, h: dict) -> str:
    rb = h["robustness"]
    add = h["addition_by_n"]
    n1_8 = sum(add[str(n)] for n in range(1, 9)) / 8
    n9_20 = sum(add[str(n)] for n in range(9, 21)) / 12
    native = min(h["native"].values())
    f = rb["f_added_params"]
    added = f"{f['nonzero']:,} / {f['allocated']:,}" if f["allocated"] else "0"
    cells = [
        label,
        how,
        pct(n1_8),
        pct(n9_20),
        f"{h['text_kl']:.4f}",
        pct(native),
        f"{rb['a_prefixed_text']['kl']:.3f}",
        pct(rb["b_termination"]["terminated"]["mean"]),
        pct(rb["c_not_first_line"]["min_mean"]),
        pct(rb["d_length_n21_30"]["mean"]),
        f"{rb['e_native_weight_change']['total_rel']:.3g}",
        added,
        "yes" if h["meets_primary_bar"] else "no",
    ]
    return "| " + " | ".join(cells) + " |"


def main() -> None:
    header = [
        "Model",
        "Weights written",
        "Add N1–8",
        "Add N9–20",
        "Text KL",
        "Native (min)",
        "(a) `A:` KL",
        "(b) stops",
        "(c) not 1st line",
        "(d) N21–30",
        "(e) native Δ (rel)",
        "(f) added params (nonzero / allocated)",
        "Primary bar",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += [row(label, how, load(path, key)) for label, how, path, key in ROWS]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
