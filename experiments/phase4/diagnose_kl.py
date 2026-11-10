import argparse
import json
from pathlib import Path

import numpy as np
import torch

from weight_splice_lab.gemma import evaluate as E
from weight_splice_lab.gemma.model import load_model
from weight_splice_lab.gemma.task import BASE_CKPT
from weight_splice_lab.harness.edit import apply_edit, load_edit

# Where does an edit's hidden text KL come from? Recomputes the per-position KL on the hidden
# Wikipedia chunks of the eval (same seed and chunking) and reports the worst chunks, where the
# drift starts, and how much of the mean the worst chunks explain.

SPIKE = 0.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=BASE_CKPT)
    ap.add_argument("--edit", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    base = load_model(a.base, device="cpu")
    model = apply_edit(base, load_edit(a.edit)).eval()
    model.embed = base.embed
    tok = E.tokenizer()
    items = E.random_chunks(E.wikitext_ids(tok, "validation"), E.N_TEXT_CHUNKS, E.TEXT_SEED)
    means, chunks = [], []
    with torch.inference_mode():
        for ci, (ids, _) in enumerate(items):
            x = torch.tensor([ids[:-1]])
            lp = (model.hidden(x)[0] @ base.embed.weight.T).log_softmax(-1)
            lq = (base.hidden(x)[0] @ base.embed.weight.T).log_softmax(-1)
            kl = (lq.exp() * (lq - lp)).sum(-1).numpy()
            means.append(float(kl.mean()))
            if kl.max() > SPIKE:
                onset = int(np.argmax(kl > SPIKE))
                chunks.append({
                    "chunk": ci, "mean_kl": float(kl.mean()), "max_kl": float(kl.max()),
                    "positions_over_spike": int((kl > SPIKE).sum()), "onset": onset,
                    "context": tok.decode(ids[max(1, onset - 20) : onset + 1]),
                    "edited_next": tok.decode(lp[onset : onset + 8].argmax(-1).tolist()),
                    "base_next": tok.decode(lq[onset : onset + 8].argmax(-1).tolist()),
                })  # fmt: skip
    m = np.array(means)
    order = np.argsort(-m)
    top5 = float(m[order[:5]].sum() / m.sum())
    out = {
        "edit": a.edit.name, "chunks": len(m), "mean_kl": float(m.mean()),
        "share_of_kl_in_worst_5": top5, "mean_kl_without_worst_5": float(m[order[5:]].mean()),
        "spike_threshold": SPIKE, "chunks_with_spikes": sorted(chunks, key=lambda c: -c["mean_kl"]),
    }  # fmt: skip
    a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items() if k != "chunks_with_spikes"}))


if __name__ == "__main__":
    main()
