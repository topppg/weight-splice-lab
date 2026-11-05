import torch

from weight_splice_lab.toy.data import TOKENIZER, MixtureSampler
from weight_splice_lab.toy.model import ToyConfig, ToyLM, apply_rope, count_params, rope_cos_sin
from weight_splice_lab.toy.splice import (
    SpliceSpec,
    build_splice,
    collect_native,
    fit_probe,
    pos_key,
    rope_query,
)


def test_positional_filter_margin_over_window() -> None:
    hd, base = 32, 10000.0
    key = pos_key(hd)
    for offset in (0, 5, 21, 41):
        q = rope_query(key, offset, hd, base)
        cos, sin = rope_cos_sin(torch.arange(120), hd, base)
        keys = apply_rope(key.expand(120, hd), cos, sin)
        qs = apply_rope(q.expand(120, hd), cos, sin)
        scores = keys[:120] @ qs[119]
        target = 119 - offset
        others = torch.cat([scores[:target], scores[target + 1 :]])
        assert scores.argmax().item() == target
        assert (scores[target] - others.max()) / scores[target] > 0.15


def test_probe_is_orthogonal_to_private_basis() -> None:
    torch.manual_seed(0)
    z = torch.randn(500, 16)
    private = torch.linalg.qr(torch.randn(16, 3))[0]
    w = fit_probe(z, torch.randn(500, 2), private)
    assert torch.allclose(private.T @ w, torch.zeros(3, 2), atol=1e-5)


def test_build_copy_splice_on_random_model_adds_capacity_only() -> None:
    torch.manual_seed(0)
    cfg = ToyConfig(
        vocab_size=TOKENIZER.vocab_size, d_model=160, n_layers=6, n_heads=2, head_dim=32, d_mlp=64
    )
    model = ToyLM(cfg).eval()
    text = TOKENIZER.encode_array(
        "plain text with a few words and 12 numbers + signs = here. " * 300
    )
    acts = collect_native(model, MixtureSampler(text, 64, 0.5, 0), n_seqs=16, batch_size=8)
    n_before = count_params(model)
    report = build_splice(model, SpliceSpec("copy", n_max=3), acts)
    # L0: 3 length detectors + marker head; L1: broadcast; L2: one read head (a); L5: emit head.
    assert [b.attn.n_heads for b in model.blocks] == [2 + 4, 2 + 1, 2 + 1, 2, 2, 2 + 1]
    # L1: 2 step neurons per length + 2 per text-energy direction (all directions orthogonal to the
    # private basis); L2: 2 per cleaned one-hot (a: 11) + one~; L4: 10 lookups.
    n_text = 2 * (cfg.d_model - len(report.layout))
    added = [b.mlp.gate_proj.out_features - 64 for b in model.blocks]
    assert added == [0, 6 + n_text, 24, 0, 10, 0]
    assert count_params(model) - n_before == report.params_allocated
    assert 0 < report.params_nonzero <= report.params_allocated
    assert any("pseudo-inverse" in op for op in report.ops)
    assert torch.isfinite(model(torch.tensor([TOKENIZER.encode("A:123+456=")]))).all()
