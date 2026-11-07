import copy

import torch

from weight_splice_lab.gemma.model import GemmaConfig, GemmaLM, add_heads, add_neurons


def tiny() -> GemmaLM:
    torch.manual_seed(0)
    cfg = GemmaConfig(
        vocab_size=300, d_model=32, n_layers=2, n_heads=2, n_kv_heads=1, head_dim=16, d_mlp=64,
        query_pre_attn_scalar=16.0, sliding_window=8, layer_types=["sliding", "full"],
    )  # fmt: skip
    return GemmaLM(cfg).eval()


def test_zero_added_units_leave_logits_unchanged() -> None:
    base = tiny()
    model = copy.deepcopy(base)
    add_heads(model, 0, 2)
    add_neurons(model, 1, 5)
    x = torch.randint(3, 300, (2, 20))
    with torch.no_grad():
        torch.testing.assert_close(model(x), base(x), atol=1e-5, rtol=1e-5)


def test_cached_generate_matches_full_forward_argmax() -> None:
    # past the sliding window, with a live added head that has its own key/value head
    model = tiny()
    add_heads(model, 0, 1)
    add_neurons(model, 1, 3)
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() == 2 and p.shape[0] != 300:
                p.add_(torch.randn_like(p) * 0.02)
    prompt = torch.randint(3, 300, (2, 6))
    out = model.generate(prompt, 10)
    ref = prompt
    with torch.no_grad():
        for _ in range(10):
            ref = torch.cat([ref, model(ref)[:, -1].argmax(-1, keepdim=True)], dim=1)
    assert torch.equal(out, ref)
