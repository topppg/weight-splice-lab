import random

import torch
from torch import nn

from weight_splice_lab.toy.data import TOKENIZER, Example
from weight_splice_lab.toy.lora import LoRALinear, apply_lora, finetune, merge_lora
from weight_splice_lab.toy.model import ToyConfig, ToyLM


def test_merge_lora_matches_wrapped_model() -> None:
    torch.manual_seed(0)
    cfg = ToyConfig(
        vocab_size=TOKENIZER.vocab_size, d_model=64, n_layers=2, n_heads=2, head_dim=16, d_mlp=96
    )
    model = ToyLM(cfg).eval()
    params = apply_lora(model, rank=4, alpha=8.0)
    with torch.no_grad():
        for p in params:
            p.normal_(std=0.05)
    ids = torch.tensor([TOKENIZER.encode("A:12+34=")])
    with torch.no_grad():
        wrapped = model(ids)
        merge_lora(model)
        merged = model(ids)
    assert not any(isinstance(m, LoRALinear) for m in model.modules())
    assert isinstance(model.blocks[0].attn.q_proj, nn.Linear)
    assert torch.allclose(wrapped, merged, atol=1e-5)


def test_finetune_mixes_fresh_stream_examples() -> None:
    torch.manual_seed(0)
    cfg = ToyConfig(
        vocab_size=TOKENIZER.vocab_size, d_model=32, n_layers=1, n_heads=2, head_dim=16, d_mlp=48
    )
    model = ToyLM(cfg)
    params = apply_lora(model, rank=2, alpha=4.0)
    calls: list[int] = []

    def stream(rng: random.Random) -> Example:
        calls.append(1)
        return Example("", "some text " + str(rng.randint(0, 9)))

    finetune(model, params, [Example("A:1+2=", "30")], 3, 8, 1e-3, 0, "cpu", stream, 3)
    assert len(calls) == 9
