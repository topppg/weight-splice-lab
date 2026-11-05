import random
import time
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from weight_splice_lab.toy.data import PAD, TOKENIZER, Example
from weight_splice_lab.toy.model import ToyLM

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        self.a = nn.Parameter(
            torch.randn(rank, base.in_features, device=base.weight.device) / base.in_features**0.5
        )
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        self.scale = alpha / rank

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x) + (x @ self.a.T @ self.b.T) * self.scale

    @property
    def weight(self) -> Tensor:
        return self.base.weight


def apply_lora(model: ToyLM, rank: int, alpha: float) -> list[nn.Parameter]:
    for p in model.parameters():
        p.requires_grad_(False)
    params: list[nn.Parameter] = []
    for block in model.blocks:
        for parent in (block.attn, block.mlp):
            for name in TARGETS:
                if hasattr(parent, name):
                    wrapped = LoRALinear(getattr(parent, name), rank, alpha)
                    setattr(parent, name, wrapped)
                    params += [wrapped.a, wrapped.b]
    return params


@torch.no_grad()
def merge_lora(model: ToyLM) -> None:
    for block in model.blocks:
        for parent in (block.attn, block.mlp):
            for name in TARGETS:
                mod = getattr(parent, name, None)
                if isinstance(mod, LoRALinear):
                    mod.base.weight += (mod.b @ mod.a) * mod.scale
                    setattr(parent, name, mod.base)


def lora_param_count(params: list[nn.Parameter]) -> int:
    return sum(p.numel() for p in params)


def encode_batch(examples: list[Example], device: str) -> tuple[Tensor, Tensor]:
    seqs = [TOKENIZER.encode(ex.prompt + ex.answer + "\n") for ex in examples]
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), PAD, dtype=torch.long)
    mask = torch.zeros(len(seqs), width - 1, dtype=torch.bool)
    for i, (s, ex) in enumerate(zip(seqs, examples, strict=True)):
        ids[i, : len(s)] = torch.tensor(s)
        start = len(ex.prompt)
        mask[i, start : len(s) - 1] = True
    return ids.to(device), mask.to(device)


def finetune(
    model: ToyLM,
    params: list[nn.Parameter],
    data: list[Example],
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: str,
    stream: Callable[[random.Random], Example] | None = None,
    stream_per_batch: int = 0,
) -> list[dict[str, float]]:
    rng = random.Random(seed)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    history: list[dict[str, float]] = []
    start = time.time()
    model.train()
    for step in range(steps):
        for group in opt.param_groups:
            group["lr"] = lr * min(1.0, (step + 1) / 100) * (1 - step / steps)
        n_stream = stream_per_batch if stream is not None else 0
        batch = [rng.choice(data) for _ in range(batch_size - n_stream)]
        if stream is not None:
            batch += [stream(rng) for _ in range(n_stream)]
        ids, mask = encode_batch(batch, device)
        logits = model(ids[:, :-1])
        loss = F.cross_entropy(logits[mask], ids[:, 1:][mask])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == steps - 1:
            history.append({"step": step, "loss": loss.item(), "elapsed": time.time() - start})
    model.eval()
    return history
