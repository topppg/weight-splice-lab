import math
import resource
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from weight_splice_lab.toy.data import MixtureSampler
from weight_splice_lab.toy.model import ToyLM


@dataclass
class TrainConfig:
    steps: int = 12000
    batch_size: int = 64
    micro_batch: int = 16
    seq_len: int = 256
    lr: float = 2e-3
    min_lr: float = 1e-4
    warmup: int = 300
    weight_decay: float = 0.1
    text_frac: float = 0.6
    seed: int = 0
    log_every: int = 250


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def mps_gb() -> float:
    return torch.mps.driver_allocated_memory() / 1e9 if torch.backends.mps.is_available() else 0.0


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    progress = (step - cfg.warmup) / max(cfg.steps - cfg.warmup, 1)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))


def train_base(
    model: ToyLM, sampler: MixtureSampler, cfg: TrainConfig, device: str
) -> list[dict[str, float]]:
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=(0.9, 0.95),
    )
    history: list[dict[str, float]] = []
    start = time.time()
    model.train()
    for step in range(cfg.steps):
        for group in opt.param_groups:
            group["lr"] = lr_at(step, cfg)
        batch = sampler.batch(cfg.batch_size).to(device)
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for micro in batch.split(cfg.micro_batch):
            logits = model(micro[:, :-1])
            loss = (
                F.cross_entropy(logits.flatten(0, 1), micro[:, 1:].flatten())
                * len(micro)
                / len(batch)
            )
            loss.backward()
            total += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            row = {
                "step": step,
                "loss": total,
                "lr": lr_at(step, cfg),
                "elapsed": time.time() - start,
                "mps_gb": mps_gb(),
            }
            history.append(row)
            print(
                f"step {step:6d} loss {row['loss']:.4f} lr {row['lr']:.2e} "
                f"t {row['elapsed']:.0f}s mps {row['mps_gb']:.2f}GB",
                flush=True,
            )
    model.eval()
    return history
