from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class ToyConfig:
    vocab_size: int
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    head_dim: int = 32
    d_mlp: int = 768
    rope_base: float = 10000.0
    max_seq_len: int = 512
    norm_eps: float = 1e-6


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def rope_cos_sin(positions: Tensor, head_dim: int, base: float) -> tuple[Tensor, Tensor]:
    inv_freq = base ** (
        -torch.arange(0, head_dim, 2, dtype=torch.float32, device=positions.device) / head_dim
    )
    angles = positions.float()[:, None] * inv_freq[None, :]
    angles = torch.cat([angles, angles], dim=-1)
    return angles.cos(), angles.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * cos + rotated * sin


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, head_dim: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)

    def forward(
        self, x: Tensor, cos: Tensor, sin: Tensor, capture: dict[str, Tensor] | None, prefix: str
    ) -> Tensor:
        b, t, _ = x.shape
        shape = (b, t, self.n_heads, self.head_dim)
        q = apply_rope(self.q_proj(x).view(shape).transpose(1, 2), cos, sin)
        k = apply_rope(self.k_proj(x).view(shape).transpose(1, 2), cos, sin)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        if capture is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            scores = q @ k.transpose(-1, -2) / self.head_dim**0.5
            mask = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
            pattern = scores.masked_fill(~mask, float("-inf")).softmax(-1)
            capture[f"{prefix}.pattern"] = pattern
            out = pattern @ v
            capture[f"{prefix}.head_out"] = out
        return self.o_proj(out.transpose(1, 2).reshape(b, t, -1))


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_mlp: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_mlp, bias=False)
        self.up_proj = nn.Linear(d_model, d_mlp, bias=False)
        self.down_proj = nn.Linear(d_mlp, d_model, bias=False)

    def forward(self, x: Tensor, capture: dict[str, Tensor] | None, prefix: str) -> Tensor:
        h = F.silu(self.gate_proj(x)) * self.up_proj(x)
        if capture is not None:
            capture[f"{prefix}.hidden"] = h
        return self.down_proj(h)


class Block(nn.Module):
    def __init__(self, cfg: ToyConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg.d_model, cfg.n_heads, cfg.head_dim)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg.d_model, cfg.d_mlp)

    def forward(
        self, x: Tensor, cos: Tensor, sin: Tensor, capture: dict[str, Tensor] | None, i: int
    ) -> Tensor:
        if capture is not None:
            capture[f"resid_pre.{i}"] = x
        x = x + self.attn(self.attn_norm(x), cos, sin, capture, f"attn.{i}")
        if capture is not None:
            capture[f"resid_mid.{i}"] = x
        return x + self.mlp(self.mlp_norm(x), capture, f"mlp.{i}")


class ToyLM(nn.Module):
    def __init__(self, cfg: ToyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx: Tensor, capture: dict[str, Tensor] | None = None) -> Tensor:
        positions = torch.arange(idx.shape[1], device=idx.device)
        cos, sin = rope_cos_sin(positions, self.cfg.head_dim, self.cfg.rope_base)
        x = self.embed(idx)
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, capture, i)
        if capture is not None:
            capture["resid_final"] = x
        return self.final_norm(x) @ self.embed.weight.T

    @torch.inference_mode()
    def generate(self, idx: Tensor, n_new: int) -> Tensor:
        for _ in range(n_new):
            nxt = self(idx)[:, -1].argmax(-1, keepdim=True)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


def _grow_rows(linear: nn.Linear, n: int) -> nn.Linear:
    new = nn.Linear(linear.in_features, linear.out_features + n, bias=False)
    new = new.to(device=linear.weight.device, dtype=linear.weight.dtype)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[: linear.out_features] = linear.weight
    return new


def _grow_cols(linear: nn.Linear, n: int) -> nn.Linear:
    new = nn.Linear(linear.in_features + n, linear.out_features, bias=False)
    new = new.to(device=linear.weight.device, dtype=linear.weight.dtype)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, : linear.in_features] = linear.weight
    return new


def add_heads(model: ToyLM, layer: int, n: int) -> list[int]:
    attn = model.blocks[layer].attn
    extra = n * attn.head_dim
    attn.q_proj = _grow_rows(attn.q_proj, extra)
    attn.k_proj = _grow_rows(attn.k_proj, extra)
    attn.v_proj = _grow_rows(attn.v_proj, extra)
    attn.o_proj = _grow_cols(attn.o_proj, extra)
    first = attn.n_heads
    attn.n_heads += n
    return list(range(first, first + n))


def add_neurons(model: ToyLM, layer: int, n: int) -> list[int]:
    mlp = model.blocks[layer].mlp
    first = mlp.gate_proj.out_features
    mlp.gate_proj = _grow_rows(mlp.gate_proj, n)
    mlp.up_proj = _grow_rows(mlp.up_proj, n)
    mlp.down_proj = _grow_cols(mlp.down_proj, n)
    return list(range(first, first + n))


def insert_layer(model: ToyLM, index: int) -> None:
    ref = model.blocks[0].attn.q_proj.weight
    block = Block(model.cfg).to(device=ref.device, dtype=ref.dtype)
    with torch.no_grad():
        for p in block.parameters():
            if p.ndim == 2:
                p.zero_()
    model.blocks.insert(index, block)


def head_slice(model: ToyLM, layer: int, head: int) -> slice:
    hd = model.blocks[layer].attn.head_dim
    return slice(head * hd, (head + 1) * hd)


def architecture(model: ToyLM) -> dict[str, list[int]]:
    return {
        "n_heads": [b.attn.n_heads for b in model.blocks],
        "d_mlp": [b.mlp.gate_proj.out_features for b in model.blocks],
    }


def save_model(model: ToyLM, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"cfg": asdict(model.cfg), "arch": architecture(model), "state": model.state_dict()}, path
    )


def load_model(path: Path, device: str = "cpu") -> ToyLM:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    cfg = ToyConfig(**ckpt["cfg"])
    arch = ckpt["arch"]
    model = ToyLM(cfg)
    while len(model.blocks) < len(arch["n_heads"]):
        model.blocks.append(Block(cfg))
    for i, (h, m) in enumerate(zip(arch["n_heads"], arch["d_mlp"], strict=True)):
        if h > cfg.n_heads:
            add_heads(model, i, h - cfg.n_heads)
        if m > cfg.d_mlp:
            add_neurons(model, i, m - cfg.d_mlp)
    model.load_state_dict(ckpt["state"])
    return model.to(device)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
