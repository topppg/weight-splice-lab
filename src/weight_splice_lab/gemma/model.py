import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

KVCache = list[tuple[Tensor, Tensor]]


@dataclass
class GemmaConfig:
    vocab_size: int = 262144
    d_model: int = 640
    n_layers: int = 18
    n_heads: int = 4
    n_kv_heads: int = 1
    head_dim: int = 256
    d_mlp: int = 2048
    query_pre_attn_scalar: float = 256.0
    rope_local_base: float = 10000.0
    rope_global_base: float = 1000000.0
    sliding_window: int = 512
    layer_types: list[str] = field(default_factory=lambda: (["sliding"] * 5 + ["full"]) * 3)
    norm_eps: float = 1e-6


class RMSNorm(nn.Module):
    # Gemma convention: the stored weight w scales by (1 + w)
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * (1.0 + self.weight)


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
    # Grouped-query attention: native query heads share the native key/value heads; every added
    # head (add_heads) gets its own key/value head. q_norm and k_norm are shared by all heads.
    def __init__(self, cfg: GemmaConfig, sliding: bool) -> None:
        super().__init__()
        d, hd = cfg.d_model, cfg.head_dim
        self.n_heads = cfg.n_heads
        self.n_native_heads = cfg.n_heads
        self.n_native_kv = cfg.n_kv_heads
        self.head_dim = hd
        self.scale = cfg.query_pre_attn_scalar**-0.5
        self.window = cfg.sliding_window if sliding else None
        self.rope_base = cfg.rope_local_base if sliding else cfg.rope_global_base
        self.q_proj = nn.Linear(d, cfg.n_heads * hd, bias=False)
        self.k_proj = nn.Linear(d, cfg.n_kv_heads * hd, bias=False)
        self.v_proj = nn.Linear(d, cfg.n_kv_heads * hd, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * hd, d, bias=False)
        self.q_norm = RMSNorm(hd, cfg.norm_eps)
        self.k_norm = RMSNorm(hd, cfg.norm_eps)

    def kv_index(self) -> list[int]:
        group = self.n_native_heads // self.n_native_kv
        native = [h // group for h in range(self.n_native_heads)]
        return native + list(range(self.n_native_kv, self.n_native_kv + self.n_added()))

    def n_added(self) -> int:
        return self.n_heads - self.n_native_heads

    def mask(self, t_q: int, t_k: int, device: torch.device) -> Tensor:
        q = torch.arange(t_k - t_q, t_k, device=device)[:, None]
        k = torch.arange(t_k, device=device)[None, :]
        allowed = k <= q
        if self.window is not None:
            allowed &= q - k < self.window
        return allowed

    def forward(
        self,
        x: Tensor,
        start: int,
        capture: dict[str, Tensor] | None,
        prefix: str,
        cache: KVCache | None,
        layer: int,
    ) -> Tensor:
        b, t, _ = x.shape
        hd = self.head_dim
        positions = torch.arange(start, start + t, device=x.device)
        cos, sin = rope_cos_sin(positions, hd, self.rope_base)
        q = self.q_norm(self.q_proj(x).view(b, t, self.n_heads, hd)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(b, t, -1, hd)).transpose(1, 2)
        v = self.v_proj(x).view(b, t, -1, hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            if len(cache) > layer:
                k = torch.cat([cache[layer][0], k], dim=2)
                v = torch.cat([cache[layer][1], v], dim=2)
                cache[layer] = (k, v)
            else:
                cache.append((k, v))
        idx = torch.tensor(self.kv_index(), device=x.device)
        k, v = k[:, idx], v[:, idx]
        allowed = self.mask(t, k.shape[2], x.device)
        if capture is None:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, scale=self.scale)
        else:
            scores = (q @ k.transpose(-1, -2)) * self.scale
            pattern = scores.masked_fill(~allowed, float("-inf")).softmax(-1)
            capture[f"{prefix}.pattern"] = pattern
            out = pattern @ v
            capture[f"{prefix}.head_out"] = out
        return self.o_proj(out.transpose(1, 2).reshape(b, t, -1))


class GeGLU(nn.Module):
    def __init__(self, d_model: int, d_mlp: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_mlp, bias=False)
        self.up_proj = nn.Linear(d_model, d_mlp, bias=False)
        self.down_proj = nn.Linear(d_mlp, d_model, bias=False)

    def forward(self, x: Tensor, capture: dict[str, Tensor] | None, prefix: str) -> Tensor:
        h = F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x)
        if capture is not None:
            capture[f"{prefix}.hidden"] = h
        return self.down_proj(h)


class Block(nn.Module):
    def __init__(self, cfg: GemmaConfig, sliding: bool) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg, sliding)
        self.post_attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = GeGLU(cfg.d_model, cfg.d_mlp)
        self.post_mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)

    def forward(
        self,
        x: Tensor,
        start: int,
        capture: dict[str, Tensor] | None,
        i: int,
        cache: KVCache | None,
    ) -> Tensor:
        if capture is not None:
            capture[f"resid_pre.{i}"] = x
        x = x + self.post_attn_norm(
            self.attn(self.attn_norm(x), start, capture, f"attn.{i}", cache, i)
        )
        if capture is not None:
            capture[f"resid_mid.{i}"] = x
        return x + self.post_mlp_norm(self.mlp(self.mlp_norm(x), capture, f"mlp.{i}"))


class GemmaLM(nn.Module):
    def __init__(self, cfg: GemmaConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, lt == "sliding") for lt in cfg.layer_types])
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)

    def hidden(
        self,
        idx: Tensor,
        capture: dict[str, Tensor] | None = None,
        cache: KVCache | None = None,
        start: int = 0,
    ) -> Tensor:
        x = self.embed(idx) * math.sqrt(self.cfg.d_model)
        for i, block in enumerate(self.blocks):
            x = block(x, start, capture, i, cache)
        if capture is not None:
            capture["resid_final"] = x
        return self.final_norm(x)

    def forward(self, idx: Tensor, capture: dict[str, Tensor] | None = None) -> Tensor:
        return self.hidden(idx, capture) @ self.embed.weight.T

    @torch.inference_mode()
    def generate(self, idx: Tensor, n_new: int) -> Tensor:
        # greedy with a key/value cache; logits only at the last position
        cache: KVCache = []
        h = self.hidden(idx, cache=cache)
        out = [idx]
        for step in range(n_new):
            nxt = (h[:, -1] @ self.embed.weight.T).argmax(-1, keepdim=True)
            out.append(nxt)
            if step + 1 < n_new:
                h = self.hidden(nxt, cache=cache, start=idx.shape[1] + step)
        return torch.cat(out, dim=1)


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


def add_heads(model: GemmaLM, layer: int, n: int) -> list[int]:
    # each new head gets new query rows, its own key/value head and new o_proj columns, all zero
    attn = model.blocks[layer].attn
    extra = n * attn.head_dim
    attn.q_proj = _grow_rows(attn.q_proj, extra)
    attn.k_proj = _grow_rows(attn.k_proj, extra)
    attn.v_proj = _grow_rows(attn.v_proj, extra)
    attn.o_proj = _grow_cols(attn.o_proj, extra)
    first = attn.n_heads
    attn.n_heads += n
    return list(range(first, first + n))


def add_neurons(model: GemmaLM, layer: int, n: int) -> list[int]:
    mlp = model.blocks[layer].mlp
    first = mlp.gate_proj.out_features
    mlp.gate_proj = _grow_rows(mlp.gate_proj, n)
    mlp.up_proj = _grow_rows(mlp.up_proj, n)
    mlp.down_proj = _grow_cols(mlp.down_proj, n)
    return list(range(first, first + n))


def architecture(model: GemmaLM) -> dict[str, list[int]]:
    return {
        "n_heads": [b.attn.n_heads for b in model.blocks],
        "d_mlp": [b.mlp.gate_proj.out_features for b in model.blocks],
    }


def save_model(model: GemmaLM, path: Path, dtype: torch.dtype = torch.bfloat16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.to(dtype) for k, v in model.state_dict().items()}
    torch.save({"cfg": asdict(model.cfg), "arch": architecture(model), "state": state}, path)


def load_model(path: Path, device: str = "cpu") -> GemmaLM:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cfg = GemmaConfig(**ckpt["cfg"])
    model = GemmaLM(cfg)
    for i, (h, m) in enumerate(zip(ckpt["arch"]["n_heads"], ckpt["arch"]["d_mlp"], strict=True)):
        if h > cfg.n_heads:
            add_heads(model, i, h - cfg.n_heads)
        if m > cfg.d_mlp:
            add_neurons(model, i, m - cfg.d_mlp)
    model.load_state_dict({k: v.float() for k, v in ckpt["state"].items()})
    return model.to(device).eval()


HF_NAMES = {
    "input_layernorm": "attn_norm",
    "post_attention_layernorm": "post_attn_norm",
    "pre_feedforward_layernorm": "mlp_norm",
    "post_feedforward_layernorm": "post_mlp_norm",
    "self_attn": "attn",
}


def from_hf(hf_dir: Path) -> GemmaLM:
    from safetensors.torch import load_file

    hf = json.loads((hf_dir / "config.json").read_text())
    cfg = GemmaConfig(
        vocab_size=hf["vocab_size"],
        d_model=hf["hidden_size"],
        n_layers=hf["num_hidden_layers"],
        n_heads=hf["num_attention_heads"],
        n_kv_heads=hf["num_key_value_heads"],
        head_dim=hf["head_dim"],
        d_mlp=hf["intermediate_size"],
        query_pre_attn_scalar=hf["query_pre_attn_scalar"],
        rope_local_base=hf["rope_local_base_freq"],
        rope_global_base=hf["rope_theta"],
        sliding_window=hf["sliding_window"],
        layer_types=["sliding" if t == "sliding_attention" else "full" for t in hf["layer_types"]],
        norm_eps=hf["rms_norm_eps"],
    )
    assert hf.get("final_logit_softcapping") is None and hf.get("attn_logit_softcapping") is None
    assert hf.get("rope_scaling") is None and hf["hidden_activation"] == "gelu_pytorch_tanh"
    state: dict[str, Tensor] = {}
    for name, t in load_file(hf_dir / "model.safetensors").items():
        name = name.removeprefix("model.")
        if name == "embed_tokens.weight":
            state["embed.weight"] = t
        elif name == "norm.weight":
            state["final_norm.weight"] = t
        else:
            parts = name.split(".")
            parts[0] = "blocks"
            parts = [HF_NAMES.get(p, p) for p in parts]
            state[".".join(parts)] = t
    model = GemmaLM(cfg)
    model.load_state_dict({k: v.float() for k, v in state.items()})
    return model.eval()


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
