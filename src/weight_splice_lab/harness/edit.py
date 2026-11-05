import copy
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import torch
from torch import Tensor, nn

REFUSED = ("embed.weight", "final_norm.weight")
CONDITIONS = ("add_capacity", "edit_only")
# an edit is "a whole-tensor constant multiple" if this share of native entries is c * original
SCALAR_MULTIPLE_FRACTION = 0.99


class EditError(ValueError):
    pass


@dataclass
class Edit:
    arch: dict[str, list[int]]
    tensors: dict[str, Tensor] = field(default_factory=dict)


def _module(base: nn.Module) -> ModuleType:
    # the model's own module (toy.model or gemma.model) provides architecture/add_heads/add_neurons
    return sys.modules[type(base).__module__]


def architecture(base: nn.Module) -> dict[str, list[int]]:
    return _module(base).architecture(base)


def target_shapes(base: nn.Module, arch: dict[str, list[int]]) -> dict[str, tuple[int, ...]]:
    # q/o grow with heads; k/v grow by one key/value head per added head (in the toy every head
    # has its own, so k/v rows equal q rows); gate/up/down grow with the MLP width
    ref = architecture(base)
    shapes: dict[str, tuple[int, ...]] = {}
    for name, t in base.state_dict().items():
        shape = tuple(t.shape)
        parts = name.split(".")
        if parts[0] == "blocks":
            layer, rest = int(parts[1]), ".".join(parts[2:])
            hd = base.blocks[layer].attn.head_dim
            added = arch["n_heads"][layer] - ref["n_heads"][layer]
            m = arch["d_mlp"][layer]
            if rest == "attn.q_proj.weight":
                shape = (arch["n_heads"][layer] * hd, shape[1])
            elif rest in ("attn.k_proj.weight", "attn.v_proj.weight"):
                shape = (shape[0] + added * hd, shape[1])
            elif rest == "attn.o_proj.weight":
                shape = (shape[0], arch["n_heads"][layer] * hd)
            elif rest in ("mlp.gate_proj.weight", "mlp.up_proj.weight"):
                shape = (m, shape[1])
            elif rest == "mlp.down_proj.weight":
                shape = (shape[0], m)
        shapes[name] = shape
    return shapes


def added_params(base: nn.Module, arch: dict[str, list[int]]) -> int:
    native = {k: math.prod(v.shape) for k, v in base.state_dict().items()}
    return sum(math.prod(s) - native[k] for k, s in target_shapes(base, arch).items())


def with_arch(base: nn.Module, arch: dict[str, list[int]]) -> nn.Module:
    # shares the embedding with the base: it can never be edited (Gemma: 63% of the weights)
    emb = base.embed.weight
    model = copy.deepcopy(base, memo={id(emb): emb})
    mod, ref = _module(base), architecture(base)
    for layer in range(len(base.blocks)):
        if (n := arch["n_heads"][layer] - ref["n_heads"][layer]) > 0:
            mod.add_heads(model, layer, n)
        if (n := arch["d_mlp"][layer] - ref["d_mlp"][layer]) > 0:
            mod.add_neurons(model, layer, n)
    return model


FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
MAX_STAGED_ADDED = 50_000_000
FILE_OVERHEAD = 1 << 16


def _arch_structure(base: nn.Module, arch: object) -> dict[str, list[int]]:
    ref = architecture(base)
    if not isinstance(arch, dict) or set(arch) != {"n_heads", "d_mlp"}:
        raise EditError("arch must have exactly the keys n_heads and d_mlp")
    for key in ("n_heads", "d_mlp"):
        vals = arch[key]
        if not isinstance(vals, list) or len(vals) != len(ref[key]):
            raise EditError(f"arch[{key}] must list one value per layer")
        if any(type(v) is not int or v < r for v, r in zip(vals, ref[key], strict=True)):
            raise EditError(f"arch[{key}] values must be integers >= the base values {ref[key]}")
    if added_params(base, arch) > MAX_STAGED_ADDED:
        raise EditError(f"arch adds more than {MAX_STAGED_ADDED:,} parameters")
    return arch


def _regular_file(p: Path) -> None:
    if p.is_symlink() or not p.is_file():
        raise EditError(f"{p.name} must be a regular file, not a link or directory")
    if p.stat().st_nlink > 1:
        raise EditError(f"{p.name} has hard links")


def _load_tensor(p: Path, shape: tuple[int, ...]) -> Tensor:
    numel = math.prod(shape)
    _regular_file(p)
    if p.stat().st_size > numel * 8 + FILE_OVERHEAD:
        raise EditError(f"{p.name}: file is larger than a {shape} tensor can be")
    try:
        t = torch.load(p, map_location="cpu", weights_only=True, mmap=True)
    except Exception as e:
        raise EditError(f"{p.name} is not a readable tensor file ({type(e).__name__}: {e})") from e
    if type(t) not in (Tensor, torch.nn.Parameter):
        raise EditError(f"{p.name} does not hold a plain tensor (got {type(t).__name__})")
    if t.layout != torch.strided or t.is_quantized or t.device.type != "cpu":
        raise EditError(f"{p.name}: only dense CPU tensors are accepted")
    if t.dtype not in FLOAT_DTYPES:
        raise EditError(f"{p.name}: dtype {t.dtype} is not a float dtype")
    if tuple(t.shape) != shape or t.numel() != numel:
        raise EditError(f"{p.name}: shape {tuple(t.shape)} != {shape} for this arch")
    nbytes = t.untyped_storage().nbytes()
    if not t.is_contiguous() or t.storage_offset() != 0 or nbytes != numel * t.element_size():
        raise EditError(f"{p.name}: the tensor must own a contiguous storage of exactly its size")
    return t.to(torch.float32, copy=True)


def read_staged(stage: Path, base: nn.Module) -> Edit:
    arch_file = stage / "arch.json"
    if arch_file.exists() or arch_file.is_symlink():
        _regular_file(arch_file)
        if arch_file.stat().st_size > FILE_OVERHEAD:
            raise EditError("arch.json is too large")
        try:
            raw = json.loads(arch_file.read_text())
        except (ValueError, UnicodeDecodeError) as e:
            raise EditError(f"arch.json is not valid JSON: {e}") from e
        arch = _arch_structure(base, raw)
    else:
        arch = architecture(base)
    shapes = target_shapes(base, arch)
    tensors: dict[str, Tensor] = {}
    tdir = stage / "tensors"
    if tdir.is_symlink():
        raise EditError("tensors/ must be a directory, not a link")
    if tdir.exists():
        for p in sorted(tdir.glob("*.pt")):
            name = p.name[:-3]
            if name not in shapes:
                raise EditError(f"unknown tensor {name}")
            if name in REFUSED:
                raise EditError(f"{name} may not be edited")
            tensors[name] = _load_tensor(p, shapes[name])
    return Edit(arch, tensors)


def _check_arch(
    base: nn.Module, arch: object, condition: str, max_added: int
) -> dict[str, list[int]]:
    arch = _arch_structure(base, arch)
    if condition == "edit_only" and arch != architecture(base):
        raise EditError("condition edit_only: no capacity may be added")
    n_added = added_params(base, arch)
    if n_added > max_added:
        raise EditError(f"{n_added:,} added parameters > cap {max_added:,}")
    return arch


def _is_scalar_multiple(t: Tensor, orig: Tensor) -> bool:
    t, orig = t.reshape(-1).double(), orig.reshape(-1).double()
    scale = orig.abs().max().item()
    if scale == 0:
        return False
    nz = orig.abs() > 1e-6 * scale
    c = (t[nz] / orig[nz]).median().item()
    if abs(c - 1.0) < 1e-6:
        return False
    close = (t - c * orig).abs() <= 1e-6 * scale + 1e-5 * (c * orig).abs()
    return close.double().mean().item() >= SCALAR_MULTIPLE_FRACTION


def validate(edit: Edit, base: nn.Module, condition: str, max_added: int) -> dict[str, object]:
    if condition not in CONDITIONS:
        raise EditError(f"unknown condition {condition}")
    arch = _check_arch(base, edit.arch, condition, max_added)
    target = with_arch(base, arch).state_dict()
    report: dict[str, dict[str, object]] = {}
    ref = architecture(base)
    for name, t in edit.tensors.items():
        if name not in target:
            raise EditError(f"unknown tensor {name}")
        if name in REFUSED:
            raise EditError(f"{name} may not be edited")
        orig = target[name]
        if t.shape != orig.shape:
            raise EditError(f"{name}: shape {tuple(t.shape)} != {tuple(orig.shape)} for this arch")
        if not torch.isfinite(t).all():
            raise EditError(f"{name}: values must be finite")
        native = tuple(slice(0, s) for s in base.state_dict()[name].shape)
        if _is_scalar_multiple(t[native], orig[native]):
            raise EditError(f"{name}: an edit may not only multiply a whole tensor by a constant")
        delta = t - orig
        changed = int((delta != 0).sum())
        if changed == 0:
            continue
        report[name] = {
            "shape": list(t.shape),
            "changed_entries": changed,
            "native_changed_entries": int((delta[native] != 0).sum()),
            "delta_norm": round(delta.norm().item(), 6),
            "native_rel_change": round(
                (delta[native].norm() / orig[native].norm().clamp_min(1e-12)).item(), 6
            ),
        }
    return {
        "condition": condition,
        "arch": arch,
        "added_heads": [a - b for a, b in zip(arch["n_heads"], ref["n_heads"], strict=True)],
        "added_neurons": [a - b for a, b in zip(arch["d_mlp"], ref["d_mlp"], strict=True)],
        "added_params": added_params(base, arch),
        "tensors": report,
    }


def apply_edit(base: nn.Module, edit: Edit) -> nn.Module:
    model = with_arch(base, edit.arch)
    state = model.state_dict()
    with torch.no_grad():
        for name, t in edit.tensors.items():
            state[name].copy_(t)
    return model.eval()


def save_edit(edit: Edit, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": edit.arch, "tensors": edit.tensors}, path)


def load_edit(path: Path) -> Edit:
    data = torch.load(path, map_location="cpu", weights_only=True)
    return Edit(data["arch"], data["tensors"])
