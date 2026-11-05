import hashlib
import json
import struct
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# bfloat16 tensors are handled as their raw uint16 bit patterns so edits can be checked bit for bit.

ST_DTYPES: dict[str, type[np.generic]] = {
    "BF16": np.uint16,
    "F16": np.float16,
    "F32": np.float32,
    "I64": np.int64,
    "I32": np.int32,
}


def bf16_bits(x: np.ndarray) -> np.ndarray:
    # float32 -> bfloat16 round-to-nearest-even, identical to torch's .to(torch.bfloat16) for finite values
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    u = u + 0x7FFF + ((u >> 16) & 1)
    return (u >> 16).astype(np.uint16)


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def safetensors_file(model_dir: Path) -> Path:
    files = sorted(model_dir.glob("*.safetensors"))
    if len(files) != 1:
        raise ValueError(f"expected one .safetensors file in {model_dir}, found {len(files)}")
    return files[0]


def read_tensors(path: Path, names: Iterable[str]) -> dict[str, np.ndarray]:
    wanted = set(names)
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        out: dict[str, np.ndarray] = {}
        for name in sorted(wanted):
            info = header[name]
            start, end = info["data_offsets"]
            f.seek(8 + n + start)
            buf = f.read(end - start)
            out[name] = (
                np.frombuffer(buf, dtype=ST_DTYPES[info["dtype"]]).reshape(info["shape"]).copy()
            )
    return out


@dataclass(frozen=True)
class TensorEdit:
    name: str
    shape: tuple[int, ...]
    form: str
    base_sha256: str
    edited_sha256: str
    changed_entries: int
    idx: np.ndarray
    val: np.ndarray
    a: np.ndarray | None = None
    b: np.ndarray | None = None


def load_published_edit(edit_dir: Path) -> list[TensorEdit]:
    manifest = json.loads((edit_dir / "manifest.json").read_text())
    z = np.load(edit_dir / "edit.npz")
    edits = []
    for i, t in enumerate(manifest["tensors"]):
        lowrank = t["form"] == "lowrank"
        edits.append(
            TensorEdit(
                name=t["name"],
                shape=tuple(t["shape"]),
                form=t["form"],
                base_sha256=t["base_sha256"],
                edited_sha256=t["edited_sha256"],
                changed_entries=t["changed_entries"],
                idx=z[f"t{i}_idx"],
                val=z[f"t{i}_val"],
                a=z[f"t{i}_a"] if lowrank else None,
                b=z[f"t{i}_b"] if lowrank else None,
            )
        )
    return edits


def apply_tensor_edit(base_bits: np.ndarray, edit: TensorEdit) -> np.ndarray:
    if edit.form == "lowrank":
        assert edit.a is not None and edit.b is not None
        w = bf16_to_f32(base_bits).astype(np.float64)
        # added one rank at a time, then rounded via float32, exactly as the published apply_edit.py
        for k in range(edit.a.shape[1]):
            w = w + np.outer(edit.a[:, k], edit.b[k])
        out = bf16_bits(w.astype(np.float32)).reshape(-1)
    else:
        out = base_bits.copy().reshape(-1)
    out[edit.idx] = edit.val
    return out.reshape(base_bits.shape)


def rebuild_published_edit(model_dir: Path, edit_dir: Path) -> dict[str, np.ndarray]:
    edits = load_published_edit(edit_dir)
    base = read_tensors(safetensors_file(model_dir), [e.name for e in edits])
    out: dict[str, np.ndarray] = {}
    for e in edits:
        b = base[e.name]
        if b.shape != e.shape or sha256(b) != e.base_sha256:
            raise ValueError(f"{e.name}: base tensor differs from the one the edit was made on")
        edited = apply_tensor_edit(b, e)
        if sha256(edited) != e.edited_sha256:
            raise ValueError(f"{e.name}: rebuilt tensor does not match the evaluated checkpoint")
        out[e.name] = edited
    return out


def ablate_direction(
    w: np.ndarray, u: np.ndarray, alpha: float, columns: Sequence[int] | None = None
) -> np.ndarray:
    # W' = W - alpha * u (u^T W), optionally only on the given output columns (MLP neurons)
    u = u / np.linalg.norm(u)
    coeff = u @ w
    if columns is not None:
        mask = np.zeros_like(coeff)
        mask[np.asarray(columns)] = 1.0
        coeff = coeff * mask
    return w - alpha * np.outer(u, coeff)


def recover_rank_one(delta: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    # delta ~= s * outer(u, v) with unit u, v; returns (u, v, share of squared norm in the top component)
    left, s, right = np.linalg.svd(delta.astype(np.float64), full_matrices=False)
    return left[:, 0], right[0], float(s[0] ** 2 / np.sum(s**2))


def count_changed(base_bits: np.ndarray, edited_bits: np.ndarray) -> int:
    return int(np.count_nonzero(base_bits != edited_bits))
