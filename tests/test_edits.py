import numpy as np
import pytest
import torch

from weight_splice_lab.edits import (
    TensorEdit,
    ablate_direction,
    apply_tensor_edit,
    bf16_bits,
    bf16_to_f32,
    count_changed,
    recover_rank_one,
    sha256,
)


def test_bf16_bits_matches_torch_rounding() -> None:
    x = np.random.default_rng(0).normal(size=4096).astype(np.float32) * 3
    ref = torch.from_numpy(x).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
    assert np.array_equal(bf16_bits(x), ref)
    assert np.array_equal(bf16_bits(bf16_to_f32(ref)), ref)


def test_ablate_direction_removes_component() -> None:
    rng = np.random.default_rng(1)
    w = rng.normal(size=(16, 10))
    u = rng.normal(size=16)
    out = ablate_direction(w, u, alpha=1.0)
    assert np.allclose((u / np.linalg.norm(u)) @ out, 0.0)
    rest = rng.normal(size=16)
    rest -= (rest @ u) / (u @ u) * u
    assert np.allclose(rest @ out, rest @ w)


def test_ablate_direction_columns_only() -> None:
    rng = np.random.default_rng(2)
    w = rng.normal(size=(8, 12))
    u = rng.normal(size=8)
    cols = [1, 5, 7]
    out = ablate_direction(w, u, alpha=6.0, columns=cols)
    untouched = [c for c in range(12) if c not in cols]
    assert np.array_equal(out[:, untouched], w[:, untouched])
    uh = u / np.linalg.norm(u)
    assert np.allclose(uh @ out[:, cols], -5.0 * (uh @ w[:, cols]))


def test_recover_rank_one() -> None:
    rng = np.random.default_rng(3)
    u, v = rng.normal(size=20), rng.normal(size=7)
    got_u, got_v, share = recover_rank_one(3.0 * np.outer(u, v))
    assert share == pytest.approx(1.0)
    assert abs(got_u @ u / np.linalg.norm(u)) == pytest.approx(1.0)
    assert abs(got_v @ v / np.linalg.norm(v)) == pytest.approx(1.0)


def _edit(base: np.ndarray, edited: np.ndarray, **kw: np.ndarray) -> TensorEdit:
    flat_b, flat_e = base.reshape(-1), edited.reshape(-1)
    idx = np.flatnonzero(flat_b != flat_e).astype(np.int32)
    return TensorEdit(
        name="t",
        shape=base.shape,
        form="lowrank" if kw else "sparse",
        base_sha256=sha256(base),
        edited_sha256=sha256(edited),
        changed_entries=idx.size,
        idx=idx,
        val=flat_e[idx],
        **kw,
    )


def test_apply_sparse_edit_roundtrip() -> None:
    base = bf16_bits(np.random.default_rng(4).normal(size=(6, 9)).astype(np.float32))
    target = base.copy()
    target[:, [2, 4]] = 0
    e = _edit(base, target)
    out = apply_tensor_edit(base, e)
    assert np.array_equal(out, target)
    assert count_changed(base, out) == e.changed_entries


def test_apply_lowrank_edit() -> None:
    rng = np.random.default_rng(5)
    base = bf16_bits(rng.normal(size=(6, 9)).astype(np.float32))
    a, b = rng.normal(size=(6, 1)), rng.normal(size=(1, 9))
    w = bf16_to_f32(base).astype(np.float64) + a @ b
    target = bf16_bits(w.astype(np.float32))
    e = TensorEdit(
        name="t",
        shape=base.shape,
        form="lowrank",
        base_sha256="",
        edited_sha256="",
        changed_entries=0,
        idx=np.zeros(0, np.int32),
        val=np.zeros(0, np.uint16),
        a=a,
        b=b,
    )
    assert np.array_equal(apply_tensor_edit(base, e), target)
