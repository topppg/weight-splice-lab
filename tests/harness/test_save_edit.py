import json
from pathlib import Path

import pytest
import torch
from conftest import tiny_model

from weight_splice_lab.harness.edit import (
    Edit,
    EditError,
    added_params,
    apply_edit,
    load_edit,
    read_staged,
    save_edit,
    validate,
)
from weight_splice_lab.toy.data import TOKENIZER
from weight_splice_lab.toy.model import architecture

BASE = tiny_model()
ARCH = architecture(BASE)
STATE = BASE.state_dict()
CAP = 100_000


def grown(layer: int = 0, heads: int = 0, neurons: int = 0) -> dict[str, list[int]]:
    arch = json.loads(json.dumps(ARCH))
    arch["n_heads"][layer] += heads
    arch["d_mlp"][layer] += neurons
    return arch


def test_native_edit_is_accepted_and_reported() -> None:
    w = STATE["blocks.1.mlp.down_proj.weight"].clone()
    w[:, 3] += 0.5
    report = validate(Edit(ARCH, {"blocks.1.mlp.down_proj.weight": w}), BASE, "edit_only", CAP)
    tensors = report["tensors"]
    assert isinstance(tensors, dict)
    t = tensors["blocks.1.mlp.down_proj.weight"]
    assert t["changed_entries"] == 64 and t["native_changed_entries"] == 64
    assert report["added_params"] == 0


@pytest.mark.parametrize("name", ["embed.weight", "final_norm.weight"])
def test_refused_tensors(name: str) -> None:
    w = STATE[name].clone()
    w.view(-1)[0] += 1.0
    with pytest.raises(EditError, match="may not be edited"):
        validate(Edit(ARCH, {name: w}), BASE, "add_capacity", CAP)


def test_unknown_name_shape_and_nonfinite() -> None:
    with pytest.raises(EditError, match="unknown tensor"):
        validate(
            Edit(ARCH, {"blocks.9.mlp.up_proj.weight": torch.zeros(1)}), BASE, "edit_only", CAP
        )
    with pytest.raises(EditError, match="shape"):
        validate(
            Edit(ARCH, {"blocks.0.mlp.up_proj.weight": torch.zeros(3, 3)}), BASE, "edit_only", CAP
        )
    w = STATE["blocks.0.mlp.up_proj.weight"].clone()
    w[0, 0] = float("nan")
    with pytest.raises(EditError, match="finite"):
        validate(Edit(ARCH, {"blocks.0.mlp.up_proj.weight": w}), BASE, "edit_only", CAP)


def test_whole_tensor_scaling_is_refused() -> None:
    w = STATE["blocks.0.attn.o_proj.weight"] * 1.7
    with pytest.raises(EditError, match="constant"):
        validate(Edit(ARCH, {"blocks.0.attn.o_proj.weight": w}), BASE, "edit_only", CAP)


def test_condition_edit_only_refuses_capacity() -> None:
    with pytest.raises(EditError, match="edit_only"):
        validate(Edit(grown(neurons=4)), BASE, "edit_only", CAP)


def test_added_parameter_cap() -> None:
    arch = grown(layer=1, heads=1, neurons=10)
    assert added_params(BASE, arch) == 4 * 64 * 16 + 10 * 3 * 64
    validate(Edit(arch), BASE, "add_capacity", 4 * 64 * 16 + 10 * 3 * 64)
    with pytest.raises(EditError, match="cap"):
        validate(Edit(arch), BASE, "add_capacity", 4 * 64 * 16 + 10 * 3 * 64 - 1)


def test_arch_cannot_shrink_or_change_layers() -> None:
    bad = grown()
    bad["d_mlp"][0] -= 1
    with pytest.raises(EditError, match=">="):
        validate(Edit(bad), BASE, "add_capacity", CAP)
    with pytest.raises(EditError, match="one value per layer"):
        validate(Edit({"n_heads": [2], "d_mlp": [96]}), BASE, "add_capacity", CAP)


def test_added_capacity_roundtrip(tmp_path: Path) -> None:
    arch = grown(layer=1, neurons=2)
    gate, up = torch.full((98, 64), 0.3), torch.full((98, 64), 0.3)
    gate[:96] = STATE["blocks.1.mlp.gate_proj.weight"]
    up[:96] = STATE["blocks.1.mlp.up_proj.weight"]
    down = torch.zeros(64, 98)
    down[:, :96] = STATE["blocks.1.mlp.down_proj.weight"]
    edit = Edit(arch, {"blocks.1.mlp.gate_proj.weight": gate, "blocks.1.mlp.up_proj.weight": up})
    report = validate(edit, BASE, "add_capacity", CAP)
    assert report["added_neurons"] == [0, 2]
    x = torch.tensor([TOKENIZER.encode("A:12+34=")])
    with torch.no_grad():
        # Zero down_proj columns: the added neurons cannot change the output yet.
        assert torch.equal(apply_edit(BASE, edit)(x), BASE(x))
        down[:, 96:] = 0.5
        edit.tensors["blocks.1.mlp.down_proj.weight"] = down
        changed = apply_edit(BASE, edit)
        assert not torch.equal(changed(x), BASE(x))
    save_edit(edit, tmp_path / "e.pt")
    again = load_edit(tmp_path / "e.pt")
    assert again.arch == arch and set(again.tensors) == set(edit.tensors)


def test_read_staged_layout(tmp_path: Path) -> None:
    stage = tmp_path / "edited"
    (stage / "tensors").mkdir(parents=True)
    (stage / "arch.json").write_text(json.dumps(grown(heads=1)))
    torch.save(torch.ones(64), stage / "tensors" / "blocks.0.attn_norm.weight.pt")
    edit = read_staged(stage, BASE)
    assert edit.arch["n_heads"] == [3, 2]
    assert list(edit.tensors) == ["blocks.0.attn_norm.weight"]
    assert read_staged(tmp_path / "missing", BASE).arch == ARCH


def test_scalar_multiple_with_a_few_entries_changed_is_refused() -> None:
    w = STATE["blocks.0.attn.o_proj.weight"] * 2
    w[0, 0] += 1.0
    w[5, 7] -= 0.3
    with pytest.raises(EditError, match="constant"):
        validate(Edit(ARCH, {"blocks.0.attn.o_proj.weight": w}), BASE, "edit_only", CAP)
    z = torch.zeros_like(STATE["blocks.0.attn.o_proj.weight"])
    z[3, 3] = 0.5
    with pytest.raises(EditError, match="constant"):
        validate(Edit(ARCH, {"blocks.0.attn.o_proj.weight": z}), BASE, "edit_only", CAP)


def test_scaling_the_native_part_of_a_grown_tensor_is_refused() -> None:
    arch = grown(layer=1, neurons=2)
    up = torch.full((98, 64), 0.3)
    up[:96] = STATE["blocks.1.mlp.up_proj.weight"] * 3
    with pytest.raises(EditError, match="constant"):
        validate(Edit(arch, {"blocks.1.mlp.up_proj.weight": up}), BASE, "add_capacity", CAP)


def test_large_column_edit_is_not_a_scalar_multiple() -> None:
    w = STATE["blocks.1.mlp.down_proj.weight"].clone()
    w[:, :40] *= 2
    validate(Edit(ARCH, {"blocks.1.mlp.down_proj.weight": w}), BASE, "edit_only", CAP)


def _stage(tmp_path: Path, name: str = "blocks.0.mlp.down_proj.weight") -> Path:
    (tmp_path / "edited" / "tensors").mkdir(parents=True)
    return tmp_path / "edited" / "tensors" / f"{name}.pt"


@pytest.mark.parametrize(
    ("write", "match"),
    [
        (lambda p: p.write_bytes(b""), "not a readable tensor"),
        (lambda p: p.write_bytes(b"not a checkpoint"), "not a readable tensor"),
        (lambda p: torch.save({"w": torch.zeros(2)}, p), "plain tensor"),
        (lambda p: torch.save([torch.zeros(2)], p), "plain tensor"),
        (lambda p: torch.save(torch.zeros(64, 96, dtype=torch.int8), p), "float dtype"),
        (lambda p: torch.save(torch.zeros(3, 3), p), "shape"),
        (lambda p: torch.save(torch.zeros(1, dtype=torch.int8).expand(64, 96), p), "dtype|storage"),
        (lambda p: torch.save(torch.zeros(1).expand(64, 96), p), "storage"),
        (lambda p: torch.save(torch.zeros(64, 200)[:, :96], p), "storage"),
        (lambda p: torch.save(torch.zeros(64 * 96 + 1)[1:].view(64, 96), p), "storage"),
        (lambda p: p.write_bytes(b"x" * (64 * 96 * 8 + (1 << 16) + 1)), "larger"),
    ],
)
def test_read_staged_rejects_bad_payloads(tmp_path: Path, write: object, match: str) -> None:
    p = _stage(tmp_path)
    write(p)  # type: ignore[operator]
    with pytest.raises(EditError, match=match):
        read_staged(tmp_path / "edited", BASE)


def test_read_staged_checks_names_before_loading(tmp_path: Path) -> None:
    p = _stage(tmp_path, "blocks.9.mlp.up_proj.weight")
    p.write_bytes(b"garbage")
    with pytest.raises(EditError, match="unknown tensor"):
        read_staged(tmp_path / "edited", BASE)
    p.unlink()
    (p.parent / "embed.weight.pt").write_bytes(b"garbage")
    with pytest.raises(EditError, match="may not be edited"):
        read_staged(tmp_path / "edited", BASE)


def test_read_staged_rejects_links(tmp_path: Path) -> None:
    outside = tmp_path / "outside.pt"
    torch.save(STATE["blocks.0.mlp.down_proj.weight"].clone(), outside)
    p = _stage(tmp_path)
    p.symlink_to(outside)
    with pytest.raises(EditError, match="link"):
        read_staged(tmp_path / "edited", BASE)
    p.unlink()
    p.hardlink_to(outside)
    with pytest.raises(EditError, match="hard links"):
        read_staged(tmp_path / "edited", BASE)
    p.unlink()
    (tmp_path / "edited" / "arch.json").symlink_to(outside)
    with pytest.raises(EditError, match="link"):
        read_staged(tmp_path / "edited", BASE)


@pytest.mark.parametrize(
    "arch",
    [
        '{"n_heads": [2, 2], "d_mlp": [96, 1000000000]}',
        "{bad json",
        '{"n_heads": [2.0, 2], "d_mlp": [96, 96]}',
    ],
)
def test_read_staged_rejects_bad_arch_before_allocating(tmp_path: Path, arch: str) -> None:
    (tmp_path / "edited").mkdir()
    (tmp_path / "edited" / "arch.json").write_text(arch)
    with pytest.raises(EditError):
        read_staged(tmp_path / "edited", BASE)


def test_read_staged_accepts_half_precision(tmp_path: Path) -> None:
    w = STATE["blocks.0.mlp.down_proj.weight"].clone()
    w[0, 0] += 1
    torch.save(w.half(), _stage(tmp_path))
    edit = read_staged(tmp_path / "edited", BASE)
    assert edit.tensors["blocks.0.mlp.down_proj.weight"].dtype == torch.float32
