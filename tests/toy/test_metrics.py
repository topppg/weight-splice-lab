import copy
import random

import numpy as np
import pytest
import torch

from weight_splice_lab.toy import metrics
from weight_splice_lab.toy.data import (
    BOS,
    TOKENIZER,
    native_lines,
    prefixed_text_chunks,
    text_before_newline,
)
from weight_splice_lab.toy.model import ToyConfig, ToyLM, add_heads, add_neurons

TEXT = TOKENIZER.encode_array("7 cats sat on 3 mats.\n" * 40 + "The end 42 is here.\n" * 40)


def tiny_model() -> ToyLM:
    torch.manual_seed(0)
    cfg = ToyConfig(
        vocab_size=TOKENIZER.vocab_size, d_model=64, n_layers=2, n_heads=2, head_dim=16, d_mlp=96
    )
    return ToyLM(cfg).eval()


def test_prefixed_text_chunks_start_with_prefix_then_non_digit() -> None:
    chunks = prefixed_text_chunks(TEXT, 64, 50, seed=0, prefix="A:")
    assert chunks.shape == (50, 65)
    digits = set(TOKENIZER.digit_ids())
    for row in chunks.tolist():
        assert row[0] == BOS
        assert TOKENIZER.decode(row[1:3]) == "A:"
        assert row[3] not in digits


def test_text_before_newline_ends_a_line() -> None:
    rng = random.Random(0)
    for length in (8, 32, 64):
        s = text_before_newline(TEXT, rng, length)
        assert len(s) == length and s.endswith("\n")


def test_native_lines_are_training_format() -> None:
    lines = native_lines(random.Random(0), 3).split("\n")
    assert len(lines) == 4 and lines[-1] == ""
    for line in lines[:-1]:
        task, rest = line.split(":", 1)
        assert task in "CRM" and "=" in rest


def test_termination_counts_end_token_after_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(model: ToyLM, prompts: list[list[int]], n_new: list[int]) -> list[list[int]]:
        out = []
        for p, n in zip(prompts, n_new, strict=True):
            a, b = TOKENIZER.decode(p)[2:-1].split("+")
            s = str(int(a[::-1]) + int(b[::-1])).zfill(len(a) + 1)[::-1]
            text = (s + "\n")[:n] if len(a) % 2 else s + "5"
            out.append(TOKENIZER.encode(text, bos=False))
        return out

    monkeypatch.setattr(metrics, "generate_ids", fake)
    res = metrics.termination(tiny_model(), [1, 2, 3], n_per=10, seed=0)
    assert res["terminated"] == {1: 1.0, 2: 0.0, 3: 1.0}
    assert res["strict"] == {1: 1.0, 2: 0.0, 3: 1.0}


def test_continuations_match_generate() -> None:
    model = tiny_model()
    prompts = ["A:12+34=", "C:5=", "A:99+01="]
    outs = metrics.continuations(model, prompts, 3)
    for p, o in zip(prompts, outs, strict=True):
        ids = torch.tensor([TOKENIZER.encode(p)])
        assert o == TOKENIZER.decode(model.generate(ids, 3)[0, ids.shape[1] :].tolist())


def test_prefixed_accuracy_uses_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake(model: ToyLM, examples: list[metrics.Example]) -> list[bool]:
        seen.extend(e.prompt for e in examples)
        return [True] * len(examples)

    monkeypatch.setattr(metrics, "answer_match", fake)
    acc = metrics.prefixed_target_accuracy(tiny_model(), [2], 4, 0, lambda r: "C:1=1\n")
    assert acc == {2: 1.0}
    assert len(seen) == 4 and all(p.startswith("C:1=1\nA:") for p in seen)


def test_native_weight_change() -> None:
    base = tiny_model()
    model = copy.deepcopy(base)
    add_heads(model, 0, 1)
    add_neurons(model, 1, 3)
    with torch.no_grad():
        model.blocks[0].attn.o_proj.weight[:, -16:] = 1.0
        model.blocks[1].mlp.down_proj.weight[:, -3:] = 1.0
    assert metrics.native_weight_change(model, base)["total_rel"] == 0.0
    with torch.no_grad():
        model.blocks[1].mlp.down_proj.weight[:, :96] *= 1.5
    rep = metrics.native_weight_change(model, base)
    assert rep["changed_tensors"] == 1
    assert rep["per_tensor_rel"] == {"blocks.1.mlp.down_proj.weight": pytest.approx(0.5)}
    assert 0 < rep["total_rel"] < 0.5


def test_added_param_report_counts_inactive_units() -> None:
    base = tiny_model()
    model = copy.deepcopy(base)
    add_heads(model, 0, 2)
    add_neurons(model, 1, 3)
    attn, mlp = model.blocks[0].attn, model.blocks[1].mlp
    with torch.no_grad():
        attn.q_proj.weight[32:64] = 1.0  # both added heads read, only head 2 writes
        attn.o_proj.weight[:5, 32:48] = 1.0
        mlp.gate_proj.weight[96:99] = 1.0  # neurons read, only neuron 96 writes
        mlp.down_proj.weight[0, 96] = 1.0
    rep = metrics.added_param_report(model, base)
    head, neuron = 4 * 64 * 16, 3 * 64
    assert rep["allocated"] == 2 * head + 3 * neuron
    assert rep["inactive"] == head + 2 * neuron
    assert rep["active"] == head + neuron
    assert rep["nonzero"] == 2 * 16 * 64 + 5 * 16 + 3 * 64 + 1
    assert rep["nonzero_active"] == 16 * 64 + 5 * 16 + 64 + 1


def test_base_against_itself_is_unchanged() -> None:
    base = tiny_model()
    assert metrics.native_weight_change(base, base)["total_rel"] == 0.0
    assert metrics.added_param_report(base, base)["allocated"] == 0
    assert np.isclose(
        metrics.kl_to_reference(base, base, prefixed_text_chunks(TEXT, 32, 4, 0, "A:")), 0.0
    )
