import json
import random
from pathlib import Path

import numpy as np
import torch
from conftest import tiny_model

from weight_splice_lab.harness.family import TOY
from weight_splice_lab.harness.forward import ForwardLimits, ForwardServer, PracticeItem
from weight_splice_lab.toy import metrics
from weight_splice_lab.toy.data import PAD, TOKENIZER, UNK, Example
from weight_splice_lab.toy.model import ToyLM


def score(item: PracticeItem, ids: list[int]) -> bool:
    return TOY.grade(TOY.tokenizer(), ids, item.task, item.answer)


NL = TOKENIZER.id("\n")


class Scripted(torch.nn.Module):
    # generate() appends a fixed continuation, so grading can be checked without a real model
    def __init__(self, cont: list[int]) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.cont = cont

    def generate(self, x: torch.Tensor, n: int) -> torch.Tensor:
        tail = torch.tensor([self.cont[:n]] * x.shape[0])
        return torch.cat([x, tail], dim=1)


def server(model: torch.nn.Module, items: list[PracticeItem], work: Path) -> ForwardServer:
    return ForwardServer(model, {i.id: i for i in items}, work, ForwardLimits())  # type: ignore[arg-type]


def test_practice_and_hidden_agree_when_a_pad_precedes_the_answer(tmp_path: Path) -> None:
    # codex review #5: answer 30, generation PAD 3 0 was practice=True, hidden=False
    three, zero = TOKENIZER.id("3"), TOKENIZER.id("0")
    model = Scripted([PAD, three, zero, NL])
    item = PracticeItem("add_x", "addition", "A:1+2=", "30", 1)
    summary = server(model, [item], tmp_path).generate(
        {"type": "generate", "model": "original", "ids": ["add_x"]}, 1
    )
    hidden = metrics.answer_match(model, [Example(item.prompt, item.answer)])  # type: ignore[arg-type]
    assert summary["original"]["accuracy"] == 0.0 and hidden == [False]
    rec = json.loads((tmp_path / "fwd_1.jsonl").read_text())
    assert rec["output"] == "<pad>30" and rec["output_ids"] == [PAD, three, zero]
    assert not score(item, [PAD, three, zero]) and score(item, [three, zero, UNK])


def test_native_practice_needs_the_newline_at_its_position() -> None:
    c = PracticeItem("copy_x", "copy", "C:7=", "7", 1)
    seven = TOKENIZER.id("7")
    assert score(c, [seven, NL]) and not score(c, [seven, UNK, NL])
    assert metrics.exact_match(Scripted([seven, UNK, NL]), [Example("C:7=", "7")]) == [False]  # type: ignore[arg-type]


def test_termination_reads_the_token_at_position_n_plus_1() -> None:
    d = TOKENIZER.id("5")
    # specials before the newline used to shift the decoded string and hide the terminator
    res = metrics.termination(Scripted([d, PAD, NL]), [1], 4, 0)  # type: ignore[arg-type]
    assert res["terminated"] == {1: 1.0} and res["strict"] == {1: 0.0}
    res = metrics.termination(Scripted([UNK, d, d]), [1], 4, 0)  # type: ignore[arg-type]
    assert res["terminated"] == {1: 0.0}


def test_hidden_rule_unchanged_on_a_real_model() -> None:
    model: ToyLM = tiny_model()
    examples = [metrics.addition_example(random.Random(i), 2) for i in range(6)]
    gens = metrics.generate_ids(model, [TOKENIZER.encode(e.prompt) for e in examples], [3] * 6)
    old = [TOKENIZER.decode(g) == e.answer for g, e in zip(gens, examples, strict=True)]
    assert metrics.answer_match(model, examples) == old


def test_teacher_forced_arrays_are_zero_at_padding(tmp_path: Path) -> None:
    model = tiny_model()
    srv = server(model, [], tmp_path)
    sites = [
        "resid_pre.0",
        "resid_final",
        "attn.0.pattern",
        "attn.1.head_out",
        "mlp.0.hidden",
        "logits",
    ]
    req = {"type": "teacher_forced", "model": "original", "texts": ["ab", "abcdef"],
           "return": {"topk": 2, "sites": sites}}  # fmt: skip
    srv.teacher_forced(req, 1)
    d = tmp_path / "fwd_1" / "original"
    short = 3  # BOS + 2 chars
    for key in [*sites, "topk_logprobs", "topk_ids", "token_logprobs"]:
        arr = np.load(d / f"{key}.npy")
        if key.endswith(".pattern"):
            assert not arr[0, :, short:, :].any() and not arr[0, :, :, short:].any()
            assert arr[0, :, :short, :short].any()
        elif key.endswith(".head_out"):
            assert not arr[0, :, short:].any() and arr[0, :, :short].any()
        elif key == "token_logprobs":
            assert not arr[0, short - 1 :].any() and arr[0, : short - 1].all()
        else:
            assert not arr[0, short:].any(), key
            assert arr[0, :short].any() and arr[1].any(), key
    meta = json.loads((tmp_path / "fwd_1" / "meta.json").read_text())
    assert "0 at padding" in meta["padding"] and meta["lengths"] == [3, 7]
