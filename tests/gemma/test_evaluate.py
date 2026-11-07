import copy
import random
import re
from pathlib import Path

import pytest
import torch
from torch import Tensor

from weight_splice_lab.gemma import evaluate
from weight_splice_lab.gemma.evaluate import EvalData, Scored
from weight_splice_lab.gemma.model import GemmaConfig, GemmaLM, add_heads, add_neurons
from weight_splice_lab.gemma.task import reversed_example
from weight_splice_lab.gemma.tokenizer import BOS, GemmaTokenizer

VOCAB = 300


class CharTokenizer(GemmaTokenizer):
    # one id per character (ord + 3), so prompts fit a tiny vocabulary
    def __init__(self) -> None:
        self.vocab_size = VOCAB
        self.digit_ids = frozenset(ord(str(d)) + 3 for d in range(10))

    def encode(self, text: str, bos: bool = True) -> list[int]:
        ids = [ord(c) % (VOCAB - 3) + 3 for c in text]
        return [BOS, *ids] if bos else ids

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i - 3) for i in ids if i >= 3)


TOK = CharTokenizer()


class FakeAdder:
    # reads "a+b" (plain or chat prompt) and emits the answer plus `tail`
    def __init__(self, tail: str, offset: int = 0) -> None:
        self.tail, self.offset = tail, offset

    def generate(self, idx: Tensor, n_new: int) -> Tensor:
        rows = []
        for ids in idx.tolist():
            a, b = re.findall(r"(\d+) ?\+ ?(\d+)", TOK.decode(ids))[-1]
            out = TOK.encode(str(int(a) + int(b) + self.offset) + self.tail * n_new, bos=False)
            rows.append(ids + out[:n_new])
        return torch.tensor(rows)


def tiny() -> GemmaLM:
    torch.manual_seed(0)
    cfg = GemmaConfig(
        vocab_size=VOCAB, d_model=32, n_layers=2, n_heads=2, n_kv_heads=1, head_dim=16, d_mlp=64,
        query_pre_attn_scalar=16.0, sliding_window=8, layer_types=["sliding", "full"],
    )  # fmt: skip
    return GemmaLM(cfg).eval()


def edited(base: GemmaLM) -> GemmaLM:
    model = copy.deepcopy(base, memo={id(base.embed.weight): base.embed.weight})
    add_heads(model, 0, 2)
    add_neurons(model, 1, 3)
    return model


def synthetic(n: int, length: int, first: int) -> list[Scored]:
    g = torch.Generator().manual_seed(n + length)
    return [([BOS, *torch.randint(3, VOCAB, (length - 1,), generator=g).tolist()], first)] * n


def test_kl_of_model_to_itself_is_zero() -> None:
    base = tiny()
    items = synthetic(3, 40, 0) + synthetic(2, 30, 12)
    same = evaluate.teacher_forced(base, base, items)
    assert same["kl"] == 0.0
    assert same["nll"] == same["nll_base"] > 0
    assert same["positions"] == 3 * 39 + 2 * (29 - 12)
    assert evaluate.teacher_forced(edited(base), base, items)["kl"] < 1e-9


def test_edited_model_has_positive_kl() -> None:
    base = tiny()
    model = edited(base)
    with torch.no_grad():
        model.blocks[1].attn.o_proj.weight[:, 32:48].normal_()
        model.blocks[1].attn.v_proj.weight.normal_()
    out = evaluate.teacher_forced(model, base, synthetic(2, 100, 0))
    assert out["kl"] > 1e-4
    assert out["nll"] != out["nll_base"]


def test_added_param_report_counts_live_and_dead_units() -> None:
    base = tiny()
    model = edited(base)
    d, hd = 32, 16
    attn, mlp = model.blocks[0].attn, model.blocks[1].mlp
    with torch.no_grad():
        # added head 3 (the second): o_proj columns and its own key/value head (kv index 2)
        attn.o_proj.weight[:, 3 * hd : 4 * hd] = 1.0
        attn.k_proj.weight[2 * hd : 3 * hd, :4] = 1.0
        attn.v_proj.weight[1 * hd : 2 * hd, 0] = 1.0  # first added head stays inactive
        mlp.gate_proj.weight[64, :5] = 1.0
        mlp.down_proj.weight[:3, 66] = 1.0
    report = evaluate.added_param_report(model, base)
    head, neuron = 4 * d * hd, 3 * d
    assert report["allocated"] == 2 * head + 3 * neuron
    assert report["active"] == head + neuron
    assert report["inactive"] == head + 2 * neuron
    assert report["nonzero"] == d * hd + hd * 4 + hd + 5 + 3
    assert report["nonzero_active"] == d * hd + hd * 4 + 3
    assert evaluate.native_weight_change(model, base)["changed_tensors"] == 0


def test_native_weight_change_ignores_added_units() -> None:
    base = tiny()
    model = edited(base)
    with torch.no_grad():
        model.blocks[0].mlp.down_proj.weight[:, :64] *= 1.1
        model.blocks[1].mlp.down_proj.weight[:, 64:] = 1.0
    change = evaluate.native_weight_change(model, base)
    assert change["changed_tensors"] == 1
    assert change["max_rel"] == pytest.approx(0.1, rel=1e-4)


@pytest.mark.parametrize("chat", [False, True])
def test_addition_grading_through_a_fake(chat: bool) -> None:
    def acc(model: FakeAdder) -> dict[int, float]:
        return evaluate.addition_accuracy(model, TOK, [1, 2, 5, 12], 20, 4321, chat)

    assert acc(FakeAdder(" ")) == {1: 1.0, 2: 1.0, 5: 1.0, 12: 1.0}
    assert set(acc(FakeAdder("7")).values()) == {0.0}  # a digit after the answer is wrong
    assert set(acc(FakeAdder(" ", offset=1)).values()) == {0.0}


def test_evaluate_model_on_synthetic_data() -> None:
    base = tiny()
    data = EvalData(TOK, synthetic(2, 32, 0), synthetic(2, 32, 0), synthetic(2, 40, 20))
    same = evaluate.evaluate_model(base, base, n_per=2, data=data)
    assert same["text_kl"] == same["chat_kl"] == same["digit_text_kl"] == 0.0
    assert same["meets_bar"] is True
    assert same["meets_primary_bar"] is False and same["meets_strict_bar"] is False
    assert set(same["addition_by_n"]) == {str(n) for n in range(1, 21)}
    assert set(same["length_n21_30"]["by_n"]) == {str(n) for n in range(21, 31)}
    assert set(same["chat_addition"]["by_n"]) == {str(n) for n in range(1, 9)}
    assert same["positions"] == {"text": 62, "digit_text": 62, "chat": 2 * 19}
    model = edited(base)
    with torch.no_grad():
        model.blocks[1].mlp.down_proj.weight[:, 64:].normal_()
        model.blocks[1].mlp.gate_proj.weight[64:].normal_()
        model.blocks[1].mlp.up_proj.weight[64:].normal_()
    diff = evaluate.evaluate_model(model, base, n_per=2, data=data)
    assert diff["text_kl"] > 0
    assert diff["added_params"]["active"] == 3 * 3 * 32


def test_base_reference_is_cached_on_its_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    key = {"checkpoint_sha256": "a"}

    def fake_edit(ckpt: Path, edit: Path | None, n: int, task: str) -> dict[str, object]:
        calls.append(n)
        return {"run": len(calls)}

    monkeypatch.setattr(evaluate, "evaluate_edit", fake_edit)
    monkeypatch.setattr(evaluate, "eval_key", lambda ckpt, n, task: dict(key, n_per=n))
    first = evaluate.base_reference(tmp_path / "base.pt", tmp_path, n_per=3)
    assert evaluate.base_reference(tmp_path / "base.pt", tmp_path, n_per=3) == first
    key["checkpoint_sha256"] = "b"
    assert evaluate.base_reference(tmp_path / "base.pt", tmp_path, n_per=3)["run"] == 2
    assert calls == [3, 3]


def test_reversed_examples_are_lsb_first_sums() -> None:
    rng = random.Random(0)
    for n in (1, 2, 7, 20):
        for _ in range(50):
            ex = reversed_example(rng, n)
            a, b = ex.prompt.removeprefix("R:").removesuffix("=").split("+")
            assert len(a) == len(b) == n and len(ex.answer) == n + 1
            assert int(ex.answer[::-1]) == int(a[::-1]) + int(b[::-1])


def test_reversed_eval_reports_plain_addition_and_its_own_key() -> None:
    base = tiny()
    data = EvalData(TOK, synthetic(1, 16, 0), synthetic(1, 16, 0), synthetic(1, 20, 10))
    out = evaluate.evaluate_model(base, base, n_per=1, data=data, task="reversed")
    assert out["task"] == "reversed"
    assert set(out["plain_addition_n1_8"]["by_n"]) == {str(n) for n in range(1, 9)}


def test_reversed_eval_module_has_the_finalize_api() -> None:
    from weight_splice_lab.gemma import evaluate_reversed

    for name in ("evaluate_edit", "base_reference", "eval_key"):
        assert getattr(evaluate_reversed, name).keywords == {"task": "reversed"}
