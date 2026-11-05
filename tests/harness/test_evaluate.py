import copy
from pathlib import Path

import pytest
import torch
from conftest import tiny_model

from weight_splice_lab.harness import evaluate
from weight_splice_lab.toy.data import TOKENIZER, text_eval_chunks
from weight_splice_lab.toy.model import add_neurons, save_model

TEXT = TOKENIZER.encode_array("The cat sat on 3 mats.\n" * 200)
OLD_KEYS = {"addition_by_n", "addition_mean_n1_8", "addition_mean_n9_20", "text_bpc", "text_kl",
            "native", "meets_bar", "params", "n_per_length", "seconds"}  # fmt: skip


def test_hardened_eval_keeps_primary_keys_and_adds_robustness() -> None:
    base = tiny_model()
    model = copy.deepcopy(base)
    add_neurons(model, 1, 4)
    with torch.no_grad():
        model.blocks[1].mlp.gate_proj.weight[96:] = 1.0
        model.blocks[0].attn.o_proj.weight *= 1.01
    chunks = text_eval_chunks(TEXT, 32, 4, seed=0)
    old = evaluate.evaluate_model(model, base, chunks, n_per=2)
    new = evaluate.evaluate_model(model, base, chunks, n_per=2, test_text=TEXT)
    assert set(old) == OLD_KEYS
    assert set(new) == OLD_KEYS | {"robustness", "meets_primary_bar", "meets_strict_bar"}
    for k in OLD_KEYS - {"seconds"}:
        assert old[k] == new[k]
    rob = new["robustness"]
    assert isinstance(rob, dict)
    assert set(rob) == {"thresholds", "a_prefixed_text", "b_termination", "c_not_first_line",
                        "d_length_n21_30", "e_native_weight_change", "f_added_params"}  # fmt: skip
    assert rob["f_added_params"]["allocated"] == 4 * 3 * 64
    assert rob["f_added_params"]["inactive"] == 4 * 3 * 64
    assert rob["e_native_weight_change"]["changed_tensors"] == 1
    assert set(rob["d_length_n21_30"]["by_n"]) == {str(n) for n in range(21, 31)}
    assert new["meets_primary_bar"] is False
    assert new["meets_strict_bar"] is False


def test_robust_pass_needs_all_three_thresholded_probes() -> None:
    ok = {"a_prefixed_text": {"pass": True}, "b_termination": {"pass": True},
          "c_not_first_line": {"pass": True}}  # fmt: skip
    assert evaluate.robust_pass(ok)
    for k in ok:
        assert not evaluate.robust_pass(ok | {k: {"pass": False}})


def test_base_cache_is_keyed_on_checkpoint_code_and_settings(
    tmp_path: Path, tiny_ckpt: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # codex review #6 / audit m9: the cache was keyed only on n and ignored the checkpoint
    calls: list[int] = []

    def fake(ckpt: Path, edit: Path | None, n: int) -> dict[str, object]:
        calls.append(n)
        return {"run": len(calls)}

    monkeypatch.setattr(evaluate, "evaluate_edit", fake)
    a = evaluate.base_reference(tiny_ckpt, tmp_path, 2)
    assert evaluate.base_reference(tiny_ckpt, tmp_path, 2) == a and calls == [2]
    evaluate.base_reference(tiny_ckpt, tmp_path, 3)
    assert calls == [2, 3]
    other = tmp_path / "other.pt"
    with torch.no_grad():
        m = tiny_model()
        m.embed.weight += 1.0
    save_model(m, other)
    evaluate.base_reference(other, tmp_path, 2)
    assert calls == [2, 3, 2]
    monkeypatch.setattr(evaluate, "EVAL_SOURCES", evaluate.EVAL_SOURCES[:-1])  # code change
    evaluate.base_reference(tiny_ckpt, tmp_path, 2)
    assert calls == [2, 3, 2, 2]
    key = evaluate.eval_key(tiny_ckpt, 2)
    assert set(key) == {
        "checkpoint_sha256",
        "eval_code_sha256",
        "dataset",
        "seeds",
        "thresholds",
        "n_per",
    }
