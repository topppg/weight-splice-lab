from pathlib import Path

import torch

from weight_splice_lab.toy.data import BOS, TOKENIZER, addition_example, ladder_example
from weight_splice_lab.toy.model import (
    ToyConfig,
    ToyLM,
    add_heads,
    add_neurons,
    apply_rope,
    count_params,
    insert_layer,
    load_model,
    rope_cos_sin,
    save_model,
)


def tiny_model() -> ToyLM:
    torch.manual_seed(0)
    return ToyLM(
        ToyConfig(
            vocab_size=TOKENIZER.vocab_size,
            d_model=64,
            n_layers=2,
            n_heads=2,
            head_dim=16,
            d_mlp=96,
        )
    )


def sample_ids() -> torch.Tensor:
    return torch.tensor([TOKENIZER.encode("A:123+456=579\nhello world")])


def test_tokenizer_round_trip() -> None:
    text = "C:0123456789=0123456789\nThe quick brown fox, 3 + 4 = 7 !"
    ids = TOKENIZER.encode(text)
    assert ids[0] == BOS
    assert TOKENIZER.decode(ids) == text
    assert all(TOKENIZER.encode(str(d), bos=False) == [TOKENIZER.digit_ids()[d]] for d in range(10))
    assert TOKENIZER.encode_array(text).tolist() == ids[1:]


def test_addition_format_is_little_endian() -> None:
    import random

    ex = addition_example(random.Random(0), 3)
    a, b = ex.prompt[2:-1].split("+")
    assert int(ex.answer[::-1]) == int(a[::-1]) + int(b[::-1])
    assert len(ex.answer) == 4
    assert ladder_example("successor", random.Random(0), 3).answer == "".join(
        str((int(c) + 1) % 10) for c in a
    )


def test_add_heads_is_exact_noop() -> None:
    model = tiny_model()
    ids = sample_ids()
    before = model(ids)
    heads = add_heads(model, 1, 3)
    assert heads == [2, 3, 4]
    assert torch.equal(model(ids), before)


def test_add_heads_with_nonzero_qkv_is_noop_until_o_written() -> None:
    model = tiny_model()
    ids = sample_ids()
    before = model(ids)
    add_heads(model, 0, 1)
    attn = model.blocks[0].attn
    with torch.no_grad():
        attn.q_proj.weight[-16:].normal_()
        attn.k_proj.weight[-16:].normal_()
        attn.v_proj.weight[-16:].normal_()
    assert torch.equal(model(ids), before)
    with torch.no_grad():
        attn.o_proj.weight[:, -16:].normal_()
    assert not torch.allclose(model(ids), before)


def test_add_neurons_is_exact_noop() -> None:
    model = tiny_model()
    ids = sample_ids()
    before = model(ids)
    neurons = add_neurons(model, 0, 10)
    assert neurons == list(range(96, 106))
    with torch.no_grad():
        model.blocks[0].mlp.gate_proj.weight[-10:].normal_()
        model.blocks[0].mlp.up_proj.weight[-10:].normal_()
    assert torch.equal(model(ids), before)


def test_insert_layer_is_exact_noop() -> None:
    model = tiny_model()
    ids = sample_ids()
    before = model(ids)
    insert_layer(model, 1)
    assert len(model.blocks) == 3
    assert torch.equal(model(ids), before)


def test_save_load_round_trip_with_capacity(tmp_path: Path) -> None:
    model = tiny_model()
    add_heads(model, 0, 2)
    add_neurons(model, 1, 5)
    with torch.no_grad():
        model.blocks[0].attn.o_proj.weight[:, -32:].normal_()
    path = tmp_path / "m.pt"
    save_model(model, path)
    loaded = load_model(path)
    ids = sample_ids()
    assert torch.allclose(loaded(ids), model(ids))
    assert count_params(loaded) == count_params(model)


def test_rope_relative_offset_query() -> None:
    hd = 16
    key = torch.zeros(hd)
    key[[0, 1, 2]] = 1.0
    offset = 5
    query = apply_rope(key, *rope_cos_sin(torch.tensor([-offset]), hd, 10000.0))[0]
    positions = torch.arange(30)
    cos, sin = rope_cos_sin(positions, hd, 10000.0)
    keys = apply_rope(key.expand(30, hd), cos, sin)
    q_at = apply_rope(query.expand(30, hd), cos, sin)
    p_q = 20
    scores = keys[: p_q + 1] @ q_at[p_q]
    assert scores.argmax().item() == p_q - offset
