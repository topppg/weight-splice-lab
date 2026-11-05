import time
from collections.abc import Callable, Iterator
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.generate import BatchGenerator

from weight_splice_lab.edits import bf16_to_f32
from weight_splice_lab.eval import STOP_TOKEN_IDS, Problem, Sampling, build_user_message
from weight_splice_lab.gpu import patch_mlx_wired_limit

# ---- generation with mlx-lm (vLLM-style sampling)


def make_vllm_sampler(s: Sampling) -> Callable[[mx.array], mx.array]:
    k = s.top_k

    # vLLM order: penalties -> temperature -> top-k -> top-p -> sample
    def sample(logprobs: mx.array) -> mx.array:
        x = logprobs / s.temperature
        idx = mx.argpartition(-x, kth=k - 1, axis=-1)[:, :k]
        vals = mx.take_along_axis(x, idx, axis=-1)
        order = mx.argsort(-vals, axis=-1)
        vals = mx.take_along_axis(vals, order, axis=-1)
        idx = mx.take_along_axis(idx, order, axis=-1)
        probs = mx.softmax(vals, axis=-1)
        before = mx.cumsum(probs, axis=-1) - probs
        vals = mx.where(before < s.top_p, vals, -float("inf"))
        choice = mx.random.categorical(vals, axis=-1)
        return mx.take_along_axis(idx, choice[:, None], axis=-1)[:, 0]

    return sample


def make_presence_penalty(
    prompt_len: int, penalty: float, vocab: int
) -> Callable[[mx.array, mx.array], mx.array]:
    # vLLM applies the presence penalty to generated tokens only (not the prompt) over the whole output;
    # mlx-lm's built-in one uses prompt + the last 20 tokens, so it is not used
    mask = mx.zeros((vocab,), dtype=mx.float32)

    def processor(context: mx.array, logits: mx.array) -> mx.array:
        nonlocal mask
        if context.size > prompt_len:
            mask[context[-1]] = penalty
        return logits - mask.astype(logits.dtype)

    return processor


def load_mlx_model(
    model_dir: Path, edited: dict[str, np.ndarray] | None = None
) -> tuple[object, object]:
    model, tok = load(str(model_dir))
    if edited:
        weights = []
        for name, bits in edited.items():
            mlx_name = name.replace("model.language_model.", "language_model.model.")
            weights.append((mlx_name, mx.array(bf16_to_f32(bits)).astype(mx.bfloat16)))
        model.load_weights(weights, strict=False)
        mx.eval(model.parameters())
        params = dict(_flatten(model.parameters()))
        for mlx_name, w in weights:
            assert mx.array_equal(params[mlx_name], w).item(), mlx_name
    return model, tok


def _flatten(tree: object, prefix: str = "") -> Iterator[tuple[str, object]]:
    if isinstance(tree, dict):
        for k, v in tree.items():
            yield from _flatten(v, f"{prefix}{k}.")
    elif isinstance(tree, list):
        for i, v in enumerate(tree):
            yield from _flatten(v, f"{prefix}{i}.")
    else:
        yield prefix[:-1], tree


def prompt_tokens(tok: object, problem: Problem) -> list[int]:
    text = tok.apply_chat_template(  # type: ignore[attr-defined]
        [{"role": "user", "content": build_user_message(problem.problem)}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return tok.encode(text, add_special_tokens=False)  # type: ignore[attr-defined]


def generate_batch(
    model: object,
    tok: object,
    prompts: list[list[int]],
    s: Sampling,
    batch_size: int,
    on_done: Callable[[int, list[int], str], None],
    vocab: int = 248320,
    stop: bool = True,
) -> dict[str, float]:
    # BatchGenerator otherwise raises the wired limit to the device maximum, overriding gpu_session's cap
    patch_mlx_wired_limit()
    gen = BatchGenerator(
        model,  # type: ignore[arg-type]
        max_tokens=s.max_tokens,
        stop_tokens=[[t] for t in STOP_TOKEN_IDS] if stop else None,
        sampler=make_vllm_sampler(s),
        completion_batch_size=batch_size,
        prefill_batch_size=min(batch_size, 8),
    )
    uids = gen.insert(
        prompts,
        max_tokens=[s.max_tokens] * len(prompts),
        logits_processors=[
            [make_presence_penalty(len(p), s.presence_penalty, vocab)] for p in prompts
        ],
    )
    index = {u: i for i, u in enumerate(uids)}
    out: dict[int, list[int]] = {u: [] for u in uids}
    t0, n_tokens = time.time(), 0
    while responses := gen.next_generated():
        for r in responses:
            n_tokens += 1
            if r.finish_reason != "stop":
                out[r.uid].append(r.token)
            if r.finish_reason is not None:
                on_done(index[r.uid], out.pop(r.uid), r.finish_reason)
    gen.close()
    mx.clear_cache()
    dt = time.time() - t0
    return {"generated_tokens": n_tokens, "seconds": dt, "tokens_per_s": n_tokens / dt}
