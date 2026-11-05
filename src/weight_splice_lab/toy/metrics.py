import math
import random
from collections import defaultdict
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from weight_splice_lab.toy.data import (
    NATIVE_TASKS,
    TOKENIZER,
    Example,
    addition_example,
    ladder_example,
)
from weight_splice_lab.toy.model import ToyLM


def free_cache() -> None:
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


@torch.inference_mode()
def text_bpc(model: ToyLM, chunks: Tensor, batch_size: int = 16) -> float:
    device = next(model.parameters()).device
    total, count = 0.0, 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size].to(device)
        logits = model(batch[:, :-1])
        loss = F.cross_entropy(logits.flatten(0, 1), batch[:, 1:].flatten(), reduction="sum")
        total += loss.item()
        count += batch[:, 1:].numel()
        free_cache()
    return total / count / math.log(2)


@torch.inference_mode()
def kl_to_reference(model: ToyLM, reference: ToyLM, chunks: Tensor, batch_size: int = 16) -> float:
    device = next(model.parameters()).device
    total, count = 0.0, 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size].to(device)[:, :-1]
        logp = model(batch).log_softmax(-1)
        logq = reference(batch).log_softmax(-1)
        total += (logq.exp() * (logq - logp)).sum().item()
        count += batch.numel()
        free_cache()
    return total / count


END_OF_ANSWER = "\n"


def target_ids(answer: str, with_terminator: bool) -> list[int]:
    return TOKENIZER.encode(answer + (END_OF_ANSWER if with_terminator else ""), bos=False)


# The one grading rule for practice and hidden eval: generated token ids at fixed positions,
# special tokens included (a PAD or UNK in the answer span is wrong, never skipped).
def grade_ids(generated: list[int], answer: str, with_terminator: bool) -> bool:
    target = target_ids(answer, with_terminator)
    return generated[: len(target)] == target


@torch.inference_mode()
def generate_ids(
    model: ToyLM, prompts: list[list[int]], n_new: list[int], batch_size: int = 64
) -> list[list[int]]:
    device = next(model.parameters()).device
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, (p, n) in enumerate(zip(prompts, n_new, strict=True)):
        groups[(len(p), n)].append(i)
    out: list[list[int]] = [[] for _ in prompts]
    for (plen, n), idxs in groups.items():
        for j in range(0, len(idxs), batch_size):
            chunk = idxs[j : j + batch_size]
            x = torch.tensor([prompts[k] for k in chunk], device=device)
            gen = model.generate(x, n)[:, plen:].tolist()
            for k, ids in zip(chunk, gen, strict=True):
                out[k] = ids
        free_cache()
    return out


def _match(
    model: ToyLM, examples: list[Example], with_terminator: bool, batch_size: int
) -> list[bool]:
    prompts = [TOKENIZER.encode(ex.prompt) for ex in examples]
    n_new = [len(target_ids(ex.answer, with_terminator)) for ex in examples]
    gens = generate_ids(model, prompts, n_new, batch_size)
    return [grade_ids(g, ex.answer, with_terminator) for g, ex in zip(gens, examples, strict=True)]


def exact_match(model: ToyLM, examples: list[Example], batch_size: int = 64) -> list[bool]:
    return _match(model, examples, True, batch_size)


def answer_match(model: ToyLM, examples: list[Example], batch_size: int = 64) -> list[bool]:
    return _match(model, examples, False, batch_size)


def accuracy(flags: list[bool]) -> float:
    return sum(flags) / max(len(flags), 1)


def native_task_accuracy(
    model: ToyLM, lengths: range = range(1, 13), n_per: int = 100, seed: int = 1234
) -> dict[str, float]:
    rng = random.Random(seed)
    result: dict[str, float] = {}
    for name, task in NATIVE_TASKS.items():
        examples = [task(rng, n) for n in lengths for _ in range(n_per)]
        result[name] = accuracy(exact_match(model, examples))
    return result


def target_accuracy(
    model: ToyLM,
    lengths: list[int],
    n_per: int = 200,
    seed: int = 4321,
    make: Callable[[random.Random, int], Example] = addition_example,
    with_terminator: bool = False,
) -> dict[int, float]:
    rng = random.Random(seed)
    result: dict[int, float] = {}
    match_fn = exact_match if with_terminator else answer_match
    for n in lengths:
        examples = [make(rng, n) for _ in range(n_per)]
        result[n] = accuracy(match_fn(model, examples))
    return result


def ladder_accuracy(
    model: ToyLM, rung: str, lengths: list[int], n_per: int = 200
) -> dict[int, float]:
    return target_accuracy(model, lengths, n_per, make=lambda r, n: ladder_example(rung, r, n))


def regression_report(model: ToyLM, reference: ToyLM, text_chunks: Tensor) -> dict[str, float]:
    report = {
        "text_bpc": text_bpc(model, text_chunks),
        "text_kl": kl_to_reference(model, reference, text_chunks),
    }
    report |= {f"task_{k}": v for k, v in native_task_accuracy(model).items()}
    return report


@torch.inference_mode()
def continuations(model: ToyLM, prompts: list[str], n_new: int, batch_size: int = 64) -> list[str]:
    device = next(model.parameters()).device
    groups: dict[int, list[int]] = defaultdict(list)
    for i, p in enumerate(prompts):
        groups[len(p)].append(i)
    out = [""] * len(prompts)
    for idxs in groups.values():
        for j in range(0, len(idxs), batch_size):
            chunk = idxs[j : j + batch_size]
            ids = torch.tensor([TOKENIZER.encode(prompts[k]) for k in chunk], device=device)
            gen = model.generate(ids, n_new)[:, ids.shape[1] :].tolist()
            for k, g in zip(chunk, gen, strict=True):
                out[k] = TOKENIZER.decode(g)
        free_cache()
    return out


def termination(
    model: ToyLM, lengths: list[int], n_per: int, seed: int, end: str = END_OF_ANSWER
) -> dict[str, dict[int, float]]:
    rng = random.Random(seed)
    end_id = TOKENIZER.id(end)
    terminated: dict[int, float] = {}
    strict: dict[int, float] = {}
    for n in lengths:
        examples = [addition_example(rng, n) for _ in range(n_per)]
        prompts = [TOKENIZER.encode(e.prompt) for e in examples]
        gens = generate_ids(model, prompts, [n + 2] * len(prompts))
        # position n+1 is the token right after the N+1 answer tokens, specials included
        terminated[n] = accuracy([g[n + 1] == end_id for g in gens])
        strict[n] = accuracy(
            [
                g == target_ids(e.answer, False) + [end_id]
                for g, e in zip(gens, examples, strict=True)
            ]
        )
    return {"terminated": terminated, "strict": strict}


def prefixed_target_accuracy(
    model: ToyLM,
    lengths: list[int],
    n_per: int,
    seed: int,
    prefix: Callable[[random.Random], str],
) -> dict[int, float]:
    rng = random.Random(seed)
    result: dict[int, float] = {}
    for n in lengths:
        examples = []
        for _ in range(n_per):
            ex = addition_example(rng, n)
            examples.append(Example(prefix(rng) + ex.prompt, ex.answer))
        result[n] = accuracy(answer_match(model, examples))
    return result


def native_weight_change(model: ToyLM, base: ToyLM) -> dict[str, object]:
    state = model.state_dict()
    per_tensor: dict[str, float] = {}
    delta_sq = base_sq = 0.0
    for name, w0 in base.state_dict().items():
        w = state[name][tuple(slice(0, s) for s in w0.shape)]
        d = (w - w0).pow(2).sum().item()
        b = w0.pow(2).sum().item()
        delta_sq += d
        base_sq += b
        if d > 0:
            per_tensor[name] = math.sqrt(d / b)
    return {
        "total_rel": math.sqrt(delta_sq / base_sq),
        "max_rel": max(per_tensor.values(), default=0.0),
        "changed_tensors": len(per_tensor),
        "n_tensors": len(base.state_dict()),
        "per_tensor_rel": per_tensor,
    }


def added_param_report(model: ToyLM, base: ToyLM) -> dict[str, int]:
    units: list[tuple[Tensor, list[Tensor]]] = []
    for blk, ref in zip(model.blocks, base.blocks, strict=True):
        attn, mlp, hd = blk.attn, blk.mlp, blk.attn.head_dim
        for h in range(ref.attn.n_heads, attn.n_heads):
            sl = slice(h * hd, (h + 1) * hd)
            qkv = [attn.q_proj.weight[sl], attn.k_proj.weight[sl], attn.v_proj.weight[sl]]
            units.append((attn.o_proj.weight[:, sl], qkv))
        for j in range(ref.mlp.gate_proj.out_features, mlp.gate_proj.out_features):
            units.append(
                (mlp.down_proj.weight[:, j], [mlp.gate_proj.weight[j], mlp.up_proj.weight[j]])
            )
    report = dict.fromkeys(("allocated", "nonzero", "inactive", "active", "nonzero_active"), 0)
    for out, ins in units:
        size = out.numel() + sum(t.numel() for t in ins)
        nz = int(out.count_nonzero()) + sum(int(t.count_nonzero()) for t in ins)
        live = bool(out.any())
        report["allocated"] += size
        report["nonzero"] += nz
        report["active" if live else "inactive"] += size
        report["nonzero_active"] += nz if live else 0
    return report
