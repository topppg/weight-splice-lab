import json
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from weight_splice_lab.harness.family import TOY, Family
from weight_splice_lab.harness.rundir import open_dir, write_atomic, write_regular
from weight_splice_lab.toy.metrics import generate_ids

PAD = 0  # both tokenizers


class ForwardError(ValueError):
    pass


@dataclass(frozen=True)
class PracticeItem:
    id: str
    task: str
    prompt: str
    answer: str
    n_digits: int


@dataclass
class ForwardLimits:
    max_items: int = 2048
    max_texts: int = 256
    max_chars: int = 2048
    max_tokens: int = 257  # including <bos>
    max_new_tokens: int = 64
    max_topk: int = 20
    max_bytes: int = 768 * 2**20
    batch_size: int = 64
    tf_batch_size: int = 8


MODELS = ("original", "edited", "both")
PADDING_NOTE = (
    "right-padded; every array is 0 at padding positions t >= length (token_ids: PAD = 0); "
    "token_logprobs is also 0 at t = length - 1 (no next token); attn.L.pattern is 0 for padded "
    "query rows and key columns"
)
_SITE = re.compile(
    r"^(resid_pre\.\d+|resid_mid\.\d+|resid_final|attn\.\d+\.pattern|attn\.\d+\.head_out"
    r"|mlp\.\d+\.hidden|logits)$"
)


def load_practice(practice_dir: Path) -> dict[str, PracticeItem]:
    items: dict[str, PracticeItem] = {}
    for name in ("addition.jsonl", "native.jsonl", "plain.jsonl"):
        if not (practice_dir / name).exists():
            continue
        for line in (practice_dir / name).read_text().splitlines():
            d = json.loads(line)
            items[d["id"]] = PracticeItem(
                d["id"], d["task"], d["prompt"], d["answer"], d["n_digits"]
            )
    return items


def _accuracy_table(records: list[dict[str, object]]) -> dict[str, object]:
    by_task: dict[str, list[bool]] = defaultdict(list)
    by_n: dict[int, list[bool]] = defaultdict(list)
    for r in records:
        if r.get("correct") is None:
            continue
        by_task[str(r["task"])].append(bool(r["correct"]))
        if r["task"] == "addition":
            by_n[int(str(r["n_digits"]))].append(bool(r["correct"]))
    graded = [c for v in by_task.values() for c in v]
    return {
        "graded": len(graded),
        "accuracy": round(sum(graded) / len(graded), 4) if graded else None,
        "accuracy_by_task": {k: round(sum(v) / len(v), 4) for k, v in sorted(by_task.items())},
        "addition_accuracy_by_n_digits": {
            n: round(sum(v) / len(v), 4) for n, v in sorted(by_n.items())
        },
    }


class ForwardServer:
    def __init__(
        self,
        base: nn.Module,
        practice: dict[str, PracticeItem],
        work: Path,
        limits: ForwardLimits,
        family: Family = TOY,
    ) -> None:
        self.base = base.eval()
        self.edited: nn.Module | None = None
        self.practice = practice
        self.work = work
        self.limits = limits
        self.family = family
        self.tok = family.tokenizer()

    def _encode(self, text: str) -> list[int]:
        if len(text) > self.limits.max_chars:
            raise ForwardError(
                f"texts and prompts may have at most {self.limits.max_chars} characters"
            )
        ids = self.tok.encode(text)
        if len(ids) > self.limits.max_tokens:
            raise ForwardError(
                f"texts and prompts may have at most {self.limits.max_tokens - 1} tokens "
                f"(this one has {len(ids) - 1})"
            )
        return ids

    def score(self, item: PracticeItem, ids: list[int]) -> bool:
        # the hidden eval's grading rule, applied to token ids
        return self.family.grade(self.tok, ids, item.task, item.answer)

    def _models(self, which: object) -> list[tuple[str, nn.Module]]:
        if which not in MODELS:
            raise ForwardError(f'"model" must be one of {list(MODELS)}')
        if which in ("edited", "both") and self.edited is None:
            raise ForwardError("no edit has been saved yet: use save_edit first")
        out = [("original", self.base)] if which in ("original", "both") else []
        if which in ("edited", "both") and self.edited is not None:
            out.append(("edited", self.edited))
        return out

    def run(
        self, req: dict[str, object], k: int, started: Callable[[], None] = lambda: None
    ) -> dict[str, object]:
        # `started` is called once the request is valid and inference is about to begin
        kind = req.get("type")
        if kind == "generate":
            return self.generate(req, k, started)
        if kind == "teacher_forced":
            return self.teacher_forced(req, k, started)
        raise ForwardError('"type" must be "generate" or "teacher_forced"')

    def generate(
        self, req: dict[str, object], k: int, started: Callable[[], None] = lambda: None
    ) -> dict[str, object]:
        unknown = set(req) - {"type", "model", "ids", "prompts", "max_new_tokens"}
        if unknown:
            raise ForwardError(f"unknown keys {sorted(unknown)}")
        models = self._models(req.get("model"))
        ids = req.get("ids", [])
        prompts = req.get("prompts", [])
        if not isinstance(ids, list) or not isinstance(prompts, list):
            raise ForwardError('"ids" and "prompts" must be lists')
        missing = [i for i in ids if i not in self.practice]
        if missing:
            raise ForwardError(f"unknown practice ids: {missing[:5]}")
        n_items = len(ids) + len(prompts)
        if n_items == 0 or n_items > self.limits.max_items:
            raise ForwardError(f"between 1 and {self.limits.max_items} ids + prompts per pass")
        max_new = req.get("max_new_tokens")
        if max_new is not None and (
            not isinstance(max_new, int) or not 1 <= max_new <= self.limits.max_new_tokens
        ):
            raise ForwardError(f'"max_new_tokens" must be 1..{self.limits.max_new_tokens}')
        items = [self.practice[str(i)] for i in ids]
        enc = [self._encode(it.prompt) for it in items]
        enc += [self._encode(str(p)) for p in prompts]
        n_new = [max_new or len(it.answer) + 1 for it in items] + [max_new or 32] * len(prompts)
        records: dict[str, list[dict[str, object]]] = {}
        summary: dict[str, object] = {
            "type": "generate",
            "model": req.get("model"),
            "items": n_items,
        }
        started()
        for name, model in models:
            outs = generate_ids(model, enc, n_new, self.limits.batch_size)
            recs: list[dict[str, object]] = []
            for it, out in zip(items, outs, strict=False):
                recs.append(
                    {"id": it.id, "task": it.task, "n_digits": it.n_digits, "prompt": it.prompt,
                     "answer": it.answer, "output": self.tok.show(out), "output_ids": out,
                     "correct": self.score(it, out)}
                )  # fmt: skip
            for p, out in zip(prompts, outs[len(items) :], strict=True):
                recs.append(
                    {"id": None, "task": "custom", "prompt": p, "output": self.tok.show(out),
                     "output_ids": out, "correct": None}
                )  # fmt: skip
            records[name] = recs
            table = _accuracy_table(recs)
            table["failures"] = [
                {"id": r["id"], "expected": r["answer"], "got": r["output"]}
                for r in recs if r["correct"] is False
            ][:5]  # fmt: skip
            if prompts:
                table["custom_outputs"] = [r["output"] for r in recs if r["task"] == "custom"][:8]
            summary[name] = table
        files = []
        for name, recs in records.items():
            fname = f"fwd_{k}.jsonl" if len(records) == 1 else f"fwd_{k}_{name}.jsonl"
            write_regular(self.work, fname, "".join(json.dumps(r) + "\n" for r in recs).encode())
            files.append(f"work/{fname}")
        summary["files"] = files
        if len(records) == 2:
            pairs = list(zip(records["original"], records["edited"], strict=True))
            graded = [(a["correct"], b["correct"]) for a, b in pairs if a["correct"] is not None]
            summary["paired"] = {
                "rescues": sum(1 for a, b in graded if not a and b),
                "losses": sum(1 for a, b in graded if a and not b),
                "outputs_changed": sum(1 for a, b in pairs if a["output"] != b["output"]),
            }
        return summary

    def teacher_forced(
        self, req: dict[str, object], k: int, started: Callable[[], None] = lambda: None
    ) -> dict[str, object]:
        unknown = set(req) - {"type", "model", "texts", "return"}
        if unknown:
            raise ForwardError(f"unknown keys {sorted(unknown)}")
        models = self._models(req.get("model"))
        texts = req.get("texts")
        if not isinstance(texts, list) or not 1 <= len(texts) <= self.limits.max_texts:
            raise ForwardError(f'"texts" must be a list of 1..{self.limits.max_texts} strings')
        ret = req.get("return", {})
        if not isinstance(ret, dict):
            raise ForwardError('"return" must be an object')
        unknown = set(ret) - {"topk", "sites"}
        if unknown:
            raise ForwardError(f"unknown return keys {sorted(unknown)}")
        topk = int(ret.get("topk", 0))
        if not 0 <= topk <= self.limits.max_topk:
            raise ForwardError(f'"topk" must be 0..{self.limits.max_topk}')
        sites = ret.get("sites", [])
        if not isinstance(sites, list):
            raise ForwardError('"sites" must be a list')
        enc = [self._encode(str(t)) for t in texts]
        lengths = [len(e) for e in enc]
        t_max = max(lengths)
        for s in sites:
            m = _SITE.match(str(s))
            layer = re.search(r"\.(\d+)", str(s))
            if m is None or (layer and int(layer.group(1)) >= len(self.base.blocks)):
                raise ForwardError(f"unknown site {s}")
        nbytes = sum(self._site_bytes(m, str(s), len(enc), t_max) for _, m in models for s in sites)
        nbytes += len(models) * len(enc) * t_max * (4 + 8 * topk)
        nbytes += 4 * len(enc) * t_max if len(models) == 2 else 0
        if nbytes > self.limits.max_bytes:
            raise ForwardError(
                f"the requested arrays would take {nbytes / 2**20:.0f} MB "
                f"(limit {self.limits.max_bytes / 2**20:.0f} MB): fewer texts, sites or layers"
            )
        out_dir = f"fwd_{k}"
        summary: dict[str, object] = {"type": "teacher_forced", "model": req.get("model")}
        started()
        arrays, nlls, kls = self._capture(models, enc, [str(s) for s in sites], topk)
        for name, _ in models:
            shapes = {}
            with open_dir(self.work, [out_dir, name], create=True) as dfd:
                for key in list(arrays[name]):
                    arr = arrays[name].pop(key)
                    with write_atomic(dfd, f"{key}.npy") as f:
                        np.save(f, arr)
                    shapes[key] = list(arr.shape)
                    del arr
            nll = nlls[name]
            summary[name] = {
                "dir": f"work/fwd_{k}/{name}",
                "arrays": shapes,
                "mean_bits_per_token": round(float(np.mean(nll)) / np.log(2), 4) if nll else None,
            }
        if len(models) == 2:
            summary["kl_original_to_edited"] = {
                "mean": round(float(np.mean(kls)), 6),
                "max_text": round(float(np.max(kls)), 6),
                "per_text_file": f"work/fwd_{k}/kl_per_text.npy",
            }
            with open_dir(self.work, [out_dir]) as dfd, write_atomic(dfd, "kl_per_text.npy") as f:
                np.save(f, np.array(kls, dtype=np.float32))
        meta = {"texts": texts, "token_ids": enc, "lengths": lengths, "padding": PADDING_NOTE,
                "positions": "position 0 is the BOS token; position t predicts token t+1"}  # fmt: skip
        write_regular(self.work, f"{out_dir}/meta.json", json.dumps(meta).encode())
        summary["meta"] = f"work/fwd_{k}/meta.json"
        return summary

    def _site_bytes(self, model: nn.Module, site: str, n: int, t: int) -> int:
        cfg = model.cfg
        m = re.search(r"\.(\d+)", site)
        layer = int(m.group(1)) if m else 0
        attn, mlp = model.blocks[layer].attn, model.blocks[layer].mlp
        per = {
            "pattern": attn.n_heads * t * t,
            "head_out": attn.n_heads * t * attn.head_dim,
            "hidden": t * mlp.gate_proj.out_features,
            "logits": t * cfg.vocab_size,
        }
        return 4 * n * per.get(site.split(".")[-1], t * cfg.d_model)

    @torch.inference_mode()
    def _capture(
        self, models: list[tuple[str, nn.Module]], enc: list[list[int]], sites: list[str], topk: int
    ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, list[float]], list[float]]:
        # Per batch: every model, then the per-text KL, then the full log-probs are dropped
        # (Gemma: 262,144 x 4 bytes per position).
        n, t_max = len(enc), max(len(e) for e in enc)
        parts: dict[str, dict[str, list[np.ndarray]]] = {
            name: defaultdict(list) for name, _ in models
        }
        nlls: dict[str, list[float]] = {name: [] for name, _ in models}
        kls: list[float] = []
        kl_pos: list[np.ndarray] = []
        bs = self.limits.tf_batch_size
        for i in range(0, n, bs):
            batch = enc[i : i + bs]
            x = torch.full((len(batch), t_max), PAD, dtype=torch.long)
            for r, e in enumerate(batch):
                x[r, : len(e)] = torch.tensor(e)
            nxt = torch.cat([x[:, 1:], torch.full((len(batch), 1), PAD)], dim=1)
            logps: list[torch.Tensor] = []
            for name, model in models:
                cap: dict[str, torch.Tensor] = {}
                logits = model(x, capture=cap if sites else None)
                for s in sites:
                    arr = logits if s == "logits" else cap[s]
                    parts[name][s].append(arr.numpy().astype(np.float32))
                del cap
                logp = logits.log_softmax(-1)
                del logits
                tok_lp = logp.gather(-1, nxt[..., None])[..., 0]
                parts[name]["token_logprobs"].append(tok_lp.numpy().astype(np.float32))
                for r, e in enumerate(batch):
                    if len(e) > 1:
                        nlls[name].append(-float(tok_lp[r, : len(e) - 1].mean()))
                if topk:
                    v, idx = logp.topk(topk, dim=-1)
                    parts[name]["topk_logprobs"].append(v.numpy())
                    parts[name]["topk_ids"].append(idx.numpy().astype(np.int32))
                logps.append(logp)
            if len(logps) == 2:
                p, q = logps
                per_pos = (p.exp() * (p - q)).sum(-1)
                for r, e in enumerate(batch):
                    kls.append(float(per_pos[r, : len(e)].mean()))
                kl_pos.append(per_pos.numpy().astype(np.float32))
            del logps
        lengths = np.array([len(e) for e in enc])
        valid = np.arange(t_max)[None, :] < lengths[:, None]
        arrays: dict[str, dict[str, np.ndarray]] = {}
        for name, _ in models:
            arrays[name] = {key: np.concatenate(parts[name].pop(key)) for key in list(parts[name])}
            for key, arr in arrays[name].items():
                if key == "token_logprobs":
                    arr *= np.arange(t_max)[None, :] < (lengths[:, None] - 1)
                elif key.endswith(".pattern"):
                    arr *= valid[:, None, :, None] & valid[:, None, None, :]
                elif key.endswith(".head_out"):
                    arr *= valid[:, None, :, None]
                else:
                    arr *= valid[:, :, None]
            arrays[name]["token_ids"] = np.array(
                [e + [PAD] * (t_max - len(e)) for e in enc], dtype=np.int32
            )
        if kl_pos:
            # position t: KL(original || edited) of the prediction of token t+1
            arrays["edited"]["kl_per_position"] = np.concatenate(kl_pos) * valid
        return arrays, nlls, kls
