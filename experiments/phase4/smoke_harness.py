import json
import sys
import tempfile
import time
from pathlib import Path

from weight_splice_lab.gpu import available_gb
from weight_splice_lab.harness import family
from weight_splice_lab.harness.driver import Driver
from weight_splice_lab.harness.fake import ScriptedEditor
from weight_splice_lab.harness.rundir import RunConfig, build_run_dir

# A scripted editor drives the real driver, kernel and forward tool on Gemma: no Codex, no eval.

CELL = """```python
# the prelude loaded: shapes, names, tokenizer; stage one added head with a random write
print(len(NAMES), shape('blocks.0.attn.q_proj.weight'), shape('blocks.0.attn.k_proj.weight'), CFG['n_layers'])
print(tok.encode('12+34='), len(tok.digit_ids))
new = add_heads(3, 1)
print(new, shape('blocks.3.attn.k_proj.weight'), shape('blocks.3.attn.o_proj.weight'), added_params())
w = load('blocks.3.attn.o_proj.weight'); w[:, 1024:] = 0.001
set_tensor('blocks.3.attn.o_proj.weight', w)
q = load('blocks.3.attn.q_proj.weight'); q[1024:] = load('blocks.3.attn.q_proj.weight')[:256]
set_tensor('blocks.3.attn.q_proj.weight', q)
k = load('blocks.3.attn.k_proj.weight'); k[256:] = k[:256]; set_tensor('blocks.3.attn.k_proj.weight', k)
v = load('blocks.3.attn.v_proj.weight'); v[256:] = v[:256]; set_tensor('blocks.3.attn.v_proj.weight', v)
staged()
```"""
FORBIDDEN = """```python
# running the model in the kernel must be refused
m = gemma_model.GemmaLM(gemma_model.GemmaConfig(vocab_size=10, d_model=8, n_layers=1, head_dim=4, d_mlp=8, layer_types=['full']))
import torch
m(torch.zeros(1, 2, dtype=torch.long))
```"""
SAVE = '```save_edit\n{"final": false, "note": "smoke"}\n```'
GEN = json.dumps(
    {"type": "generate", "model": "both", "ids": [f"add_n{n:02d}_000" for n in range(1, 21)],
     "prompts": ["<start_of_turn>user\nWhat is 2 + 3?<end_of_turn>\n<start_of_turn>model\n"]}
)  # fmt: skip
TF = json.dumps(
    {"type": "teacher_forced", "model": "both", "texts": ["The cat sat on the mat.", "12+34=46"],
     "return": {"topk": 5, "sites": ["resid_pre.3", "attn.3.pattern", "attn.3.head_out", "mlp.3.hidden"]}}
)  # fmt: skip
FINAL = '```save_edit\n{"final": true, "end": true, "note": "smoke final"}\n```'


def main() -> None:
    if available_gb() < 4.0:
        raise SystemExit(f"only {available_gb():.1f} GB available")
    root = Path(tempfile.mkdtemp(prefix="gemma-smoke-"))
    chat = root / "chat_practice.jsonl"
    chat.write_text(json.dumps({"id": "c0", "instruction": "Hi", "prompt_text": "p",
                                "response_text": "r", "prompt_ids": [2], "response_ids": [3]}) + "\n")  # fmt: skip
    family.CHAT_PRACTICE = chat
    cfg = RunConfig(run_id="smoke", family="gemma", forward_budget=3, max_turns=12)
    paths = build_run_dir(cfg, root, text="The cat sat on the mat. " * 50)
    replies = [CELL, FORBIDDEN, SAVE, f"```forward\n{GEN}\n```", f"```forward\n{TF}\n```", FINAL]
    t0 = time.time()
    driver = Driver(cfg, paths, ScriptedEditor(replies))
    st = driver.run(None)
    calls = [json.loads(x) for x in (paths.private / "tool_calls.jsonl").read_text().splitlines()]
    for c in calls:
        print(f"--- {c['tool']} ok={c.get('ok')}\n{json.dumps(c)[:1800]}")
    print(f"stop: {st.stop_reason}; passes {st.passes}; {time.time() - t0:.0f} s; dir {paths.root}")
    sys.exit(0 if st.stop_reason and st.stop_reason.startswith("done:") else 1)


if __name__ == "__main__":
    main()
