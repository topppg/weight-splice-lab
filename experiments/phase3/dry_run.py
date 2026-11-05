import argparse
import json
import time

from weight_splice_lab.harness.driver import Driver, finalize
from weight_splice_lab.harness.fake import ScriptedEditor
from weight_splice_lab.harness.rundir import ROOT, RUNS_ROOT, RunConfig, build_run_dir

IDS = [f"add_n{n:02d}_{i:03d}" for n in (1, 2, 3) for i in range(4)] + [
    f"{t}_n04_{i:03d}" for t in ("copy", "reverse", "max") for i in range(2)
]


STAGE = """# Stage a trivial edit: two zero neurons in layer 3 and a tiny change to one column
new = add_neurons(3, 2)
w = load('blocks.5.mlp.down_proj.weight')
w[:, 0] += 1e-3
set_tensor('blocks.5.mlp.down_proj.weight', w)
print(new, staged(), added_params())"""

GUARDS = f"""# Check that the kernel guards hold (each attempt is expected to fail)
for f in (lambda: torch.ones(2).requires_grad_(), lambda: torch.optim.SGD,
          lambda: open({str(ROOT / "README.md")!r}).read()):
    try:
        f()
    except Exception as e:
        print(type(e).__name__, str(e)[:80])"""


def block(tool: str, body: str) -> str:
    return f"Explanation of the next step.\n\n```{tool}\n{body}\n```"


def script() -> list[str]:
    fwd = {"type": "generate", "model": "original", "ids": IDS}
    tf = {
        "type": "teacher_forced",
        "model": "both",
        "texts": ["The cat sat on the mat.", "A:398+259=5481", "C:123=123"],
        "return": {"topk": 3, "sites": ["resid_pre.2", "attn.1.pattern", "mlp.3.hidden"]},
    }
    return [
        block(
            "python",
            "# Inspect the setup\nprint(len(NAMES), arch(), CONDITION)\nshape('blocks.0.attn.q_proj.weight')",
        ),
        block("forward", json.dumps(fwd)),
        "I will think about this before acting.",
        block("python", STAGE),
        block("save_edit", json.dumps({"final": False, "note": "dry run: trivial edit"})),
        block("forward", json.dumps(fwd | {"model": "both"})),
        block("forward", json.dumps(tf)),
        block("python", GUARDS),
        block("save_edit", json.dumps({"final": True, "note": "dry run: final trivial edit"})),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=time.strftime("dryrun-%Y%m%d-%H%M%S"))
    ap.add_argument("--eval-n-per", type=int, default=20)
    a = ap.parse_args()
    cfg = RunConfig(run_id=a.run_id, forward_budget=3, max_turns=20, kernel_timeout_s=120)
    paths = build_run_dir(cfg, RUNS_ROOT)
    driver = Driver(cfg, paths, ScriptedEditor(script(), points_per_turn=0.1))
    state = driver.run()
    result = finalize(cfg, paths, state, n_per=a.eval_n_per)
    keys = ("run_id", "stop_reason", "evaluated_edit", "counters", "budget", "hidden_eval")
    print(json.dumps({k: result[k] for k in keys}, indent=1))
    print(json.dumps({k: v for k, v in result["audit"].items() if k != "findings"}, indent=1))


if __name__ == "__main__":
    main()
