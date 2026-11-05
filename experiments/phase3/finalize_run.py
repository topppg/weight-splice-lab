import argparse
import json
from dataclasses import fields
from pathlib import Path

import psutil

from weight_splice_lab.harness.driver import BASE_CACHE_DIR, REPO_RUNS, RunState, finalize
from weight_splice_lab.harness.rundir import RUNS_ROOT, RunConfig, RunPaths

KILLED = "killed: the driver did not finish; finalized from state.json"


def finalize_from_state(
    run_dir: Path, n_per: int, mirror_root: Path, base_cache_dir: Path, force: bool = False
) -> dict[str, object]:
    paths = RunPaths(run_dir)
    state = json.loads((paths.private / "state.json").read_text())
    pid = state.get("pid")
    if isinstance(pid, int) and psutil.pid_exists(pid):
        cmd = " ".join(psutil.Process(pid).cmdline())
        if "run_editor" in cmd or "dry_run" in cmd:
            raise SystemExit(f"the driver (pid {pid}) is still running: {cmd}")
    if (paths.private / "result.json").exists() and not force:
        raise SystemExit(f"{paths.private / 'result.json'} exists; pass --force to redo it")
    names = {f.name for f in fields(RunConfig)}
    raw = json.loads((paths.private / "config.json").read_text())
    cfg = RunConfig(**{k: v for k, v in raw.items() if k in names})
    st = RunState.from_dict(state["state"])
    if st.stop_reason is None:
        st.stop_reason, st.crashed = KILLED, True
    return finalize(cfg, paths, st, n_per, mirror_root, base_cache_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="finalize a killed run from its state.json")
    ap.add_argument("run_id")
    ap.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    ap.add_argument("--eval-n-per", type=int, default=200)
    ap.add_argument("--mirror-root", type=Path, default=REPO_RUNS)
    ap.add_argument("--base-cache-dir", type=Path, default=BASE_CACHE_DIR)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    res = finalize_from_state(
        a.runs_root / a.run_id, a.eval_n_per, a.mirror_root, a.base_cache_dir, a.force
    )
    print(json.dumps({k: res[k] for k in ("run_id", "stop_reason", "evaluated_edit", "scored")}))


if __name__ == "__main__":
    main()
