import argparse
import json
from pathlib import Path

import psutil

from weight_splice_lab.gpu import watchdog_running
from weight_splice_lab.harness.budget import latest_reading
from weight_splice_lab.harness.codex import CodexEditor, CodexSettings, codex_home
from weight_splice_lab.harness.driver import Driver, finalize
from weight_splice_lab.harness.edit import CONDITIONS
from weight_splice_lab.harness.family import FAMILIES
from weight_splice_lab.harness.rundir import RUNS_ROOT, RunConfig, build_run_dir, git_state
from weight_splice_lab.memwatch import PRESSURE_CRITICAL, pressure_level


def preflight(min_available_gb: float) -> None:
    if not watchdog_running():
        raise SystemExit("memwatch is not running; it must be running before a run starts")
    avail = psutil.virtual_memory().available / 1e9
    level = pressure_level()
    if avail < min_available_gb or level >= PRESSURE_CRITICAL:
        raise SystemExit(
            f"memory too tight for a run: {avail:.1f} GB available (need {min_available_gb}), "
            f"kernel memory pressure level {level}"
        )


def check_code(allow_dirty: bool, expect_sha: str | None) -> None:
    g = git_state()
    if g["dirty"] and not allow_dirty:
        raise SystemExit(
            f"the code tree is dirty ({g['dirty_files']}); commit it or pass --allow-dirty"
        )
    if expect_sha is not None and g["git_sha"] != expect_sha:
        raise SystemExit(f"HEAD is {g['git_sha']}, expected {expect_sha}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--family", choices=sorted(FAMILIES), default="toy")
    ap.add_argument("--condition", choices=CONDITIONS, default="add_capacity")
    ap.add_argument("--forward-budget", type=int, default=50)
    ap.add_argument("--python-runs", type=int, default=300)
    ap.add_argument("--max-added-params", type=int, help="default: the family's cap")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--model", default="gpt-6-astra")
    ap.add_argument("--profile", default="personal")
    ap.add_argument("--max-limit-points", type=float, default=5.0)
    ap.add_argument("--max-input-tokens", type=int, default=20_000_000)
    ap.add_argument("--max-turns", type=int, default=200)
    ap.add_argument("--turn-timeout", type=float, default=3600.0)
    ap.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    ap.add_argument("--eval-n-per", type=int, default=200)
    ap.add_argument("--max-attempts", type=int, default=260)
    ap.add_argument("--min-available-gb", type=float, default=6.5)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--kernel-forward", action="store_true", help="the model may run in the kernel")
    ap.add_argument("--kernel-max-rss-gb", type=float, help="default: the family's")
    ap.add_argument("--expect-sha", help="refuse to start unless HEAD is this commit")
    ap.add_argument("--mirror-root", type=Path, help="default: the family's runs/phaseN")
    ap.add_argument("--base-cache-dir", type=Path, help="default: the family's eval cache")
    a = ap.parse_args()
    check_code(a.allow_dirty, a.expect_sha)
    preflight(a.min_available_gb)
    cfg = RunConfig(
        require_watchdog=True,
        run_id=a.run_id,
        family=a.family,
        condition=a.condition,
        forward_budget=a.forward_budget,
        python_runs=a.python_runs,
        max_added_params=a.max_added_params,
        model=a.model,
        profile=a.profile,
        effort=a.effort,
        max_limit_points=a.max_limit_points,
        max_input_tokens=a.max_input_tokens,
        max_turns=a.max_turns,
        max_attempts=a.max_attempts,
        kernel_forward=a.kernel_forward,
        kernel_max_rss_gb=a.kernel_max_rss_gb,
    )
    paths = build_run_dir(cfg, a.runs_root)
    settings = CodexSettings(
        instructions_file=paths.private / "instructions.md",
        profile=cfg.profile,
        model=cfg.model,
        effort=cfg.effort,
        timeout_s=a.turn_timeout,
    )
    pre = latest_reading(codex_home(a.profile) / "sessions")
    driver = Driver(cfg, paths, CodexEditor(settings, paths.sandbox))
    driver.log(f"run {cfg.run_id} in {paths.root}; pre-run weekly usage {pre}")
    crashed = False
    try:
        driver.run(pre)
    except Exception:  # noqa: BLE001 - recorded in the state as "crash: ..."; finalize still runs
        crashed = True
    finally:
        result = finalize(cfg, paths, driver.st, a.eval_n_per, a.mirror_root, a.base_cache_dir)
    keys = ("run_id", "stop_reason", "evaluated_edit", "scored", "compliance", "audit_ok",
            "budget_verified", "integrity_count", "budget", "hidden_eval")  # fmt: skip
    print(json.dumps({k: result[k] for k in keys}, indent=1))
    if crashed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
