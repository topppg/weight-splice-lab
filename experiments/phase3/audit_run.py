import argparse
import json

from weight_splice_lab.harness.audit import audit_run
from weight_splice_lab.harness.driver import REPO_RUNS, mirror
from weight_splice_lab.harness.rundir import RUNS_ROOT, RunPaths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    a = ap.parse_args()
    paths = RunPaths(RUNS_ROOT / a.run_id)
    cfg = json.loads((paths.private / "config.json").read_text())
    audit = audit_run(paths.private, bool(cfg.get("kernel_forward")))
    (paths.private / "audit.json").write_text(json.dumps(audit, indent=1))
    result_file = paths.private / "result.json"
    if result_file.exists():
        result = json.loads(result_file.read_text())
        result["audit"] = audit | {"findings": audit["findings"][:200]}
        result_file.write_text(json.dumps(result, indent=1))
    mirror(paths, REPO_RUNS / a.run_id)
    print(json.dumps({k: v for k, v in audit.items() if k != "findings"}, indent=1))
    for f in audit["findings"][:50]:
        print(f)


if __name__ == "__main__":
    main()
