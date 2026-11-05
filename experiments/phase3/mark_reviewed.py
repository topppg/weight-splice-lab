import argparse
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from weight_splice_lab.harness.driver import REPO_RUNS
from weight_splice_lab.harness.rundir import ROOT, RUNS_ROOT


def mark_reviewed(
    run_id: str, reviewer: str, review_doc: Path, runs_root: Path, mirror_root: Path
) -> dict[str, object]:
    private = runs_root / run_id / "harness" / "result.json"
    result = json.loads(private.read_text())
    if result.get("compliance") != "pending_review":
        raise SystemExit(
            f"{run_id}: compliance is {result.get('compliance')!r}, not pending_review"
        )
    if not review_doc.is_file():
        raise SystemExit(f"review doc {review_doc} does not exist")
    doc = review_doc.resolve()
    result["compliance"] = "reviewed_ok"
    result["review"] = {
        "reviewer": reviewer,
        "review_doc": str(doc.relative_to(ROOT) if doc.is_relative_to(ROOT) else doc),
        "reviewed_at": time.time(),
    }
    data = json.dumps(result, indent=1)
    private.write_text(data)
    (mirror_root / run_id).mkdir(parents=True, exist_ok=True)
    (mirror_root / run_id / "result.json").write_text(data)
    return result


def release(run_id: str, result: dict[str, object], runs_root: Path, target: str) -> None:
    k = result.get("evaluated_edit")
    if result.get("compliance") != "reviewed_ok" or k is None:
        raise SystemExit(f"{run_id}: nothing to release (compliance or evaluated_edit)")
    edit = runs_root / run_id / "harness" / "edits" / f"edit_{k}" / "edit.pt"
    tmp = Path(tempfile.mkdtemp()) / f"{run_id}_final_edit.pt"
    shutil.copy(edit, tmp)
    review = result["review"]
    assert isinstance(review, dict)
    notes = (
        f"Evaluated edit {k} from {run_id}, reviewed by {review['reviewer']} "
        f"({review['review_doc']}). Apply to runs/phase2/base.pt via weight_splice_lab.harness.edit."
    )
    subprocess.run(
        ["gh", "release", "create", f"checkpoints-{run_id}", str(tmp), "--target", target,
         "--title", f"{run_id} final edit", "--notes", notes],
        check=True,
    )  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description="record a compliance review and release the edit")
    ap.add_argument("run_id")
    ap.add_argument("--reviewer", required=True)
    ap.add_argument("--review-doc", type=Path, required=True)
    ap.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    ap.add_argument("--mirror-root", type=Path, default=REPO_RUNS)
    ap.add_argument("--target", help="commit for the release tag (push the reviewed result first)")
    ap.add_argument("--no-release", action="store_true")
    a = ap.parse_args()
    result = mark_reviewed(a.run_id, a.reviewer, a.review_doc, a.runs_root, a.mirror_root)
    print(f"{a.run_id}: compliance reviewed_ok by {a.reviewer}")
    if not a.no_release:
        if not a.target:
            raise SystemExit("--target <pushed sha> is required to release (or pass --no-release)")
        release(a.run_id, result, a.runs_root, a.target)


if __name__ == "__main__":
    main()
