import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "experiments" / "phase3" / "overnight.sh"
pytestmark = pytest.mark.skipif(shutil.which("zsh") is None, reason="needs zsh")


def git(cwd: Path, *args: str) -> str:
    env = os.environ | {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                        "GIT_COMMITTER_EMAIL": "t@t"}  # fmt: skip
    return subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True,
                          text=True).stdout.strip()  # fmt: skip


@pytest.fixture
def world(tmp_path: Path) -> dict[str, Path]:
    remote, repo, other = tmp_path / "remote.git", tmp_path / "repo", tmp_path / "other"
    git(tmp_path, "init", "-q", "--bare", "-b", "main", str(remote))
    git(tmp_path, "clone", "-q", str(remote), str(repo))
    (repo / "code.py").write_text("x = 1\n")
    (repo / "runs" / "phase3").mkdir(parents=True)
    (repo / "runs" / "phase3" / "keep.txt").write_text("k\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "init")
    git(repo, "push", "-q", "origin", "HEAD:main")
    git(tmp_path, "clone", "-q", str(remote), str(other))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text(f'#!/bin/sh\necho "$@" >> {tmp_path / "gh.log"}\n')
    gh.chmod(0o755)
    return {"tmp": tmp_path, "remote": remote, "repo": repo, "other": other, "bin": bin_dir,
            "runs": tmp_path / "wsl", "pub": tmp_path / "pub"}  # fmt: skip


def fake_run(w: dict[str, Path], run_id: str, sha: str, scored: bool = True) -> None:
    h = w["runs"] / run_id / "harness"
    (h / "edits" / "edit_2").mkdir(parents=True)
    (h / "edits" / "edit_2" / "edit.pt").write_bytes(b"edit")
    result = {"run_id": run_id, "harness_version": "v4", "finalized_at": 1.0, "scored": scored,
              "compliance": "pending_review",
              "crashed": False, "evaluated_edit": 2,
              "provenance": {"git_sha": sha, "dirty": False}}  # fmt: skip
    (h / "result.json").write_text(json.dumps(result))
    mirror = w["pub"] / "runs" / "phase3" / run_id
    mirror.mkdir(parents=True)
    (mirror / "result.json").write_text(json.dumps(result))
    (mirror / "turns.jsonl").write_text("{}\n")


def run_lib(w: dict[str, Path], commands: str) -> subprocess.CompletedProcess[str]:
    env = os.environ | {
        "OVERNIGHT_LIB": "1", "OVERNIGHT_REPO": str(w["repo"]), "OVERNIGHT_PY": sys.executable,
        "OVERNIGHT_RUNS_ROOT": str(w["runs"]), "OVERNIGHT_PUBLISH_DIR": str(w["pub"]),
        "OVERNIGHT_PUSH_WAIT": "0", "PATH": f"{w['bin']}:{os.environ['PATH']}",
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }  # fmt: skip
    script = f"source {SCRIPT}\n{commands}\n"
    return subprocess.run(
        ["zsh", "-c", script], env=env, capture_output=True, text=True, check=False
    )


def test_publish_commits_only_the_run_dir_and_never_releases(
    world: dict[str, Path],
) -> None:
    w = world
    pin = git(w["repo"], "rev-parse", "HEAD")
    # someone else pushes meanwhile; publishing must not move the pinned checkout
    (w["other"] / "theirs.txt").write_text("t\n")
    git(w["other"], "add", ".")
    git(w["other"], "commit", "-q", "-m", "theirs")
    p = run_lib(w, "pin && echo PIN=$PIN")
    assert p.returncode == 0, p.stdout + p.stderr
    git(w["other"], "push", "-q", "origin", "HEAD:main")
    fake_run(w, "pilotx", pin)
    (w["pub"] / "stray.txt").write_text("s\n")
    git(w["pub"], "add", "stray.txt")  # something else staged must not ride along
    p = run_lib(w, f"PIN={pin}; publish pilotx")
    assert p.returncode == 0, p.stdout + p.stderr
    remote_head = git(w["remote"], "rev-parse", "main")
    files = git(w["remote"], "show", "--name-only", "--format=", remote_head).split()
    assert sorted(files) == ["runs/phase3/pilotx/result.json", "runs/phase3/pilotx/turns.jsonl"]
    assert (
        git(w["remote"], "log", "-1", "--format=%s", "main")
        == "feat: add pilotx run logs and hidden eval results"
    )
    assert "theirs" in git(w["remote"], "log", "--format=%s", "main")
    assert git(w["repo"], "rev-parse", "HEAD") == pin
    assert not (w["tmp"] / "gh.log").exists()  # releases only via mark_reviewed.py


def test_invalid_or_unscored_results_are_not_released(world: dict[str, Path]) -> None:
    w = world
    pin = git(w["repo"], "rev-parse", "HEAD")
    assert run_lib(w, "pin").returncode == 0
    fake_run(w, "wrongsha", "0" * 40)
    p = run_lib(w, f"PIN={pin}; publish wrongsha")
    assert p.returncode != 0 and "git_sha" in p.stdout + p.stderr
    assert git(w["remote"], "rev-parse", "main") == pin
    fake_run(w, "unscored", pin, scored=False)
    p = run_lib(w, f"PIN={pin}; publish unscored")
    assert p.returncode == 0 and "no release" in p.stdout
    assert not (w["tmp"] / "gh.log").exists()


def test_failed_launch_stops_the_chain_and_an_old_run_dir_is_refused(
    world: dict[str, Path],
) -> None:
    w = world
    pin = git(w["repo"], "rev-parse", "HEAD")
    assert run_lib(w, "pin").returncode == 0
    fake_run(w, "old", pin)
    p = run_lib(w, f"PIN={pin}; run_pilot old add_capacity")
    assert p.returncode != 0 and "already exists" in p.stdout
    assert git(w["remote"], "rev-parse", "main") == pin
    # the editor writes its run and exits non-zero: logs are published, then the chain stops
    fake_run(w, "crashy", pin)
    (w["runs"] / "crashy").rename(w["tmp"] / "crashy")
    stub = f"wait_ready() {{ return 0; }}; uv() {{ mv {w['tmp'] / 'crashy'} $RUNS/; return 3; }}"
    p = run_lib(w, f"PIN={pin}; {stub}; run_pilot crashy add_capacity")
    assert p.returncode != 0 and "rc=3" in p.stdout and "chain stopped" in p.stdout
    assert git(w["remote"], "rev-parse", "main") != pin
    assert not (w["tmp"] / "gh.log").exists()


def test_mark_reviewed_records_the_review_and_releases(world: dict[str, Path]) -> None:
    w = world
    fake_run(w, "rev", "a" * 40)
    doc = w["tmp"] / "review.md"
    doc.write_text("ok\n")
    script = SCRIPT.parent / "mark_reviewed.py"
    env = os.environ | {"PATH": f"{w['bin']}:{os.environ['PATH']}"}
    args = [sys.executable, str(script), "rev", "--reviewer", "gs", "--review-doc", str(doc),
            "--runs-root", str(w["runs"]), "--mirror-root", str(w["pub"] / "runs" / "phase3")]  # fmt: skip
    p = subprocess.run(
        [*args, "--target", "abc123"], check=False, env=env, capture_output=True, text=True
    )
    assert p.returncode == 0, p.stderr
    for f in (w["runs"] / "rev" / "harness" / "result.json",
              w["pub"] / "runs" / "phase3" / "rev" / "result.json"):  # fmt: skip
        r = json.loads(f.read_text())
        assert r["compliance"] == "reviewed_ok" and r["review"]["reviewer"] == "gs"
        assert r["review"]["review_doc"] == str(doc.resolve())
    gh = (w["tmp"] / "gh.log").read_text()
    assert "release create checkpoints-rev" in gh and "--target abc123" in gh
    p = subprocess.run(
        [*args, "--no-release"], check=False, env=env, capture_output=True, text=True
    )
    assert p.returncode != 0 and "not pending_review" in p.stderr


def test_dirty_tree_or_moved_head_stops_the_chain(world: dict[str, Path]) -> None:
    w = world
    (w["repo"] / "code.py").write_text("x = 2\n")
    p = run_lib(w, "pin")
    assert p.returncode != 0 and "dirty" in p.stdout
    git(w["repo"], "checkout", "-q", "code.py")
    pin = git(w["repo"], "rev-parse", "HEAD")
    (w["repo"] / "runs" / "memwatch.log").write_text("busy\n")  # run artefacts are not code
    assert run_lib(w, "pin").returncode == 0
    git(w["repo"], "commit", "-q", "--allow-empty", "-m", "moved")
    p = run_lib(w, f"PIN={pin}; verify_pin")
    assert p.returncode != 0 and "HEAD moved" in p.stdout
