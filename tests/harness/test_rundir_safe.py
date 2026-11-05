import os
from pathlib import Path

import pytest

from weight_splice_lab.harness.rundir import (
    HARNESS_VERSION,
    IntegrityError,
    append_regular,
    copy_tree_regular,
    open_dir,
    provenance,
    read_regular,
    write_atomic,
    write_regular,
)


@pytest.fixture
def work(tmp_path: Path) -> Path:
    (tmp_path / "work").mkdir()
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "secret.txt").write_text("CANARY")
    return tmp_path / "work"


def test_read_refuses_symlink_hardlink_and_fifo(work: Path) -> None:
    secret = work.parent / "outside" / "secret.txt"
    (work / "notes.md").symlink_to(secret)
    with pytest.raises(IntegrityError, match="symbolic link"):
        read_regular(work, "notes.md", 1000)
    os.link(secret, work / "hard.md")
    with pytest.raises(IntegrityError, match="hard link"):
        read_regular(work, "hard.md", 1000)
    os.mkfifo(work / "fifo.md")
    with pytest.raises(IntegrityError, match="not a regular file"):
        read_regular(work, "fifo.md", 1000)  # must not block
    (work / "ok.md").write_text("x" * 100 + "END")
    assert read_regular(work, "ok.md", 3, tail=True) == b"END"
    with pytest.raises(FileNotFoundError):
        read_regular(work, "missing.md", 10)


def test_append_never_writes_through_a_link(work: Path) -> None:
    target = work.parent / "outside" / "appended.pth"
    (work / "cells.py").symlink_to(target)
    with pytest.raises(IntegrityError):
        append_regular(work, "cells.py", "import os\n")
    assert not target.exists()
    (work / "own.py").write_text("a\n")
    append_regular(work, "own.py", "b\n")
    assert (work / "own.py").read_text() == "a\nb\n"


def test_writes_refuse_symlinked_dirs_and_files(work: Path) -> None:
    outside = work.parent / "outside"
    (work / "fwd_1").symlink_to(outside, target_is_directory=True)
    with pytest.raises(IntegrityError):
        write_regular(work, "fwd_1/meta.json", b"{}", create_dirs=True)
    (work / "fwd_2.jsonl").symlink_to(outside / "secret.txt")
    with pytest.raises(IntegrityError):
        write_regular(work, "fwd_2.jsonl", b"overwrite")
    assert (outside / "secret.txt").read_text() == "CANARY"
    assert sorted(p.name for p in outside.iterdir()) == ["secret.txt"]
    write_regular(work, "fwd_3/m/a.npy", b"data", create_dirs=True)
    assert (work / "fwd_3" / "m" / "a.npy").read_bytes() == b"data"
    with open_dir(work, ["fwd_3", "m"]) as dfd, write_atomic(dfd, "a.npy") as f:
        f.write(b"new")
    assert (work / "fwd_3" / "m" / "a.npy").read_bytes() == b"new"
    assert sorted(p.name for p in (work / "fwd_3" / "m").iterdir()) == ["a.npy"]


def test_hardlinked_output_is_replaced_not_written_through(work: Path) -> None:
    secret = work.parent / "outside" / "secret.txt"
    os.link(secret, work / "fwd_4.jsonl")
    write_regular(work, "fwd_4.jsonl", b"new")
    assert secret.read_text() == "CANARY" and (work / "fwd_4.jsonl").read_bytes() == b"new"


def test_copy_tree_rejects_links_and_size(work: Path, tmp_path: Path) -> None:
    (work / "edited" / "tensors").mkdir(parents=True)
    (work / "edited" / "arch.json").write_text("{}")
    (work / "edited" / "tensors" / "a.pt").write_bytes(b"1234")
    got = copy_tree_regular(work, "edited", tmp_path / "snap1", 100, 100)
    assert got == ["arch.json", "tensors/a.pt"]
    assert (tmp_path / "snap1" / "tensors" / "a.pt").read_bytes() == b"1234"
    with pytest.raises(IntegrityError, match="size"):
        copy_tree_regular(work, "edited", tmp_path / "snap2", 3, 100)
    (work / "edited" / "tensors" / "b.pt").symlink_to(work.parent / "outside" / "secret.txt")
    with pytest.raises(IntegrityError):
        copy_tree_regular(work, "edited", tmp_path / "snap3", 100, 100)


def test_provenance_fields() -> None:
    prov = provenance()
    assert prov["harness_version"] == HARNESS_VERSION == "v4"
    assert isinstance(prov["git_sha"], str) and len(prov["git_sha"]) == 40
    assert isinstance(prov["dirty"], bool)
    assert len(str(prov["harness_source_sha256"])) == 64
