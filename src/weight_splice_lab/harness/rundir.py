import hashlib
import inspect
import json
import os
import secrets
import shutil
import stat
import subprocess
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from string import Template
from typing import BinaryIO

import torch

from weight_splice_lab.harness.family import FAMILIES, Family

ROOT = Path(__file__).resolve().parents[3]
RUNS_ROOT = Path.home() / "wsl-runs"
HARNESS_VERSION = "v4"


@dataclass
class RunConfig:
    run_id: str
    family: str = "toy"
    condition: str = "add_capacity"
    forward_budget: int = 50
    python_runs: int = 300
    max_added_params: int | None = None
    model: str = "gpt-6-astra"
    profile: str = "personal"
    effort: str = "high"
    handoff_turns: int = 40
    require_watchdog: bool = False
    handoff_tokens: int = 180_000
    max_limit_points: float = 5.0
    max_input_tokens: int = 20_000_000
    max_turns: int = 200
    max_attempts: int = 260
    stale_rate_turns: int = 3
    kernel_timeout_s: float = 900.0
    kernel_max_rss_gb: float | None = None
    output_chars: int = 10_000
    # save_edit "end": true; off by default since gemma1 ended after 11 of 50 passes
    allow_end: bool = False
    # phase 4c: the editor may run the model in its kernel (build_model); no gradients still
    kernel_forward: bool = False

    def __post_init__(self) -> None:
        # None means the family's default
        fam = FAMILIES[self.family]
        if self.max_added_params is None:
            self.max_added_params = fam.max_added_params
        if self.kernel_max_rss_gb is None:
            self.kernel_max_rss_gb = fam.kernel_max_rss_gb

    @property
    def fam(self) -> Family:
        return FAMILIES[self.family]


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def sandbox(self) -> Path:
        return self.root / "sandbox"

    @property
    def private(self) -> Path:
        return self.root / "harness"

    @property
    def work(self) -> Path:
        return self.sandbox / "work"

    @property
    def stage(self) -> Path:
        return self.work / "edited"

    @property
    def edits(self) -> Path:
        return self.private / "edits"

    @property
    def turns(self) -> Path:
        return self.private / "turns"


RULE_B = (
    "(b) running the model other than through the forward-pass tool: do not implement or "
    "simulate the network's computation on any input in the kernel;"
)
RULE_B_KERNEL = (
    "(b) searching over weight values by running the model: do not write code that tries many "
    "candidate weights and keeps the best by a measured score (that is optimisation by trial). "
    "Running the model in the kernel with build_model() is allowed, to look at what it does and "
    "to check a candidate you constructed; when a check fails, work out why and construct the "
    "next candidate from that analysis;"
)
KERNEL_MODEL = (
    " Also defined: `build_model(edited=True)`, the model itself, which you MAY RUN IN THIS KERNEL "
    "(float32, eval mode, gradients disabled). `m = build_model()` returns a new copy with your "
    "current STAGED edit and added capacity applied (staged, not saved: no save_edit needed); "
    "`build_model(False)` returns the unedited model (one cached copy, loaded on first use, about "
    "1.1 GB: do not modify it, every later copy starts from it). Each edited copy shares the "
    "embedding and adds about 0.4 GB, so keep only a few. `m.hidden(ids)` returns the final-normed states "
    "[batch, positions, 640]; `m(ids)` the logits, which take about 1 MB per position, so for long "
    "or many texts multiply only the positions you need by `m.embed.weight.T`. `m.hidden(ids, "
    "capture=d)` fills the dict d with the teacher-forced tool's sites under the same names (for "
    "example d['resid_pre.3'], d['attn.3.pattern'], d['mlp.3.hidden']). `m.generate(ids, n)` "
    "decodes greedily, as the evaluation does. ids are a LongTensor [batch, positions] that start "
    "with <bos> (`tok.encode` adds it); pad nothing: batch prompts of equal length, or one at a "
    "time. The forward-pass tool below remains available and is the reference for grading and KL.\n"
)


def render_prompt(cfg: RunConfig, mcfg: dict[str, object], template: Path | None = None) -> str:
    d, hd = int(str(mcfg["d_model"])), int(str(mcfg["head_dim"]))
    condition = {
        "add_capacity": (
            "ADD_CAPACITY: you may change existing weights and you may also add new attention heads "
            "and MLP neurons (add_heads / add_neurons in the kernel; new capacity starts at zero, "
            f"so it changes nothing until you write it). At most {cfg.max_added_params:,} added "
            f"parameters in total (an added head costs 4 x {d} x {hd} = {4 * d * hd:,}, an added "
            f"neuron 3 x {d} = {3 * d:,})."
        ),
        "edit_only": (
            "EDIT_ONLY: you may change existing weights, but the architecture is fixed: no heads or "
            "neurons can be added."
        ),
    }[cfg.condition]
    return Template((template or cfg.fam.prompt_file).read_text()).substitute(
        forward_budget=cfg.forward_budget,
        python_runs=cfg.python_runs,
        condition=condition,
        timeout=int(cfg.kernel_timeout_s),
        memory=f"{cfg.kernel_max_rss_gb:g}",
        output_chars=f"{cfg.output_chars:,}",
        forward_items=f"{cfg.fam.forward_items:,}",
        kernel_model=KERNEL_MODEL if cfg.kernel_forward else "",
        rule_b=RULE_B_KERNEL if cfg.kernel_forward else RULE_B,
    )


def build_run_dir(
    cfg: RunConfig,
    runs_root: Path = RUNS_ROOT,
    text: str | None = None,
    base_ckpt: Path | None = None,
) -> RunPaths:
    fam = cfg.fam
    base_ckpt = base_ckpt or fam.base_ckpt
    paths = RunPaths(runs_root / cfg.run_id)
    if paths.root.exists():
        raise FileExistsError(f"{paths.root} exists; pick a new run id")
    for d in (paths.sandbox / "model", paths.sandbox / "practice", paths.stage, paths.private):
        d.mkdir(parents=True)
    model_dir = paths.sandbox / "model"
    shutil.copy(base_ckpt, model_dir / "base.pt")
    shutil.copy(inspect.getsourcefile(fam.model_module) or "", model_dir / "model.py")
    fam.write_model_extras(model_dir)
    ckpt = torch.load(base_ckpt, map_location="cpu", weights_only=True)
    (model_dir / "config.json").write_text(json.dumps(
        {"model": ckpt["cfg"], "family": fam.name, "module_alias": fam.kernel_alias,
         "lm_class": fam.lm_class, "condition": cfg.condition,
         "max_added_params": cfg.max_added_params, "kernel_forward": cfg.kernel_forward},
        indent=1,
    ))  # fmt: skip
    practice = paths.sandbox / "practice"
    for name, rows in fam.practice().items():
        (practice / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    (practice / "text.txt").write_text(text if text is not None else fam.practice_text())
    (paths.sandbox / "TASK.md").write_text(render_prompt(cfg, ckpt["cfg"]))
    (paths.work / "notes.md").write_text("")
    (paths.private / "instructions.md").write_text("Answer the user's request.\n")
    (paths.private / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
    return paths


# Editor-controlled paths (everything under sandbox/work) are only touched through these helpers:
# directory-relative, no-follow opens, regular single-link files only, writes via temp + rename.
class IntegrityError(Exception):
    pass


_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _is_link(name: str, dir_fd: int) -> bool:
    try:
        return stat.S_ISLNK(os.stat(name, dir_fd=dir_fd, follow_symlinks=False).st_mode)
    except FileNotFoundError:
        return False


def _check_name(part: str) -> None:
    if part in ("", ".", "..") or "/" in part or "\0" in part:
        raise IntegrityError(f"invalid path component {part!r}")


def _open_dir_fd(root: Path, parts: Sequence[str], create: bool) -> int:
    fd = os.open(root, _DIR_FLAGS)
    try:
        for part in parts:
            _check_name(part)
            if _is_link(part, fd):
                raise IntegrityError(f"{part} is a symbolic link")
            if create:
                try:
                    os.mkdir(part, 0o755, dir_fd=fd)
                except FileExistsError:
                    pass
            try:
                nfd = os.open(part, _DIR_FLAGS, dir_fd=fd)
            except OSError:
                if _is_link(part, fd):
                    raise IntegrityError(f"{part} is a symbolic link") from None
                raise
            os.close(fd)
            fd = nfd
    except BaseException:
        os.close(fd)
        raise
    return fd


@contextmanager
def open_dir(root: Path, parts: Sequence[str] = (), create: bool = False) -> Iterator[int]:
    fd = _open_dir_fd(root, parts, create)
    try:
        yield fd
    finally:
        os.close(fd)


def _split(rel: str) -> tuple[list[str], str]:
    parts = rel.split("/")
    return parts[:-1], parts[-1]


def open_regular(dir_fd: int, name: str, flags: int) -> int:
    _check_name(name)
    try:
        fd = os.open(name, flags | os.O_NOFOLLOW | os.O_NONBLOCK, 0o644, dir_fd=dir_fd)
    except OSError:
        if _is_link(name, dir_fd):
            raise IntegrityError(f"{name} is a symbolic link") from None
        raise
    st = os.fstat(fd)
    if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
        os.close(fd)
        kind = "not a regular file" if not stat.S_ISREG(st.st_mode) else "a hard link"
        raise IntegrityError(f"{name} is {kind}")
    return fd


def read_regular(root: Path, rel: str, max_bytes: int, tail: bool = False) -> bytes:
    dirs, name = _split(rel)
    with open_dir(root, dirs) as dfd:
        fd = open_regular(dfd, name, os.O_RDONLY)
        with os.fdopen(fd, "rb") as f:
            size = os.fstat(f.fileno()).st_size
            if size > max_bytes and not tail:
                raise IntegrityError(f"{rel} is larger than {max_bytes:,} bytes")
            if tail and size > max_bytes:
                f.seek(size - max_bytes)
            return f.read(max_bytes)


def append_regular(root: Path, rel: str, text: str) -> None:
    dirs, name = _split(rel)
    with open_dir(root, dirs) as dfd:
        fd = open_regular(dfd, name, os.O_WRONLY | os.O_APPEND | os.O_CREAT)
        with os.fdopen(fd, "a") as f:
            f.write(text)


@contextmanager
def write_atomic(dir_fd: int, name: str) -> Iterator[BinaryIO]:
    _check_name(name)
    if _is_link(name, dir_fd):
        raise IntegrityError(f"{name} is a symbolic link")
    tmp = f".{name}.tmp-{os.getpid()}-{secrets.token_hex(4)}"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644, dir_fd=dir_fd)
    try:
        with os.fdopen(fd, "wb") as f:
            yield f
        os.replace(tmp, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
    except BaseException:
        try:
            os.unlink(tmp, dir_fd=dir_fd)
        except FileNotFoundError:
            pass
        raise


def write_regular(root: Path, rel: str, data: bytes, create_dirs: bool = False) -> None:
    dirs, name = _split(rel)
    with open_dir(root, dirs, create=create_dirs) as dfd, write_atomic(dfd, name) as f:
        f.write(data)


def copy_tree_regular(root: Path, rel: str, dest: Path, max_file: int, max_total: int) -> list[str]:
    # Copies regular files only; any link or special file aborts with IntegrityError.
    copied: list[str] = []
    total = 0

    def walk(dfd: int, out: Path, prefix: str) -> None:
        nonlocal total
        out.mkdir(parents=True, exist_ok=True)
        for name in sorted(os.listdir(dfd)):
            st = os.stat(name, dir_fd=dfd, follow_symlinks=False)
            label = f"{prefix}{name}"
            if stat.S_ISDIR(st.st_mode):
                sub = os.open(name, _DIR_FLAGS, dir_fd=dfd)
                try:
                    walk(sub, out / name, label + "/")
                finally:
                    os.close(sub)
                continue
            if not stat.S_ISREG(st.st_mode):
                raise IntegrityError(f"{label} is a symbolic link or special file")
            fd = open_regular(dfd, name, os.O_RDONLY)
            with os.fdopen(fd, "rb") as f, (out / name).open("wb") as g:
                # bound the bytes actually read: the file may grow after any size check
                limit = min(max_file, max_total - total)
                n = 0
                while chunk := f.read(min(2**20, limit - n + 1)):
                    n += len(chunk)
                    if n > limit:
                        raise IntegrityError(f"{label}: staged files exceed the size limit")
                    g.write(chunk)
                total += n
            copied.append(label)

    with open_dir(root, [p for p in rel.split("/") if p]) as dfd:
        walk(dfd, dest, "")
    return copied


def harness_files(root: Path = ROOT) -> list[Path]:
    pkg = root / "src" / "weight_splice_lab"
    files = [
        *(pkg / "harness").glob("*.py"),
        *(pkg / "toy").glob("*.py"),
        *(pkg / "gemma").glob("*.py"),
    ]
    files += sorted((root / "prompts").glob("*.md"))
    return sorted(f for f in files if f.exists())


def source_hash(files: list[Path], root: Path = ROOT) -> str:
    h = hashlib.sha256()
    for f in files:
        h.update(str(f.relative_to(root)).encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def _git(root: Path, *args: str) -> str | None:
    try:
        p = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
    except OSError:
        return None
    return p.stdout if p.returncode == 0 else None


def git_state(root: Path = ROOT) -> dict[str, object]:
    sha = _git(root, "rev-parse", "HEAD")
    # run artefacts under runs/ (memwatch.log, mirrors, caches) do not make the code dirty
    status = _git(root, "status", "--porcelain", "--", ".", ":(exclude)runs")
    dirty_files = [line[3:] for line in (status or "").splitlines() if line.strip()]
    return {
        "git_sha": sha.strip() if sha else None,
        "dirty": status is None or bool(dirty_files),
        "dirty_files": dirty_files[:50],
    }


def provenance(root: Path = ROOT) -> dict[str, object]:
    return {
        "harness_version": HARNESS_VERSION,
        **git_state(root),
        "harness_source_sha256": source_hash(harness_files(root), root),
    }
