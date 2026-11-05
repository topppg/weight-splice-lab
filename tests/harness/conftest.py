from pathlib import Path

import pytest
import torch

from weight_splice_lab.harness.rundir import RunConfig, RunPaths, build_run_dir
from weight_splice_lab.toy.data import TOKENIZER
from weight_splice_lab.toy.model import ToyConfig, ToyLM, save_model

DATA = Path(__file__).parent / "data"


def tiny_model() -> ToyLM:
    torch.manual_seed(0)
    cfg = ToyConfig(
        vocab_size=TOKENIZER.vocab_size, d_model=64, n_layers=2, n_heads=2, head_dim=16, d_mlp=96
    )
    return ToyLM(cfg).eval()


@pytest.fixture(scope="session")
def tiny_ckpt(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("ckpt") / "base.pt"
    save_model(tiny_model(), path)
    return path


def make_run(root: Path, ckpt: Path, cfg: RunConfig) -> RunPaths:
    return build_run_dir(cfg, root, text="The cat sat on the mat. " * 50, base_ckpt=ckpt)


@pytest.fixture(scope="session")
def run_paths(tiny_ckpt: Path, tmp_path_factory: pytest.TempPathFactory) -> RunPaths:
    cfg = RunConfig(run_id="t", kernel_timeout_s=30.0)
    return make_run(tmp_path_factory.mktemp("runs"), tiny_ckpt, cfg)
