import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

from weight_splice_lab.gemma.model import count_params, from_hf, save_model
from weight_splice_lab.gemma.task import BASE_CKPT, BASE_DIR, HF_MODEL

REVISION = "23cf460f6bb16954176b3ddcc8d4f250501458a9"


def main() -> None:
    hf_dir = snapshot_download(
        HF_MODEL, revision=REVISION, allow_patterns=["*.json", "*.safetensors"]
    )
    model = from_hf(Path(hf_dir))
    save_model(model, BASE_CKPT)
    shutil.copy(f"{hf_dir}/tokenizer.json", BASE_DIR / "tokenizer.json")
    meta = {
        "hf_model": HF_MODEL,
        "revision": REVISION,
        "params": count_params(model),
        "dtype": "bfloat16",
    }
    (BASE_DIR / "source.json").write_text(json.dumps(meta, indent=1))
    print(meta)


if __name__ == "__main__":
    main()
