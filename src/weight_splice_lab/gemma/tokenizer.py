from pathlib import Path

from tokenizers import Tokenizer

PAD, EOS, BOS = 0, 1, 2


class GemmaTokenizer:
    def __init__(self, path: Path) -> None:
        self.tk = Tokenizer.from_file(str(path))
        self.vocab_size = self.tk.get_vocab_size()
        self.digit_ids = frozenset(self.tk.token_to_id(str(d)) for d in range(10))

    def encode(self, text: str, bos: bool = True) -> list[int]:
        ids = self.tk.encode(text, add_special_tokens=False).ids
        return [BOS, *ids] if bos else ids

    def decode(self, ids: list[int]) -> str:
        return self.tk.decode(ids, skip_special_tokens=False)

    def tokens(self, ids: list[int]) -> list[str]:
        return [self.tk.id_to_token(i) for i in ids]


def load(path: Path | None = None) -> GemmaTokenizer:
    # in the sandbox, tokenizer.json sits next to this file (model/)
    return GemmaTokenizer(path or Path(__file__).with_name("tokenizer.json"))
