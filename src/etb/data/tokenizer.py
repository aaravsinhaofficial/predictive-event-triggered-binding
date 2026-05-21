from __future__ import annotations

from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
PUNCTUATION_TOKENS = [".", "!", "?", ";", ":"]
CLAUSE_TOKENS = [",", ";", ":", "because", "that", "when", "while", "although", "if"]


def train_word_tokenizer(
    files: Iterable[str | Path],
    out_dir: str | Path,
    vocab_size: int,
) -> PreTrainedTokenizerFast:
    tokenizer = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Whitespace(), pre_tokenizers.Punctuation()]
    )
    trainer = trainers.WordLevelTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)
    tokenizer.train([str(path) for path in files], trainer)

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    fast.model_max_length = 10**9
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(str(out_dir))
    return fast


def load_or_train_tokenizer(
    tokenizer_dir: str | Path,
    train_files: Iterable[str | Path],
    vocab_size: int,
) -> PreTrainedTokenizerFast:
    tokenizer_path = Path(tokenizer_dir)
    if (tokenizer_path / "tokenizer.json").exists():
        return PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))
    return train_word_tokenizer(train_files, tokenizer_path, vocab_size)


def token_ids_for(tokenizer: PreTrainedTokenizerFast, tokens: Iterable[str]) -> list[int]:
    ids: list[int] = []
    unk_id = tokenizer.unk_token_id
    for token in tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id != unk_id:
            ids.append(int(token_id))
    return sorted(set(ids))

