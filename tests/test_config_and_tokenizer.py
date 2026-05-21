from pathlib import Path

from etb.config import load_experiment_config
from etb.data.tokenizer import PUNCTUATION_TOKENS, load_or_train_tokenizer, token_ids_for


def test_load_smoke_config():
    config = load_experiment_config("configs/smoke.yaml")
    assert config.run_name == "smoke"
    assert config.model["variant"] == "etb"
    assert Path(config.data["train_text_path"]).exists()


def test_train_and_load_tokenizer(tmp_path):
    tokenizer = load_or_train_tokenizer(
        tokenizer_dir=tmp_path / "tok",
        train_files=["data/fixtures/tiny_corpus.txt"],
        vocab_size=128,
    )
    reloaded = load_or_train_tokenizer(
        tokenizer_dir=tmp_path / "tok",
        train_files=["data/fixtures/tiny_corpus.txt"],
        vocab_size=128,
    )
    assert len(reloaded) == len(tokenizer)
    punctuation_ids = token_ids_for(reloaded, PUNCTUATION_TOKENS)
    assert punctuation_ids

