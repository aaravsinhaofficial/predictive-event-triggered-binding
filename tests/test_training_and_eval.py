from pathlib import Path

from etb.config import ExperimentConfig
from etb.eval.runner import evaluate
from etb.eval.scoring import load_checkpoint, sentence_log_likelihood
from etb.training import train


def _test_config(tmp_path):
    return ExperimentConfig(
        raw={
            "run_name": "pytest_smoke",
            "seed": 7,
            "device": "cpu",
            "output_dir": str(tmp_path / "run"),
            "data": {
                "train_text_path": "data/fixtures/tiny_corpus.txt",
                "eval_text_path": "data/fixtures/tiny_eval.txt",
                "tokenizer_dir": str(tmp_path / "tok"),
                "vocab_size": 128,
                "block_size": 24,
            },
            "model": {
                "variant": "etb",
                "d_model": 32,
                "hidden_size": 32,
                "num_layers": 1,
                "dropout": 0.0,
                "memory_slots": 3,
                "memory_roles": 4,
                "vsa_dim": 64,
                "sparsity_lambda": 0.01,
            },
            "training": {
                "batch_size": 2,
                "max_steps": 2,
                "learning_rate": 0.001,
                "log_every": 1,
            },
            "evaluation": {
                "tasks": {
                    "blimp": "data/fixtures/blimp/tiny.jsonl",
                    "syntaxgym": "data/fixtures/syntaxgym/tiny_suite.json",
                    "fillergap": "data/fixtures/fillergap/tiny.csv",
                    "naturalstories": "data/fixtures/naturalstories/tiny.tsv",
                }
            },
        }
    )


def test_train_save_load_score_and_evaluate(tmp_path):
    config = _test_config(tmp_path)
    checkpoint = train(config)
    assert (checkpoint / "config.json").exists()
    assert (checkpoint / "tokenizer.json").exists()
    model, tokenizer, device = load_checkpoint(checkpoint, "cpu")
    score = sentence_log_likelihood(model, tokenizer, "The scientist wrote a note.", device, True)
    assert score["num_tokens"] > 0
    assert "trace" in score
    metrics = evaluate(config, checkpoint=checkpoint, tasks="fixture")
    assert "language_modeling" in metrics
    assert "blimp" in metrics
    assert Path(config.output_dir / "eval" / "metrics.json").exists()

