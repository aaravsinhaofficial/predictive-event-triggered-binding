import torch

from etb.data.tokenizer import CLAUSE_TOKENS, PUNCTUATION_TOKENS, load_or_train_tokenizer, token_ids_for
from etb.models.configuration_etb import ETBConfig
from etb.models.gates import cheap_predictive_stats, cue_interference
from etb.models.modeling_etb import ETBForCausalLM
from etb.models.vsa import bind


def _tiny_model(tmp_path, variant="etb"):
    tokenizer = load_or_train_tokenizer(
        tokenizer_dir=tmp_path / f"tok-{variant}",
        train_files=["data/fixtures/tiny_corpus.txt"],
        vocab_size=128,
    )
    config = ETBConfig(
        vocab_size=len(tokenizer),
        d_model=32,
        hidden_size=32,
        num_layers=1,
        variant=variant,
        memory_slots=3,
        memory_roles=4,
        vsa_dim=64,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        unk_token_id=tokenizer.unk_token_id,
        punctuation_token_ids=token_ids_for(tokenizer, PUNCTUATION_TOKENS),
        clause_token_ids=token_ids_for(tokenizer, CLAUSE_TOKENS),
    )
    return ETBForCausalLM(config), tokenizer


def test_forward_contract_and_memory_events(tmp_path):
    model, tokenizer = _tiny_model(tmp_path)
    ids = tokenizer.encode("Who did the teacher solve the puzzle ?", add_special_tokens=False)
    input_ids = torch.tensor([[tokenizer.bos_token_id, *ids, tokenizer.eos_token_id]])
    output = model(input_ids=input_ids, labels=input_ids)
    assert output.logits.shape[:2] == input_ids.shape
    assert output.gate_probs.shape == input_ids.shape
    assert output.gate_activations.shape == input_ids.shape
    assert output.memory_events["write_slot"].shape == input_ids.shape
    assert output.information_gain.shape == input_ids.shape
    assert output.gate_targets.shape == input_ids.shape
    assert output.benefit_loss.item() >= 0
    assert output.budget_loss.item() >= 0
    assert output.activated_flops.item() > 0
    assert output.loss.item() > 0


def test_baseline_variants_forward(tmp_path):
    for variant in [
        "cheap_only",
        "dense_gru",
        "always_on",
        "punctuation_only",
        "generic_dynamic",
        "anira_emergent",
    ]:
        model, tokenizer = _tiny_model(tmp_path, variant=variant)
        ids = tokenizer.encode("The scientist wrote a note .", add_special_tokens=False)
        input_ids = torch.tensor([[tokenizer.bos_token_id, *ids, tokenizer.eos_token_id]])
        output = model(input_ids=input_ids, labels=input_ids)
        assert output.logits.shape[-1] == len(tokenizer)


def test_cheap_surprisal_feature_is_causal():
    input_ids = torch.tensor([[1, 2, 3, 4]])
    logits = torch.zeros(1, 4, 8)
    stats_a = cheap_predictive_stats(logits, input_ids)
    logits_perturbed = logits.clone()
    logits_perturbed[:, 2, 4] = 100.0
    stats_b = cheap_predictive_stats(logits_perturbed, input_ids)
    assert torch.allclose(stats_a[:, 2, 1], stats_b[:, 2, 1])


def test_cue_interference_is_causal():
    hidden = torch.randn(1, 5, 8)
    base = cue_interference(hidden)
    changed = hidden.clone()
    changed[:, 4, :] = torch.randn(1, 8) * 100
    perturbed = cue_interference(changed)
    assert torch.allclose(base[:, :4], perturbed[:, :4])
    assert base.shape == (1, 5, 1)


def test_vsa_fallback_bind_shape():
    left = torch.randn(2, 8)
    right = torch.randn(2, 8)
    out = bind(left, right)
    assert out.shape == left.shape
