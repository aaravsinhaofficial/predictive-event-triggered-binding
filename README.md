# Predictive Event-Triggered Binding

This repository implements a locally runnable research suite for **Predictive Event-Triggered Binding**: a low-compute language model that keeps a cheap recurrent predictive state active on every token and invokes explicit role-filler memory only at sparse, high-value events.

The first implementation is intentionally narrow and reproducible:

- GRU cheap backbone
- sparse causal event gate
- tiny MAP-style vector-symbolic role-filler memory
- Hugging Face-compatible checkpoints
- local fixtures and smoke tests
- BabyLM 2026 Strict/Strict-Small fetch and pilot configs
- BLiMP, SyntaxGym-style, filler-gap, Natural Stories, BabyLM export, and optional Brain-Score hooks

## Quickstart

```bash
uv run pytest
uv run etb train --config configs/smoke.yaml
uv run etb evaluate --config configs/smoke.yaml --checkpoint outputs/smoke/checkpoint-final --tasks fixture
```

Run the small BabyLM pilot:

```bash
uv run etb train --config configs/pilot_100k.yaml
uv run etb evaluate --config configs/pilot_100k.yaml --checkpoint outputs/pilot_100k/checkpoint-final --tasks fixture
```

The 10M and 100M BabyLM configs are included for reproducibility but are not launched by default.

Run a smoke-size baseline grid:

```bash
uv run etb run-baselines --config configs/smoke.yaml --max-steps 4 --tasks fixture
uv run etb claim-report --summary-csv outputs/baselines/smoke/summary.csv --out outputs/baselines/smoke/claim_report.json
```

## Data

The default paper target is BabyLM 2026:

- Strict: `BabyLM-community/BabyLM-2026-Strict`
- Strict-Small: `BabyLM-community/BabyLM-2026-Strict-Small`
- Dev/test: `BabyLM-community/BabyLM-dev`, `BabyLM-community/BabyLM-Test`

Fetch prepared text files explicitly:

```bash
uv run etb fetch-babylm --track strict-small --out-dir data/babylm --max-tokens 100000
uv run etb fetch-babylm --track strict --out-dir data/babylm
```

External evaluation sources the repo is designed to interoperate with:

- BabyLM 2026 guidelines: https://babylm.github.io/guidelines.html
- BLiMP: https://github.com/alexwarstadt/blimp
- SyntaxGym Core: https://cpllab.github.io/syntaxgym-core/
- Natural Stories: https://github.com/languageMIT/naturalstories
- Brain-Score Language: https://github.com/brain-score/language
- TorchHD: https://github.com/hyperdimensional-computing/torchhd
- EMNLP 2025 filler-gap suite: https://github.com/um-cap-lab/EMNLP-2025-submission

## Architecture

At each token, the model computes cheap logits from a GRU state. A causal gate then decides whether to run explicit memory using only information available at that token: hidden state, boundary features, cheap predictive entropy, and the realized surprisal of the current observed token under the previous cheap distribution. The memory module writes a role-bound filler vector into a small slot memory and reads it back as a residual next-token predictor.

The learned ETB gate is trained with three theory-facing auxiliary signals:

- an information-gain target from an always-on memory candidate, approximating `log p_f - log p_c`
- a target-sparsity budget, so the optimum is not the degenerate never-fire policy
- causal cue-overlap/interference and event-boundary priors

The forward pass returns:

- `logits`
- `loss`
- `gate_probs`
- `gate_activations`
- `memory_events`
- `activated_flops`
- `aux_loss`

## Baselines

Set `model.variant` in a config to:

- `etb`
- `cheap_only`
- `dense_gru`
- `always_on`
- `generic_dynamic`
- `anira_emergent`
- `punctuation_only`
- `random_matched`

All variants share the same training/evaluation interface.

## Outputs

Training writes:

- `metrics.jsonl`
- `config.yaml`
- `checkpoint-final/`
- tokenizer files

Evaluation writes:

- `metrics.json`
- `sentence_loglik.jsonl`
- `gate_traces.jsonl`
- task-specific CSV/JSONL files

Analysis commands generate:

- compute-accuracy Pareto plots
- Natural Stories token gate traces
- gate-rate breakdowns by token/event class

## V7 Insights Results

The final leakage-controlled V7 analysis package is checked in under
`outputs/insights/v7_publishable/`.

Start with:

- `outputs/insights/v7_publishable/README.md` for the interpretation
- `outputs/insights/v7_publishable/report.md` for the generated report
- `outputs/insights/v7_publishable/*.csv` for final tables
- `outputs/insights/v7_publishable/figures/` for PNG/PDF figures

The result supports an Insights-style failure-analysis framing rather than a
clean positive ETB-superiority claim.
