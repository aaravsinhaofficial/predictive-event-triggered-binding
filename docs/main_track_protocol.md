# EMNLP Main-Track Protocol

The decisive claim is:

> Where the gate fires matters more than raw sparsity.

The result is main-track credible only if the psycholinguistically motivated ETB gate beats both:

- `generic_dynamic`: matched-FLOP entropy/confidence routing
- `anira_emergent`: matched-FLOP learned emergent allocation without psycholinguistic gate inputs or information-gain supervision

Required wins:

- higher filler-gap/island accuracy
- higher Natural Stories reading-time residual variance beyond surprisal
- activated FLOPs/token matched within the claim-report tolerance

## Smoke Decision Aid

```bash
uv run etb run-baselines --config configs/smoke.yaml --max-steps 4 --tasks fixture
uv run etb claim-report \
  --summary-csv outputs/baselines/smoke/summary.csv \
  --out outputs/baselines/smoke/claim_report.json
```

Fixture results only validate plumbing. They are not paper evidence.

## Pilot Runs

Use the 10M track to tune failure modes and the 100M track for the paper table:

```bash
uv run etb run-baselines --config configs/babylm_10m.yaml --tasks blimp,syntaxgym,fillergap,naturalstories
uv run etb claim-report \
  --summary-csv outputs/baselines/babylm_10m/summary.csv \
  --out outputs/baselines/babylm_10m/claim_report.json
```

For final reporting, run three seeds by copying the config and changing `seed` plus `run_name`.
Report means and standard errors for perplexity, activated FLOPs/token, BLiMP, SyntaxGym,
filler-gap/island accuracy, and Natural Stories `delta_r2_gate`.

## Falsification Criteria

The core claim is weakened if:

- `generic_dynamic` matches ETB on filler-gap/island accuracy at matched FLOPs
- `anira_emergent` matches ETB, implying the gain is generic learned routing rather than psycholinguistic routing
- ETB gate activations fail to improve reading-time fit beyond surprisal and lexical controls
- ETB only wins by using substantially more activated FLOPs/token

