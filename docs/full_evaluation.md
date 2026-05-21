# Full Evaluation Notes

The fixture tasks in `data/fixtures/` are smoke tests. They prove that the scoring APIs work; they are not evidence for the paper claim.

For full evaluation, prepare the public datasets under `data/eval/` and point `configs/full_eval_template.yaml` at them:

- BLiMP: put the official `.jsonl` files in `data/eval/blimp/data/`. The evaluator accepts either a single JSONL file or a directory of JSONL files.
- SyntaxGym: put one or more suite `.json` files in `data/eval/syntaxgym/`. The evaluator accepts either a single suite or a directory.
- Filler-gap: convert the Chang et al. 2025 suite, or the local reimplementation, to the CSV schema used by `data/fixtures/fillergap/tiny.csv`.
- Natural Stories: convert token-level reading-time data to the TSV schema used by `data/fixtures/naturalstories/tiny.tsv`.

Run a frozen checkpoint on full local eval data:

```bash
uv run etb evaluate \
  --config configs/full_eval_template.yaml \
  --checkpoint outputs/pilot_100k/checkpoint-final \
  --tasks blimp,syntaxgym,fillergap,naturalstories
```

Run the small comparison grid without large training:

```bash
uv run etb run-baselines --config configs/smoke.yaml --max-steps 4 --tasks fixture
uv run etb claim-report --summary-csv outputs/baselines/smoke/summary.csv --out outputs/baselines/smoke/claim_report.json
```

The baseline summary is written to `outputs/baselines/<run_name>/summary.csv`.

For the EMNLP main-track claim, replace `configs/smoke.yaml` with the 10M/100M configs and full
evaluation paths. The claim report expects ETB to beat both `generic_dynamic` and
`anira_emergent` on filler-gap accuracy and Natural Stories residual `delta_r2_gate`, while staying
within the configured activated-FLOP tolerance.
