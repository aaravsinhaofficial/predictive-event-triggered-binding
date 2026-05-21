# Predictive ETB Results Log

## 2026-05-21: BabyLM 10M Full Baseline Grid

- Repository revision at launch: `d69ed32 Implement predictive event-triggered binding suite` plus local reproducibility updates in this commit.
- Hardware: NVIDIA L40S, 46,068 MiB VRAM; 248 GiB system RAM.
- Command log: `outputs/logs/babylm_10m_full_grid.log`.
- tmux session: `etb_full_10m`.
- Launch command:

```bash
tmux new-session -d -s etb_full_10m '
cd /home/ubuntu/predictive-event-triggered-binding &&
mkdir -p outputs/logs data/sources &&
{
  date -u
  git status --short --branch
  git pull --ff-only
  uv run pytest
  if [ ! -d data/sources/naturalstories/.git ]; then
    git clone --depth 1 https://github.com/languageMIT/naturalstories data/sources/naturalstories
  fi
  uv run etb fetch-babylm --track strict-small --out-dir data/babylm --max-tokens 10000000
  uv run python scripts/prepare_eval_data.py --out-dir data/eval --naturalstories-source data/sources/naturalstories
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
  uv run etb run-baselines --config configs/babylm_10m.yaml --tasks blimp,syntaxgym,fillergap,naturalstories
  uv run etb claim-report \
    --summary-csv outputs/baselines/babylm_10m/summary.csv \
    --out outputs/baselines/babylm_10m/claim_report.json
  date -u
} 2>&1 | tee outputs/logs/babylm_10m_full_grid.log
'
```

### Completed Setup

- `git pull --ff-only`: already up to date at launch.
- `uv run pytest`: passed, 8 tests in 4.38s.
- BabyLM slice: `data/babylm/strict_small_10000000.txt`, 10M-word Strict-Small slice.
- Evaluation data prepared under `data/eval/`:
  - BLiMP: 67 configs, 67,000 minimal pairs.
  - SyntaxGym: 31 suites, 799 items, 1,478 converted predictions.
  - Filler-gap/island: 8,000 BLiMP-derived island minimal pairs, closest available non-fixture filler-gap/island CSV.
  - Natural Stories: 10,256 tokens from `languageMIT/naturalstories`, with token length and log-frequency controls.

### Current Status

- As of 2026-05-21 01:32 UTC, `etb_full_10m` is training the first grid variant, `etb`.
- Observed progress: step 19 / 12,000, gate rate about 0.152, loss about 10.327.
- GPU utilization at the same poll: L40S using about 11.2 GiB VRAM.

### Next Actions

- Let `etb_full_10m` complete all eight variants:
  `etb`, `cheap_only`, `dense_gru`, `always_on`, `generic_dynamic`,
  `anira_emergent`, `punctuation_only`, and `random_matched`.
- Inspect `outputs/baselines/babylm_10m/summary.csv` and
  `outputs/baselines/babylm_10m/claim_report.json`.
- If the claim report fails, document whether the blocker is filler-gap/island accuracy,
  Natural Stories `delta_r2_gate`, or activated-FLOP mismatch.
