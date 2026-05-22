#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p outputs/logs

{
  date -u
  git log -1 --oneline
  git status --short --branch
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
  uv run pytest
  uv run etb run-baselines \
    --config configs/babylm_10m_v5.yaml \
    --variants etb,generic_dynamic,anira_emergent,punctuation_only \
    --tasks blimp,syntaxgym,fillergap,naturalstories
  uv run etb claim-report \
    --summary-csv outputs/baselines/babylm_10m_v5/summary.csv \
    --out outputs/baselines/babylm_10m_v5/claim_report.json
  date -u
} 2>&1 | tee outputs/logs/babylm_10m_v5_targeted.log
