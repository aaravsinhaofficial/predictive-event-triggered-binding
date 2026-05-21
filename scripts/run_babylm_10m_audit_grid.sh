#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p outputs/logs

{
  date -u
  git log -1 --oneline
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
  uv run etb run-baselines \
    --config configs/babylm_10m_audit.yaml \
    --tasks blimp,syntaxgym,fillergap,naturalstories
  uv run etb claim-report \
    --summary-csv outputs/baselines/babylm_10m_audit/summary.csv \
    --out outputs/baselines/babylm_10m_audit/claim_report.json
  date -u
} 2>&1 | tee outputs/logs/babylm_10m_audit_grid.log
