#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p outputs/logs

{
  date -u
  git log -1 --oneline
  git status --short --branch
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
  uv run etb run-baselines \
    --config configs/babylm_10m_v6_audit.yaml \
    --variants etb \
    --tasks blimp,syntaxgym,fillergap,naturalstories
  date -u
} 2>&1 | tee outputs/logs/babylm_10m_v6_proof.log
