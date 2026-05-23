#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

WATCH_SESSION="${WATCH_SESSION:-etb_v7_publishable_clean}"
SLEEP_SECS="${SLEEP_SECS:-300}"
OUT_DIR="${OUT_DIR:-outputs/insights/v7_publishable}"
BOOTSTRAP_REPEATS="${BOOTSTRAP_REPEATS:-1000}"
PERMUTATION_REPEATS="${PERMUTATION_REPEATS:-1000}"

mkdir -p outputs/logs "$OUT_DIR"

{
  date -u
  echo "Waiting for tmux session: $WATCH_SESSION"
  while tmux has-session -t "$WATCH_SESSION" 2>/dev/null; do
    date -u
    tmux capture-pane -pt "$WATCH_SESSION" -S -8 || true
    sleep "$SLEEP_SECS"
  done

  date -u
  echo "Training queue finished; building Insights report."
  uv run python scripts/build_insights_report.py \
    --root outputs/baselines \
    --pattern 'babylm_10m_v7_publishable_*' \
    --out-dir "$OUT_DIR" \
    --bootstrap-repeats "$BOOTSTRAP_REPEATS" \
    --permutation-repeats "$PERMUTATION_REPEATS"
  date -u
} 2>&1 | tee outputs/logs/babylm_10m_v7_insights_finalizer.log
