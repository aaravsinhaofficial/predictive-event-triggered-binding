#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p outputs/logs outputs/generated_configs outputs/insights/v7_publishable

TASKS="${TASKS:-blimp,syntaxgym,fillergap,naturalstories}"
STOP_SESSION="${STOP_SESSION:-etb_v7_publishable_clean}"
WAIT_FOR_PATH="${WAIT_FOR_PATH:-outputs/baselines/babylm_10m_v7_publishable_s29/dense_gru/eval/metrics.json}"
POLL_SECS="${POLL_SECS:-60}"
BOOTSTRAP_REPEATS="${BOOTSTRAP_REPEATS:-1000}"
PERMUTATION_REPEATS="${PERMUTATION_REPEATS:-1000}"

make_config() {
  local seed="$1"
  local label="$2"
  local out="$3"
  SEED="$seed" LABEL="$label" OUT="$out" uv run python - <<'PY'
import os
from pathlib import Path

import yaml

seed = int(os.environ["SEED"])
label = os.environ["LABEL"]
out = Path(os.environ["OUT"])
with Path("configs/babylm_10m_v7_publishable.yaml").open("r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)

suffix = f"s{seed}" if label == "main" else f"{label}_s{seed}"
run_name = f"babylm_10m_v7_publishable_{suffix}"
cfg["seed"] = seed
cfg["run_name"] = run_name
cfg["output_dir"] = f"outputs/{run_name}"
cfg["data"]["tokenizer_dir"] = f"outputs/{run_name}/tokenizer"

if label == "rt_supervised_upper_bound":
    cfg["training"]["rt_gate"]["enabled"] = True
elif label == "main":
    cfg["training"]["rt_gate"]["enabled"] = False
else:
    raise ValueError(f"Unknown config label: {label}")

out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
print(out)
PY
}

run_variant_if_missing() {
  local cfg="$1"
  local run_name="$2"
  local variant="$3"
  local marker="outputs/baselines/${run_name}/${variant}/eval/metrics.json"
  if [[ -f "$marker" ]]; then
    echo "Skipping existing $run_name/$variant"
    return 0
  fi
  uv run etb run-baselines \
    --config "$cfg" \
    --variants "$variant" \
    --tasks "$TASKS"
}

{
  date -u
  git log -1 --oneline
  git status --short --branch

  echo "Waiting for current seed-29 main grid marker: $WAIT_FOR_PATH"
  while [[ ! -f "$WAIT_FOR_PATH" ]]; do
    date -u
    tmux capture-pane -pt "$STOP_SESSION" -S -6 || true
    sleep "$POLL_SECS"
  done

  echo "Seed-29 main grid marker exists; stopping exhaustive queue session: $STOP_SESSION"
  if tmux has-session -t "$STOP_SESSION" 2>/dev/null; then
    tmux send-keys -t "$STOP_SESSION" C-c || true
    sleep 10
    if tmux has-session -t "$STOP_SESSION" 2>/dev/null; then
      tmux kill-session -t "$STOP_SESSION" || true
    fi
  fi

  if [[ -f outputs/baselines/babylm_10m_v7_publishable_s29/summary.csv ]]; then
    uv run etb claim-report \
      --summary-csv outputs/baselines/babylm_10m_v7_publishable_s29/summary.csv \
      --out outputs/baselines/babylm_10m_v7_publishable_s29/claim_report.json || true
  fi

  cfg29_rt="outputs/generated_configs/babylm_10m_v7_publishable_rt_supervised_upper_bound_s29.yaml"
  make_config 29 rt_supervised_upper_bound "$cfg29_rt"
  run_variant_if_missing "$cfg29_rt" "babylm_10m_v7_publishable_rt_supervised_upper_bound_s29" etb

  cfg31="outputs/generated_configs/babylm_10m_v7_publishable_s31.yaml"
  make_config 31 main "$cfg31"
  run_variant_if_missing "$cfg31" "babylm_10m_v7_publishable_s31" etb
  run_variant_if_missing "$cfg31" "babylm_10m_v7_publishable_s31" generic_dynamic
  run_variant_if_missing "$cfg31" "babylm_10m_v7_publishable_s31" punctuation_only

  cfg31_rt="outputs/generated_configs/babylm_10m_v7_publishable_rt_supervised_upper_bound_s31.yaml"
  make_config 31 rt_supervised_upper_bound "$cfg31_rt"
  run_variant_if_missing "$cfg31_rt" "babylm_10m_v7_publishable_rt_supervised_upper_bound_s31" etb

  uv run python scripts/build_insights_report.py \
    --root outputs/baselines \
    --pattern 'babylm_10m_v7_publishable_*' \
    --out-dir outputs/insights/v7_publishable \
    --bootstrap-repeats "$BOOTSTRAP_REPEATS" \
    --permutation-repeats "$PERMUTATION_REPEATS"

  date -u
} 2>&1 | tee outputs/logs/babylm_10m_v7_insights_fast_finish.log
