#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p outputs/logs outputs/generated_configs

SEEDS="${SEEDS:-23 29 31}"
MAIN_VARIANTS="${MAIN_VARIANTS:-etb,generic_dynamic,anira_emergent,punctuation_only,random_matched,cheap_only,dense_gru}"
ABLATIONS="${ABLATIONS:-no_contrastive no_aux rt_supervised_upper_bound}"
TASKS="${TASKS:-blimp,syntaxgym,fillergap,naturalstories}"

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

if label == "no_contrastive":
    cfg["training"]["contrastive"]["enabled"] = False
elif label == "no_rt":
    cfg["training"]["rt_gate"]["enabled"] = False
elif label == "rt_supervised_upper_bound":
    cfg["training"]["rt_gate"]["enabled"] = True
elif label == "no_aux":
    cfg["training"]["contrastive"]["enabled"] = False
    cfg["training"]["rt_gate"]["enabled"] = False
elif label != "main":
    raise ValueError(f"Unknown config label: {label}")

out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
print(out)
PY
}

{
  date -u
  git log -1 --oneline
  git status --short --branch
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

  uv run python scripts/create_publishable_splits.py \
    --source-dir data/eval \
    --out-dir data/eval_publishable \
    --seed 23

  uv run pytest

  for seed in ${SEEDS}; do
    main_cfg="outputs/generated_configs/babylm_10m_v7_publishable_s${seed}.yaml"
    make_config "$seed" main "$main_cfg"
    uv run etb run-baselines \
      --config "$main_cfg" \
      --variants "$MAIN_VARIANTS" \
      --tasks "$TASKS"
    uv run etb claim-report \
      --summary-csv "outputs/baselines/babylm_10m_v7_publishable_s${seed}/summary.csv" \
      --out "outputs/baselines/babylm_10m_v7_publishable_s${seed}/claim_report.json"

    for ablation in ${ABLATIONS}; do
      ablation_cfg="outputs/generated_configs/babylm_10m_v7_publishable_${ablation}_s${seed}.yaml"
      make_config "$seed" "$ablation" "$ablation_cfg"
      uv run etb run-baselines \
        --config "$ablation_cfg" \
        --variants etb \
        --tasks "$TASKS"
    done
  done

  date -u
} 2>&1 | tee outputs/logs/babylm_10m_v7_publishable_grid.log
