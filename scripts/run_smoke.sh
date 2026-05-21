#!/usr/bin/env bash
set -euo pipefail

uv run etb train --config configs/smoke.yaml
uv run etb evaluate --config configs/smoke.yaml --checkpoint outputs/smoke/checkpoint-final --tasks fixture
uv run etb analyze --run-dir outputs/smoke

