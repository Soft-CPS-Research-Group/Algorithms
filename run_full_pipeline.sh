#!/usr/bin/env bash
# Full entity-interface offline RL pipeline: collect → IQL → CQL
#
# Usage:
#   chmod +x run_full_pipeline.sh
#   nohup ./run_full_pipeline.sh > logs/pipeline.log 2>&1 &
#
# Assumes the venv is at .venv relative to the repo root
# and this script is run from the repo root.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$REPO_ROOT/../../../.venv/bin/python"  # adjust if needed
# Try finding venv in common locations
if [ ! -f "$VENV_PYTHON" ]; then
    VENV_PYTHON="$(find "$REPO_ROOT" -maxdepth 3 -name "python" -path "*/bin/python" 2>/dev/null | head -1)"
fi
if [ ! -f "$VENV_PYTHON" ]; then
    VENV_PYTHON="$(which python3)"
fi

echo "[pipeline] VENV_PYTHON = $VENV_PYTHON"
echo "[pipeline] REPO_ROOT   = $REPO_ROOT"
echo "[pipeline] Started: $(date)"

# ----------------------------------------------------------------
# Step 1: Collect full dataset (10 seeds x 1 episode)
# ----------------------------------------------------------------
DATA_DIR="$REPO_ROOT/datasets/offline_rl/rbcsmart_entity"
SEEDS="22 23 24 25 26 27 28 29 30 31"
EPISODES=1
VAL_SEEDS="26"  # held-out val seed
TRAIN_SEEDS="22,23,24,25,26"  # training seeds (include val so loader can split)

echo ""
echo "[pipeline] === Step 1: Data collection ==="
"$VENV_PYTHON" -m scripts.collect_rbcsmart_dataset \
    --seeds $SEEDS \
    --episodes "$EPISODES"
echo "[pipeline] Collection done: $(date)"

# ----------------------------------------------------------------
# Step 2: IQL training (all 4 groups, seeds 22-25 train / 26 val)
# ----------------------------------------------------------------
IQL_OUTPUT="$REPO_ROOT/runs/offline_iql_entity/run-001"
mkdir -p "$IQL_OUTPUT"

echo ""
echo "[pipeline] === Step 2: IQL training ==="
"$VENV_PYTHON" -m scripts.train_iql_entity \
    --data-dir "$DATA_DIR" \
    --output "$IQL_OUTPUT" \
    --seeds "$TRAIN_SEEDS" \
    --val-seeds "$VAL_SEEDS" \
    --gradient-steps 150000 \
    --hidden-layers 256,256 \
    --eval-every 2500 \
    --device cpu
echo "[pipeline] IQL training done: $(date)"

# ----------------------------------------------------------------
# Step 3: CQL training (all 4 groups, seeds 22-25 train / 26 val)
# ----------------------------------------------------------------
CQL_OUTPUT="$REPO_ROOT/runs/offline_cql_entity/run-001"
mkdir -p "$CQL_OUTPUT"

echo ""
echo "[pipeline] === Step 3: CQL training ==="
"$VENV_PYTHON" -m scripts.train_cql_entity \
    --data-dir "$DATA_DIR" \
    --output "$CQL_OUTPUT" \
    --seeds "$TRAIN_SEEDS" \
    --val-seeds "$VAL_SEEDS" \
    --gradient-steps 150000 \
    --hidden-layers 256,256 \
    --eval-every 2500 \
    --cql-alpha 0.2 \
    --device cpu
echo "[pipeline] CQL training done: $(date)"

echo ""
echo "[pipeline] All training complete: $(date)"
echo "[pipeline] IQL models: $IQL_OUTPUT"
echo "[pipeline] CQL models: $CQL_OUTPUT"
echo "[pipeline] Next: run benchmark_entity_agents.py to compare vs RBCSmartPolicy"
