#!/usr/bin/env bash
# Run the bounded local TPPO BC gates before starting a Wave A server run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${LOG_DIR:-runs/local_bc_checks}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "$LOG_DIR" = /* ]]; then
    BASE_DIR="$LOG_DIR"
else
    BASE_DIR="$REPO_ROOT/$LOG_DIR"
fi

mkdir -p "$BASE_DIR"

run_check() {
    local name="$1"
    local config="$2"
    local job_id="tppo-bc-local-${name}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
    local job_dir="$BASE_DIR/jobs/$job_id"
    local log_file="$job_dir/logs/$job_id.log"
    local transcript="$BASE_DIR/${job_id}.stdout.log"

    printf 'Running %s: python run_experiment.py --config %s --job_id %s\n' \
        "$name" "$config" "$job_id"
    (
        cd "$REPO_ROOT"
        "$PYTHON_BIN" run_experiment.py \
            --config "$config" \
            --job_id "$job_id" \
            --base-dir "$BASE_DIR"
    ) 2>&1 | tee "$transcript"

    if [[ ! -f "$log_file" ]]; then
        printf 'Missing experiment log: %s\n' "$log_file" >&2
        return 1
    fi

    if ! grep -Eq 'Completed[[:space:]]+episode[[:space:]]+3/3([,[:space:]]|$)' "$log_file"; then
        printf 'Run %s did not complete episode 3/3.\n' "$name" >&2
        return 1
    fi

    # Require positive usable samples for every configured BC building. Accept
    # both human-readable logs and JSON-style metric keys.
    while IFS= read -r building; do
        if ! grep -Eiq "${building}[^[:cntrl:]]*(usable[ _-]?samples|demonstration[ _-]?samples)[^[:digit:]]*[1-9][0-9]*(\\.[0-9]+)?|behavior_cloning_building_${building}_(usable|demonstration)_samples\\\"?[[:space:]]*:[[:space:]]*[1-9][0-9]*(\\.[0-9]+)?" "$log_file" "$job_dir/logs/metrics.jsonl" 2>/dev/null; then
            printf 'Run %s has no positive usable BC samples for %s.\n' "$name" "$building" >&2
            return 1
        fi
    done < <(
        "$PYTHON_BIN" - "$REPO_ROOT/$config" "$REPO_ROOT" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
repo_root = Path(sys.argv[2])
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
dataset_path = Path(config["simulator"]["dataset_path"])
if not dataset_path.is_absolute():
    dataset_path = repo_root / dataset_path
schema = json.loads(dataset_path.read_text(encoding="utf-8"))
print("\n".join(schema["buildings"]))
PY
    )

    if ! grep -Eiq '(trained[ _-]?batches|pretraining[ _-]?batches|behavior_cloning(_pretraining)?_trained_batches)[^[:digit:]]*[1-9][0-9]*(\.[0-9]+)?' "$log_file" "$job_dir/logs/metrics.jsonl" 2>/dev/null; then
        printf 'Run %s has no positive BC trained-batch total.\n' "$name" >&2
        return 1
    fi

    if grep -Eiq 'Skipping[[:space:]]+behavior-cloning[[:space:]]+demonstrations' "$log_file" "$job_dir/logs/metrics.jsonl" 2>/dev/null; then
        printf 'Run %s emitted the obsolete skipped-demonstrations warning.\n' "$name" >&2
        return 1
    fi

    if grep -Eiq 'stall[ _-]?watchdog[^[:cntrl:]]*(activat|fired|timed[ _-]?out|timeout|traceback)|Traceback \(most recent call last\)' "$log_file" "$job_dir/logs/metrics.jsonl" 2>/dev/null; then
        printf 'Run %s has watchdog failure or traceback evidence.\n' "$name" >&2
        return 1
    fi

    printf 'PASS %s: %s\n' "$name" "$job_id"
}

run_check canary configs/recovery/tppo/wave_a/local/tppo_bc_pretrain_canary.yaml
run_check smoke configs/recovery/tppo/wave_a/local/tppo_bc_pretrain_smoke.yaml
