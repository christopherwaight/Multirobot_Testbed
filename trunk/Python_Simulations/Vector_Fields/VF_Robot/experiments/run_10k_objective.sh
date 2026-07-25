#!/bin/bash
# Final 10,000-trial Monte Carlo sweeps, objective (rotation-equivariant) controller.
#
# Run from the VF_Robot root:
#   bash experiments/run_10k_objective.sh
#
# Sequential on purpose: each sweep already saturates 8 workers, so running
# them in parallel would only thrash. Projected ~5 h total on an 8-core M-series
# (calibrated from 20-trial runs: 29.5 s, 4.3 s, 4.1 s per sweep).
#
# Pre-fix 500-trial CSVs are archived read-only at
#   experiments/outputs/mc_oecs_traverse/archive_500_percomponent_presat/

set -u  # not -e: a failed sweep should not abort the ones after it

cd "$(dirname "$0")/.." || exit 1

TRIALS=10000
# 6, not 8. This is an 8 GB machine: two 10,000-trial runs were killed by
# memory pressure (swap at 5.8 of 7.2 GB, ~60 MB free). The parent no longer
# accumulates trial rows, but each worker still batches its own results, so
# leaving headroom matters more here than the last 25 percent of throughput.
WORKERS=6
LOG_DIR="experiments/outputs/mc_oecs_traverse/logs_10k_objective"
mkdir -p "$LOG_DIR"

echo "=== 10k objective sweeps started $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
echo "commit: $(git rev-parse --short HEAD)"
echo "trials/cell: $TRIALS  workers: $WORKERS"
echo

run_sweep () {
    local name="$1"; shift
    local start
    start=$(date +%s)
    echo "--- $name starting $(date '+%H:%M:%S') ---"
    if ./venv/bin/python3 "$@" --trials "$TRIALS" --workers "$WORKERS" \
            > "$LOG_DIR/${name}.log" 2>&1; then
        echo "--- $name DONE in $(( ($(date +%s) - start) / 60 )) min ---"
    else
        echo "--- $name FAILED (exit $?), see $LOG_DIR/${name}.log ---"
    fi
    echo
}

run_sweep oecs_traverse       experiments/mc_sweep_oecs_traverse.py
run_sweep flip_resolution     experiments/mc_sweep_flip_resolution.py
run_sweep flip_resolution_sp  experiments/mc_sweep_flip_resolution_sigma_p.py

echo "=== all sweeps finished $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
