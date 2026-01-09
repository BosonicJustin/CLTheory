#!/bin/bash
#
# Run linear probe evaluation on all completed experiments
#
# Usage:
#   ./run_linear_probe_all.sh [--type regular|adjusted] [--force]
#
# Options:
#   --type    Which experiment type: 'regular' or 'adjusted' (default: regular)
#   --force   Re-run linear probe even if already completed
#

# Parse arguments
EXP_TYPE="regular"
FORCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --type)
            EXP_TYPE="$2"
            shift 2
            ;;
        --force)
            FORCE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--type regular|adjusted] [--force]"
            exit 1
            ;;
    esac
done

# Configuration based on experiment type
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$EXP_TYPE" = "regular" ]; then
    RESULTS_DIR="${SCRIPT_DIR}/experiment_results"
    EPOCHS=200
elif [ "$EXP_TYPE" = "adjusted" ]; then
    RESULTS_DIR="${SCRIPT_DIR}/experiment_results_adjusted"
    EPOCHS=200
else
    echo "Unknown experiment type: $EXP_TYPE"
    echo "Use --type regular or --type adjusted"
    exit 1
fi

ENCODERS=("resnet" "vit" "mlp")
AUG_MODES=("all" "crop" "all-no-crop")
NUM_RUNS=5

# Linear probe settings
LP_EPOCHS=100
LP_BATCH_SIZE=256
LP_LR=0.1

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Linear Probe Evaluation - All Experiments"
echo "=========================================="
echo "Type: $EXP_TYPE"
echo "Results directory: $RESULTS_DIR"
echo "Linear probe epochs: $LP_EPOCHS"
echo "Force re-run: $FORCE"
echo "=========================================="
echo ""

# Track counts
COMPLETE=0
ALREADY_DONE=0
SKIPPED=0
FAILED=0

# Arrays to store experiments to process
declare -a TO_PROCESS

# First pass: identify experiments to process
for encoder in "${ENCODERS[@]}"; do
    for aug_mode in "${AUG_MODES[@]}"; do
        for run in $(seq 1 ${NUM_RUNS}); do
            EXP_NAME="${encoder}_${aug_mode}_run${run}"
            EXP_DIR="${RESULTS_DIR}/${encoder}/${aug_mode}/run_${run}"

            # Check if experiment is complete (has final checkpoint)
            if [ ! -f "${EXP_DIR}/checkpoint_epoch_${EPOCHS}.pt" ]; then
                echo -e "${RED}[SKIP]${NC} $EXP_NAME - not complete (no final checkpoint)"
                SKIPPED=$((SKIPPED + 1))
                continue
            fi

            # Check if linear probe already done
            if [ -f "${EXP_DIR}/linear_probe_summary.json" ] && [ "$FORCE" = false ]; then
                # Extract best accuracy from summary
                BEST_ACC=$(grep -o '"best_test_acc": [0-9.]*' "${EXP_DIR}/linear_probe_summary.json" | grep -o '[0-9.]*')
                echo -e "${GREEN}[DONE]${NC} $EXP_NAME - best acc: ${BEST_ACC}%"
                ALREADY_DONE=$((ALREADY_DONE + 1))
                continue
            fi

            # Add to processing list
            TO_PROCESS+=("${encoder}|${aug_mode}|${run}")
        done
    done
done

TOTAL_TO_PROCESS=${#TO_PROCESS[@]}

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo -e "${GREEN}Already done:${NC}  $ALREADY_DONE"
echo -e "${RED}Skipped:${NC}       $SKIPPED"
echo -e "${BLUE}To process:${NC}    $TOTAL_TO_PROCESS"
echo "=========================================="

if [ $TOTAL_TO_PROCESS -eq 0 ]; then
    echo ""
    echo "Nothing to process!"
    exit 0
fi

echo ""
echo "Starting linear probe evaluation for $TOTAL_TO_PROCESS experiments..."
echo ""

# Create log directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p "${LOG_DIR}"
LP_LOG="${LOG_DIR}/linear_probe_${TIMESTAMP}.txt"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LP_LOG}"
}

# Process each experiment
CURRENT=0
for exp_info in "${TO_PROCESS[@]}"; do
    IFS='|' read -r encoder aug_mode run <<< "$exp_info"
    CURRENT=$((CURRENT + 1))

    EXP_NAME="${encoder}_${aug_mode}_run${run}"
    EXP_DIR="${RESULTS_DIR}/${encoder}/${aug_mode}/run_${run}"

    log "------------------------------------------"
    log "Processing ${CURRENT}/${TOTAL_TO_PROCESS}: ${EXP_NAME}"
    log "Directory: ${EXP_DIR}"

    EXP_START_TIME=$(date +%s)

    # Run linear probe
    python "${SCRIPT_DIR}/linear_probe_eval.py" \
        --run-dir "${EXP_DIR}" \
        --epochs "${LP_EPOCHS}" \
        --batch-size "${LP_BATCH_SIZE}" \
        --lr "${LP_LR}" \
        2>&1 | tee -a "${LP_LOG}"

    EXIT_CODE=${PIPESTATUS[0]}

    EXP_END_TIME=$(date +%s)
    EXP_DURATION=$(( (EXP_END_TIME - EXP_START_TIME) / 60 ))

    if [ $EXIT_CODE -eq 0 ]; then
        # Extract best accuracy from summary
        if [ -f "${EXP_DIR}/linear_probe_summary.json" ]; then
            BEST_ACC=$(grep -o '"best_test_acc": [0-9.]*' "${EXP_DIR}/linear_probe_summary.json" | grep -o '[0-9.]*')
            log "Completed in ${EXP_DURATION} min - Best acc: ${BEST_ACC}%"
        else
            log "Completed in ${EXP_DURATION} min"
        fi
        COMPLETE=$((COMPLETE + 1))
    else
        log "FAILED with exit code $EXIT_CODE"
        FAILED=$((FAILED + 1))
    fi
done

log "=========================================="
log "Linear Probe Evaluation Complete"
log "=========================================="
log "Successful: $COMPLETE"
log "Failed: $FAILED"
log "=========================================="

# Generate summary CSV
SUMMARY_CSV="${RESULTS_DIR}/linear_probe_summary_${TIMESTAMP}.csv"
log "Generating summary CSV: ${SUMMARY_CSV}"

echo "encoder,aug_mode,run,best_test_acc,final_test_acc,lp_epochs" > "${SUMMARY_CSV}"

for encoder in "${ENCODERS[@]}"; do
    for aug_mode in "${AUG_MODES[@]}"; do
        for run in $(seq 1 ${NUM_RUNS}); do
            EXP_DIR="${RESULTS_DIR}/${encoder}/${aug_mode}/run_${run}"
            SUMMARY_FILE="${EXP_DIR}/linear_probe_summary.json"

            if [ -f "${SUMMARY_FILE}" ]; then
                BEST_ACC=$(grep -o '"best_test_acc": [0-9.]*' "${SUMMARY_FILE}" | grep -o '[0-9.]*')
                FINAL_ACC=$(grep -o '"final_test_acc": [0-9.]*' "${SUMMARY_FILE}" | grep -o '[0-9.]*')
                LP_EPS=$(grep -o '"total_epochs": [0-9]*' "${SUMMARY_FILE}" | grep -o '[0-9]*')
                echo "${encoder},${aug_mode},${run},${BEST_ACC},${FINAL_ACC},${LP_EPS}" >> "${SUMMARY_CSV}"
            else
                echo "${encoder},${aug_mode},${run},,,," >> "${SUMMARY_CSV}"
            fi
        done
    done
done

log "Summary CSV generated: ${SUMMARY_CSV}"
log "Done!"
