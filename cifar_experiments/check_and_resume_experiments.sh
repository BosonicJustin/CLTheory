#!/bin/bash
#
# Check experiment status and optionally resume incomplete experiments
#
# Usage:
#   ./check_and_resume_experiments.sh [--check-only] [--resume] [--type regular|adjusted]
#
# Options:
#   --check-only   Only report status, don't run anything (default)
#   --resume       Resume incomplete experiments
#   --type         Which experiment type: 'regular' or 'adjusted' (default: regular)
#

set -e

# Parse arguments
CHECK_ONLY=true
EXP_TYPE="regular"

while [[ $# -gt 0 ]]; do
    case $1 in
        --check-only)
            CHECK_ONLY=true
            shift
            ;;
        --resume)
            CHECK_ONLY=false
            shift
            ;;
        --type)
            EXP_TYPE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--check-only] [--resume] [--type regular|adjusted]"
            exit 1
            ;;
    esac
done

# Configuration based on experiment type
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$EXP_TYPE" = "regular" ]; then
    RESULTS_DIR="${SCRIPT_DIR}/experiment_results"
    EXPERIMENT_SCRIPT="experiments.py"
    EPOCHS=200
    BATCH_SIZE=2000
    EXTRA_ARGS=""
elif [ "$EXP_TYPE" = "adjusted" ]; then
    RESULTS_DIR="${SCRIPT_DIR}/experiment_results_adjusted"
    EXPERIMENT_SCRIPT="experiments_adjusted.py"
    EPOCHS=200
    BATCH_SIZE=64
    NUM_NEGATIVES=256
    EXTRA_ARGS="--num-negatives ${NUM_NEGATIVES}"
else
    echo "Unknown experiment type: $EXP_TYPE"
    echo "Use --type regular or --type adjusted"
    exit 1
fi

ENCODERS=("resnet" "vit" "mlp")
AUG_MODES=("all" "crop" "all-no-crop")
NUM_RUNS=5
TEMPERATURE=0.5
LR=3e-4
WEIGHT_DECAY=1e-4
SAVE_FREQ=50
VAL_FREQ=10

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Experiment Status Check"
echo "=========================================="
echo "Type: $EXP_TYPE"
echo "Results directory: $RESULTS_DIR"
echo "Expected epochs: $EPOCHS"
echo "=========================================="
echo ""

# Track counts
COMPLETE=0
INCOMPLETE=0
NOT_STARTED=0
PARTIAL=0

# Arrays to store incomplete experiments
declare -a INCOMPLETE_EXPS

# Check each experiment
for encoder in "${ENCODERS[@]}"; do
    for aug_mode in "${AUG_MODES[@]}"; do
        for run in $(seq 1 ${NUM_RUNS}); do
            EXP_NAME="${encoder}_${aug_mode}_run${run}"
            EXP_DIR="${RESULTS_DIR}/${encoder}/${aug_mode}/run_${run}"

            CONFIG_EXISTS=false
            FINAL_CHECKPOINT_EXISTS=false
            LATEST_CHECKPOINT=""
            LATEST_EPOCH=0

            # Check for config
            if [ -f "${EXP_DIR}/config.json" ]; then
                CONFIG_EXISTS=true
            fi

            # Check for final checkpoint
            if [ -f "${EXP_DIR}/checkpoint_epoch_${EPOCHS}.pt" ]; then
                FINAL_CHECKPOINT_EXISTS=true
            fi

            # Find latest checkpoint if not complete
            if [ "$FINAL_CHECKPOINT_EXISTS" = false ] && [ -d "$EXP_DIR" ]; then
                for ckpt in "${EXP_DIR}"/checkpoint_epoch_*.pt; do
                    if [ -f "$ckpt" ]; then
                        # Extract epoch number
                        epoch_num=$(echo "$ckpt" | grep -o 'epoch_[0-9]*' | grep -o '[0-9]*')
                        if [ -n "$epoch_num" ] && [ "$epoch_num" -gt "$LATEST_EPOCH" ]; then
                            LATEST_EPOCH=$epoch_num
                            LATEST_CHECKPOINT=$ckpt
                        fi
                    fi
                done
            fi

            # Determine status
            if [ "$FINAL_CHECKPOINT_EXISTS" = true ]; then
                echo -e "${GREEN}[COMPLETE]${NC} $EXP_NAME"
                COMPLETE=$((COMPLETE + 1))
            elif [ "$CONFIG_EXISTS" = true ]; then
                if [ "$LATEST_EPOCH" -gt 0 ]; then
                    echo -e "${YELLOW}[PARTIAL]${NC}  $EXP_NAME - stopped at epoch $LATEST_EPOCH/$EPOCHS"
                    PARTIAL=$((PARTIAL + 1))
                else
                    echo -e "${YELLOW}[STARTED]${NC} $EXP_NAME - config exists but no checkpoints"
                    PARTIAL=$((PARTIAL + 1))
                fi
                INCOMPLETE_EXPS+=("${encoder}|${aug_mode}|${run}")
            else
                echo -e "${RED}[NOT STARTED]${NC} $EXP_NAME"
                NOT_STARTED=$((NOT_STARTED + 1))
                INCOMPLETE_EXPS+=("${encoder}|${aug_mode}|${run}")
            fi
        done
    done
done

TOTAL_INCOMPLETE=$((PARTIAL + NOT_STARTED))

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo -e "${GREEN}Complete:${NC}    $COMPLETE"
echo -e "${YELLOW}Partial:${NC}     $PARTIAL"
echo -e "${RED}Not Started:${NC} $NOT_STARTED"
echo "----------------------------------------"
echo "Total Incomplete: $TOTAL_INCOMPLETE"
echo "=========================================="

# If check-only mode, exit here
if [ "$CHECK_ONLY" = true ]; then
    if [ $TOTAL_INCOMPLETE -gt 0 ]; then
        echo ""
        echo "To resume incomplete experiments, run:"
        echo "  $0 --resume --type $EXP_TYPE"
    fi
    exit 0
fi

# Resume mode - run incomplete experiments
if [ $TOTAL_INCOMPLETE -eq 0 ]; then
    echo "All experiments complete! Nothing to resume."
    exit 0
fi

echo ""
echo "Resuming $TOTAL_INCOMPLETE incomplete experiments..."
echo ""

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p "${LOG_DIR}"
RESUME_LOG="${LOG_DIR}/resume_log_${TIMESTAMP}.txt"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${RESUME_LOG}"
}

CURRENT=0
for exp_info in "${INCOMPLETE_EXPS[@]}"; do
    IFS='|' read -r encoder aug_mode run <<< "$exp_info"
    CURRENT=$((CURRENT + 1))

    EXP_NAME="${encoder}_${aug_mode}_run${run}"
    EXP_DIR="${RESULTS_DIR}/${encoder}/${aug_mode}/run_${run}"
    EXP_LOG="${LOG_DIR}/${EXP_NAME}_resume_${TIMESTAMP}.log"

    log "------------------------------------------"
    log "Resuming ${CURRENT}/${TOTAL_INCOMPLETE}: ${EXP_NAME}"
    log "Output directory: ${EXP_DIR}"

    # Safety check: only delete if path matches expected pattern
    # Expected: .../experiment_results[_adjusted]/{encoder}/{aug_mode}/run_{n}
    if [ -d "${EXP_DIR}" ]; then
        if [[ "${EXP_DIR}" == */run_[0-9]* ]] && [[ "${EXP_DIR}" == *experiment_results* ]]; then
            log "Removing incomplete experiment directory: ${EXP_DIR}"
            rm -rf "${EXP_DIR}"
        else
            log "WARNING: Skipping deletion - path doesn't match expected pattern: ${EXP_DIR}"
        fi
    fi
    mkdir -p "${EXP_DIR}"

    EXP_START_TIME=$(date +%s)

    python "${SCRIPT_DIR}/${EXPERIMENT_SCRIPT}" \
        --model "${encoder}" \
        --aug-mode "${aug_mode}" \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --temperature "${TEMPERATURE}" \
        --lr "${LR}" \
        --weight-decay "${WEIGHT_DECAY}" \
        --save-freq "${SAVE_FREQ}" \
        --val-freq "${VAL_FREQ}" \
        --save-dir "${EXP_DIR}" \
        --data-root "${SCRIPT_DIR}/data" \
        ${EXTRA_ARGS} \
        2>&1 | tee "${EXP_LOG}"

    EXP_END_TIME=$(date +%s)
    EXP_DURATION=$(( (EXP_END_TIME - EXP_START_TIME) / 60 ))

    log "Completed in ${EXP_DURATION} minutes"
done

log "=========================================="
log "All incomplete experiments resumed!"
log "=========================================="
