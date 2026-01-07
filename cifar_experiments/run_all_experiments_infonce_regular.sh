#!/bin/bash
#
# Run all SimCLR experiments on CIFAR-10
# 3 encoders × 3 augmentation modes × 5 runs = 45 total experiments
#

set -e  # Exit on error

# Configuration
ENCODERS=("resnet" "vit" "mlp")
AUG_MODES=("all" "crop" "all-no-crop")
NUM_RUNS=5
EPOCHS=200
BATCH_SIZE=2000
TEMPERATURE=0.5
LR=3e-4
WEIGHT_DECAY=1e-4
SAVE_FREQ=50
VAL_FREQ=10

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/experiment_results"
LOG_DIR="${RESULTS_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MASTER_LOG="${LOG_DIR}/master_log_${TIMESTAMP}.txt"

# Create directories
mkdir -p "${RESULTS_DIR}"
mkdir -p "${LOG_DIR}"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${MASTER_LOG}"
}

# Count total experiments
TOTAL_EXPERIMENTS=$((${#ENCODERS[@]} * ${#AUG_MODES[@]} * NUM_RUNS))
CURRENT_EXPERIMENT=0

log "=========================================="
log "SimCLR CIFAR-10 Experiment Suite"
log "=========================================="
log "Total experiments: ${TOTAL_EXPERIMENTS}"
log "Encoders: ${ENCODERS[*]}"
log "Augmentation modes: ${AUG_MODES[*]}"
log "Runs per config: ${NUM_RUNS}"
log "Epochs per run: ${EPOCHS}"
log "Batch size: ${BATCH_SIZE}"
log "Temperature: ${TEMPERATURE}"
log "Learning rate: ${LR}"
log "Results directory: ${RESULTS_DIR}"
log "=========================================="

# Track timing
SUITE_START_TIME=$(date +%s)

# Run experiments
for encoder in "${ENCODERS[@]}"; do
    for aug_mode in "${AUG_MODES[@]}"; do
        for run in $(seq 1 ${NUM_RUNS}); do
            CURRENT_EXPERIMENT=$((CURRENT_EXPERIMENT + 1))

            # Create experiment name and directory
            EXP_NAME="${encoder}_${aug_mode}_run${run}"
            EXP_DIR="${RESULTS_DIR}/${encoder}/${aug_mode}/run_${run}"
            EXP_LOG="${LOG_DIR}/${EXP_NAME}_${TIMESTAMP}.log"

            log "------------------------------------------"
            log "Experiment ${CURRENT_EXPERIMENT}/${TOTAL_EXPERIMENTS}: ${EXP_NAME}"
            log "Output directory: ${EXP_DIR}"

            # Check if experiment already completed
            if [ -f "${EXP_DIR}/config.json" ] && [ -f "${EXP_DIR}/checkpoint_epoch_${EPOCHS}.pt" ]; then
                log "SKIPPING: Experiment already completed"
                continue
            fi

            # Create experiment directory
            mkdir -p "${EXP_DIR}"

            # Record start time
            EXP_START_TIME=$(date +%s)

            # Run experiment
            log "Starting training..."

            python "${SCRIPT_DIR}/experiments.py" \
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
                2>&1 | tee "${EXP_LOG}"

            # Record end time and duration
            EXP_END_TIME=$(date +%s)
            EXP_DURATION=$((EXP_END_TIME - EXP_START_TIME))
            EXP_DURATION_MIN=$((EXP_DURATION / 60))

            log "Completed in ${EXP_DURATION_MIN} minutes"

            # Estimate remaining time
            ELAPSED=$((EXP_END_TIME - SUITE_START_TIME))
            AVG_TIME=$((ELAPSED / CURRENT_EXPERIMENT))
            REMAINING_EXPERIMENTS=$((TOTAL_EXPERIMENTS - CURRENT_EXPERIMENT))
            ESTIMATED_REMAINING=$((AVG_TIME * REMAINING_EXPERIMENTS / 60))

            log "Estimated remaining time: ~${ESTIMATED_REMAINING} minutes"
        done
    done
done

# Final summary
SUITE_END_TIME=$(date +%s)
TOTAL_DURATION=$(((SUITE_END_TIME - SUITE_START_TIME) / 60))

log "=========================================="
log "ALL EXPERIMENTS COMPLETED"
log "Total time: ${TOTAL_DURATION} minutes"
log "Results saved to: ${RESULTS_DIR}"
log "=========================================="

# Generate summary CSV
SUMMARY_CSV="${RESULTS_DIR}/experiment_summary_${TIMESTAMP}.csv"
log "Generating summary CSV: ${SUMMARY_CSV}"

echo "encoder,aug_mode,run,status,config_exists,final_checkpoint_exists" > "${SUMMARY_CSV}"

for encoder in "${ENCODERS[@]}"; do
    for aug_mode in "${AUG_MODES[@]}"; do
        for run in $(seq 1 ${NUM_RUNS}); do
            EXP_DIR="${RESULTS_DIR}/${encoder}/${aug_mode}/run_${run}"

            CONFIG_EXISTS="no"
            CHECKPOINT_EXISTS="no"
            STATUS="incomplete"

            if [ -f "${EXP_DIR}/config.json" ]; then
                CONFIG_EXISTS="yes"
            fi

            if [ -f "${EXP_DIR}/checkpoint_epoch_${EPOCHS}.pt" ]; then
                CHECKPOINT_EXISTS="yes"
                STATUS="complete"
            fi

            echo "${encoder},${aug_mode},${run},${STATUS},${CONFIG_EXISTS},${CHECKPOINT_EXISTS}" >> "${SUMMARY_CSV}"
        done
    done
done

log "Summary CSV generated"
log "Done!"
