#!/bin/bash
# Monitor training and run validation every 5 steps

# Configuration
VAL_INTERVAL=5
LOG_FILE="training.log"
REWARD_LOG="reward_samples.log"
VAL_METRICS_FILE="validation_metrics.jsonl"

# Initialize
LAST_VALIDATED_STEP=-1
echo "Starting training monitor..."
echo "Validation interval: every $VAL_INTERVAL steps"
echo ""

# Function to extract current step from training log
get_current_step() {
    if [ -f "$LOG_FILE" ]; then
        # Look for patterns like "20/25" in training progress
        STEP=$(tail -50 "$LOG_FILE" | grep -oP '\d+/\d+' | tail -1 | cut -d'/' -f1)
        if [ -n "$STEP" ]; then
            echo "$STEP"
            return
        fi
    fi
    echo "0"
}

# Function to run validation
run_validation() {
    local step=$1
    echo ""
    echo "======================================================================="
    echo "VALIDATION AT STEP $step"
    echo "======================================================================="
    
    # Run validation script
    uv run python GRPO/compute_passatk.py --log-file "$REWARD_LOG" --wandb --step "$step"
    
    echo "Validation complete at step $step"
    echo ""
}

# Main monitoring loop
while true; do
    # Get current training step
    CURRENT_STEP=$(get_current_step)
    
    # Check if we should validate
    if [ "$CURRENT_STEP" -gt 0 ]; then
        # Check if step is multiple of VAL_INTERVAL
        if [ $((CURRENT_STEP % VAL_INTERVAL)) -eq 0 ]; then
            # Only validate if we haven't validated this step yet
            if [ "$CURRENT_STEP" -gt "$LAST_VALIDATED_STEP" ]; then
                run_validation "$CURRENT_STEP"
                LAST_VALIDATED_STEP=$CURRENT_STEP
            fi
        fi
        
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training at step $CURRENT_STEP (next validation at step $(( (CURRENT_STEP / VAL_INTERVAL + 1) * VAL_INTERVAL )))"
    fi
    
    # Wait before checking again
    sleep 30
    
    # Check if training is still running
    if ! pgrep -f "train_from_config.py" > /dev/null; then
        echo "Training process not found. Exiting monitor."
        break
    fi
done

echo "Monitor stopped."

