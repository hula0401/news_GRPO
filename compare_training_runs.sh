#!/bin/bash
# Compare training efficiency between two runs with different logging settings
#
# Usage:
#   ./compare_training_runs.sh run1_metrics.json run2_metrics.json
#   OR
#   ./compare_training_runs.sh training_log1.log training_log2.log

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <run1_log_or_metrics> <run2_log_or_metrics>"
    echo ""
    echo "Examples:"
    echo "  # Compare from log files (will export metrics first)"
    echo "  $0 training.log training_no_logging.log"
    echo ""
    echo "  # Compare from exported metrics"
    echo "  $0 metrics_with_logging.json metrics_no_logging.json"
    exit 1
fi

RUN1=$1
RUN2=$2

# Check if files are JSON (metrics) or log files
if [[ "$RUN1" == *.json ]]; then
    METRICS1=$RUN1
else
    echo "Exporting metrics from $RUN1..."
    METRICS1="${RUN1%.*}_metrics.json"
    python GRPO/monitor_efficiency.py --log-file "$RUN1" --export "$METRICS1"
fi

if [[ "$RUN2" == *.json ]]; then
    METRICS2=$RUN2
else
    echo "Exporting metrics from $RUN2..."
    METRICS2="${RUN2%.*}_metrics.json"
    python GRPO/monitor_efficiency.py --log-file "$RUN2" --export "$METRICS2"
fi

echo ""
echo "Comparing runs..."
python GRPO/monitor_efficiency.py --compare "$METRICS1" "$METRICS2"

