# Training Efficiency Monitoring Guide

This guide explains how to monitor and compare training efficiency, especially when comparing runs with different logging settings.

## Quick Start

### 1. Monitor Current Training

```bash
# Monitor efficiency metrics from current training log
python GRPO/monitor_efficiency.py --log-file training.log

# Export metrics to JSON for later comparison
python GRPO/monitor_efficiency.py --log-file training.log --export metrics.json
```

### 2. Compare Two Runs

**Option A: Compare from log files**
```bash
./compare_training_runs.sh training_with_logging.log training_no_logging.log
```

**Option B: Compare from exported metrics**
```bash
python GRPO/monitor_efficiency.py --compare metrics_run1.json metrics_run2.json
```

## Metrics Tracked

The monitoring script tracks:

- **Average Step Time**: Time per training step (seconds)
- **Steps per Second**: Training throughput
- **Total Training Time**: Wall-clock time for training
- **Total Reward Calls**: Number of reward function calls
- **Samples per Step**: Average samples processed per step

## Comparing Logging Settings

### Step 1: Run Training with Logging Enabled

```yaml
# config.yaml
environment:
  reward_enable_logging: "true"
  reward_enable_console_output: "true"  # Enable for comparison
```

```bash
python GRPO/train_from_config.py --config GRPO/config.yaml
# After training completes:
python GRPO/monitor_efficiency.py --log-file training.log --export metrics_with_logging.json
```

### Step 2: Run Training with Logging Disabled

```yaml
# config.yaml
environment:
  reward_enable_logging: "true"  # File logging still needed for validation
  reward_enable_console_output: "false"  # Disable console output
```

```bash
python GRPO/train_from_config.py --config GRPO/config.yaml
# After training completes:
python GRPO/monitor_efficiency.py --log-file training.log --export metrics_no_logging.json
```

### Step 3: Compare Results

```bash
python GRPO/monitor_efficiency.py --compare metrics_with_logging.json metrics_no_logging.json
```

## Expected Results

When comparing console logging enabled vs disabled, you should see:

- **Faster step times** when console output is disabled
- **Higher throughput** (steps/sec) when console output is disabled
- **Similar total reward calls** (both runs process same data)
- **Performance improvement** typically 1-5% depending on:
  - Number of workers
  - Logging frequency
  - System I/O performance

## Logging to WandB

You can also log efficiency metrics to WandB for visualization:

```bash
python GRPO/monitor_efficiency.py --log-file training.log --wandb
```

This will log metrics like:
- `efficiency/avg_step_time`
- `efficiency/steps_per_second`
- `efficiency/total_training_time`
- `efficiency/total_reward_calls`

## Tips

1. **Run multiple steps**: Compare runs with at least 10-20 steps for meaningful statistics
2. **Same hardware**: Compare runs on the same hardware for accurate results
3. **Same config**: Use identical training configs except for logging settings
4. **Warm-up**: Ignore first few steps (warm-up period) when comparing

## Troubleshooting

**No step times found**: The script looks for "Training Progress: X/Y" patterns in logs. If your logs use different format, you may need to adjust the regex patterns.

**Missing reward calls**: Ensure `reward_samples.log` exists and contains sample entries.

**Timing not accurate**: For more accurate timing, consider adding timing instrumentation directly in the reward function (see `REWARD_ENABLE_TIMING` env var).

