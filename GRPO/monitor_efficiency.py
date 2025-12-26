"""
Training Efficiency Monitoring Script

This script monitors training efficiency metrics including:
- Throughput (samples/sec, steps/sec)
- Time per step
- Reward function call overhead
- Comparison between logging enabled/disabled runs

Usage:
    # Monitor current training
    python GRPO/monitor_efficiency.py --log-file training.log

    # Compare two runs
    python GRPO/monitor_efficiency.py --compare run1.json run2.json

    # Export metrics to JSON for comparison
    python GRPO/monitor_efficiency.py --log-file training.log --export metrics.json
"""

import re
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingEfficiencyMonitor:
    """Monitor training efficiency metrics from logs."""
    
    def __init__(self, log_file: Path, reward_log: Optional[Path] = None):
        self.log_file = log_file
        self.reward_log = reward_log or Path("reward_samples.log")
        self.metrics = {
            'steps': [],
            'step_times': [],
            'samples_processed': [],
            'reward_calls': 0,
            'reward_call_times': [],
            'start_time': None,
            'end_time': None,
            'config': {},
        }
    
    def parse_training_log(self) -> Dict:
        """Parse training log to extract efficiency metrics."""
        if not self.log_file.exists():
            logger.error(f"Training log not found: {self.log_file}")
            return {}
        
        logger.info(f"Parsing {self.log_file}...")
        
        step_pattern = re.compile(r'Training Progress:\s*(\d+)/(\d+)')
        step_time_pattern = re.compile(r'step[:\s]+(\d+)')
        timestamp_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})')
        
        current_step = None
        step_times = {}
        timestamps = []
        
        with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
            # Extract start time
            for line in lines[:100]:
                match = timestamp_pattern.search(line)
                if match:
                    self.metrics['start_time'] = match.group(1)
                    break
            
            # Extract step information
            for i, line in enumerate(lines):
                # Look for step progress
                step_match = step_pattern.search(line)
                if step_match:
                    step = int(step_match.group(1))
                    total = int(step_match.group(2))
                    
                    if step not in step_times:
                        # Find timestamp for this step
                        timestamp_match = timestamp_pattern.search(line)
                        if timestamp_match:
                            step_times[step] = timestamp_match.group(1)
                            timestamps.append((step, timestamp_match.group(1)))
                            self.metrics['steps'].append(step)
            
            # Extract end time
            for line in reversed(lines[-100:]):
                match = timestamp_pattern.search(line)
                if match:
                    self.metrics['end_time'] = match.group(1)
                    break
        
        # Calculate step times
        if len(timestamps) >= 2:
            for i in range(1, len(timestamps)):
                step, ts_str = timestamps[i]
                prev_step, prev_ts_str = timestamps[i-1]
                
                try:
                    ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                    prev_ts = datetime.strptime(prev_ts_str, "%Y-%m-%d %H:%M:%S")
                    step_time = (ts - prev_ts).total_seconds()
                    
                    if step_time > 0 and step_time < 3600:  # Reasonable range
                        self.metrics['step_times'].append({
                            'step': step,
                            'time': step_time,
                            'timestamp': ts_str
                        })
                except ValueError:
                    continue
        
        # Count reward function calls from reward log
        if self.reward_log.exists():
            self.metrics['reward_calls'] = self._count_reward_calls()
        
        return self.metrics
    
    def _count_reward_calls(self) -> int:
        """Count total reward function calls from reward log."""
        count = 0
        try:
            with open(self.reward_log, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if 'ROLLOUT SAMPLE' in line or 'Call #' in line:
                        count += 1
        except Exception as e:
            logger.warning(f"Could not count reward calls: {e}")
        return count
    
    def calculate_metrics(self) -> Dict:
        """Calculate efficiency metrics from parsed data."""
        metrics = {}
        
        # Step timing metrics
        if self.metrics['step_times']:
            step_times = [s['time'] for s in self.metrics['step_times']]
            metrics['avg_step_time'] = sum(step_times) / len(step_times)
            metrics['min_step_time'] = min(step_times)
            metrics['max_step_time'] = max(step_times)
            metrics['total_steps'] = len(self.metrics['step_times'])
            
            # Throughput (steps per second)
            metrics['steps_per_second'] = 1.0 / metrics['avg_step_time'] if metrics['avg_step_time'] > 0 else 0
        
        # Total training time
        if self.metrics['start_time'] and self.metrics['end_time']:
            try:
                start = datetime.strptime(self.metrics['start_time'], "%Y-%m-%d %H:%M:%S")
                end = datetime.strptime(self.metrics['end_time'], "%Y-%m-%d %H:%M:%S")
                metrics['total_training_time'] = (end - start).total_seconds()
            except ValueError:
                pass
        
        # Reward call metrics
        metrics['total_reward_calls'] = self.metrics['reward_calls']
        
        # Samples processed (estimated from reward calls)
        if metrics['total_reward_calls'] > 0 and metrics.get('total_steps', 0) > 0:
            metrics['samples_per_step'] = metrics['total_reward_calls'] / metrics['total_steps']
        
        return metrics
    
    def export_metrics(self, output_file: Path):
        """Export metrics to JSON file."""
        calculated = self.calculate_metrics()
        export_data = {
            'raw_metrics': self.metrics,
            'calculated_metrics': calculated,
            'timestamp': datetime.now().isoformat(),
            'log_file': str(self.log_file),
            'reward_log': str(self.reward_log),
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Metrics exported to {output_file}")
        return export_data


def compare_runs(run1_file: Path, run2_file: Path):
    """Compare efficiency metrics between two runs."""
    logger.info(f"Comparing runs:")
    logger.info(f"  Run 1: {run1_file}")
    logger.info(f"  Run 2: {run2_file}")
    
    with open(run1_file, 'r') as f:
        run1 = json.load(f)
    
    with open(run2_file, 'r') as f:
        run2 = json.load(f)
    
    m1 = run1.get('calculated_metrics', {})
    m2 = run2.get('calculated_metrics', {})
    
    print("\n" + "=" * 70)
    print("TRAINING EFFICIENCY COMPARISON")
    print("=" * 70)
    
    comparisons = [
        ('Average Step Time (s)', 'avg_step_time', 'lower'),
        ('Steps per Second', 'steps_per_second', 'higher'),
        ('Total Training Time (s)', 'total_training_time', 'lower'),
        ('Total Reward Calls', 'total_reward_calls', 'same'),
        ('Samples per Step', 'samples_per_step', 'same'),
    ]
    
    for label, key, better in comparisons:
        v1 = m1.get(key, 0)
        v2 = m2.get(key, 0)
        
        if v1 == 0 or v2 == 0:
            continue
        
        diff_pct = ((v2 - v1) / v1) * 100
        
        print(f"\n{label}:")
        print(f"  Run 1: {v1:.4f}")
        print(f"  Run 2: {v2:.4f}")
        print(f"  Difference: {diff_pct:+.2f}%")
        
        if better == 'lower':
            if v2 < v1:
                print(f"  ✓ Run 2 is {abs(diff_pct):.2f}% faster")
            else:
                print(f"  ✗ Run 2 is {abs(diff_pct):.2f}% slower")
        elif better == 'higher':
            if v2 > v1:
                print(f"  ✓ Run 2 is {abs(diff_pct):.2f}% faster")
            else:
                print(f"  ✗ Run 2 is {abs(diff_pct):.2f}% slower")
    
    print("\n" + "=" * 70)


def print_metrics(metrics: Dict, label: str = "Training Efficiency"):
    """Print efficiency metrics in a readable format."""
    print("\n" + "=" * 70)
    print(label.upper())
    print("=" * 70)
    
    if 'avg_step_time' in metrics:
        print(f"Average Step Time:     {metrics['avg_step_time']:.4f} seconds")
        print(f"Min Step Time:         {metrics.get('min_step_time', 0):.4f} seconds")
        print(f"Max Step Time:         {metrics.get('max_step_time', 0):.4f} seconds")
    
    if 'steps_per_second' in metrics:
        print(f"Steps per Second:      {metrics['steps_per_second']:.4f}")
    
    if 'total_training_time' in metrics:
        hours = metrics['total_training_time'] / 3600
        print(f"Total Training Time:   {hours:.2f} hours ({metrics['total_training_time']:.0f} seconds)")
    
    if 'total_reward_calls' in metrics:
        print(f"Total Reward Calls:    {metrics['total_reward_calls']:,}")
    
    if 'samples_per_step' in metrics:
        print(f"Samples per Step:      {metrics['samples_per_step']:.2f}")
    
    if 'total_steps' in metrics:
        print(f"Total Steps Tracked:   {metrics['total_steps']}")
    
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Monitor training efficiency metrics"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="training.log",
        help="Path to training log file"
    )
    parser.add_argument(
        "--reward-log",
        type=str,
        default="reward_samples.log",
        help="Path to reward samples log file"
    )
    parser.add_argument(
        "--export",
        type=str,
        help="Export metrics to JSON file"
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("RUN1", "RUN2"),
        help="Compare two exported metric files"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log metrics to wandb"
    )
    
    args = parser.parse_args()
    
    # Compare mode
    if args.compare:
        compare_runs(Path(args.compare[0]), Path(args.compare[1]))
        return
    
    # Monitor mode
    monitor = TrainingEfficiencyMonitor(
        Path(args.log_file),
        Path(args.reward_log) if args.reward_log else None
    )
    
    monitor.parse_training_log()
    metrics = monitor.calculate_metrics()
    
    print_metrics(metrics)
    
    # Export if requested
    if args.export:
        monitor.export_metrics(Path(args.export))
    
    # Log to wandb if requested
    if args.wandb and WANDB_AVAILABLE:
        if wandb.run is None:
            wandb.init(project="training-efficiency", job_type="monitor")
        
        wandb.log({
            "efficiency/avg_step_time": metrics.get('avg_step_time', 0),
            "efficiency/steps_per_second": metrics.get('steps_per_second', 0),
            "efficiency/total_training_time": metrics.get('total_training_time', 0),
            "efficiency/total_reward_calls": metrics.get('total_reward_calls', 0),
            "efficiency/samples_per_step": metrics.get('samples_per_step', 0),
        })
        logger.info("Metrics logged to wandb")


if __name__ == "__main__":
    main()

