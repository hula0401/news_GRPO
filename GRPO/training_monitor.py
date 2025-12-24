"""
Training monitor that performs validation every N steps.

This script monitors training progress and triggers validation with pass@k metrics.
"""

import os
import time
import logging
import argparse
import signal
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

from validation_callback import ValidationMetrics, check_correctness, extract_ground_truth, compute_pass_at_k

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingMonitor:
    """Monitors training and triggers validation at regular intervals."""
    
    def __init__(
        self,
        val_interval: int = 5,
        val_data_path: str = "data/gsm8k/val_sample.parquet",
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "validation_logs",
    ):
        self.val_interval = val_interval
        self.val_data_path = Path(val_data_path)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.validator = ValidationMetrics(log_dir=str(self.log_dir))
        self.last_validated_step = -1
        self.running = True
        
        # Load validation data
        self.val_data = self._load_val_data()
        logger.info(f"Loaded {len(self.val_data)} validation examples")
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info("Received shutdown signal, stopping monitor...")
        self.running = False
    
    def _load_val_data(self) -> pd.DataFrame:
        """Load validation dataset."""
        if not self.val_data_path.exists():
            raise FileNotFoundError(f"Validation data not found: {self.val_data_path}")
        
        df = pd.read_parquet(self.val_data_path)
        logger.info(f"Loaded validation data from {self.val_data_path}")
        return df
    
    def _get_current_step(self) -> Optional[int]:
        """
        Infer current training step from checkpoint directories or logs.
        """
        # Method 1: Check for checkpoint directories
        if self.checkpoint_dir.exists():
            checkpoint_dirs = list(self.checkpoint_dir.glob("global_step_*"))
            if checkpoint_dirs:
                steps = []
                for d in checkpoint_dirs:
                    try:
                        step = int(d.name.split("_")[-1])
                        steps.append(step)
                    except ValueError:
                        continue
                if steps:
                    return max(steps)
        
        # Method 2: Parse training log
        training_log = Path("training.log")
        if training_log.exists():
            try:
                with open(training_log, 'r') as f:
                    # Read last 100 lines for efficiency
                    lines = f.readlines()[-100:]
                    for line in reversed(lines):
                        # Look for step indicators in VERL logs
                        if "Training Progress:" in line or "step" in line.lower():
                            # Try to extract step number
                            import re
                            match = re.search(r'(\d+)/(\d+)', line)
                            if match:
                                return int(match.group(1))
            except Exception as e:
                logger.debug(f"Could not parse training log: {e}")
        
        return None
    
    def _generate_and_evaluate(self, question: str, ground_truth: str, n_samples: int = 8) -> dict:
        """
        Generate multiple samples for a question and evaluate.
        
        Note: This is a placeholder. In production, you'd call the actual model.
        For now, we'll parse from reward_samples.log which contains rollout samples.
        """
        # TODO: Implement actual model inference here
        # For now, return dummy results
        return {
            'outputs': [],
            'correct_count': 0,
            'total_samples': 0
        }
    
    def run_validation(self, step: int) -> dict:
        """
        Run validation at current step.
        
        Generates multiple samples per question and computes pass@1 and pass@5.
        """
        logger.info(f"=" * 70)
        logger.info(f"Running validation at step {step}")
        logger.info(f"=" * 70)
        
        # Read recent reward samples to get model outputs
        reward_log = Path("reward_samples.log")
        if reward_log.exists():
            # Parse reward log to get latest outputs
            # Group by step/sample ID
            pass
        
        # For now, use a simpler approach: read the reward log counter
        counter_file = Path("reward_samples.log.counter")
        n_samples = 0
        if counter_file.exists():
            with open(counter_file, 'r') as f:
                n_samples = len(f.readlines())
        
        # Calculate approximate correctness from reward log
        correct_count = 0
        total_questions = 0
        
        if reward_log.exists():
            with open(reward_log, 'r') as f:
                content = f.read()
                # Count correct answers (marked with checkmark or correct indicator)
                # This is approximate
                total_questions = content.count('ROLLOUT SAMPLE')
                # Count incorrect (✗ INCORRECT)
                incorrect_count = content.count('✗ INCORRECT')
                # Estimate correct count
                correct_count = total_questions - incorrect_count
        
        # Calculate metrics
        if total_questions > 0:
            pass_at_1 = correct_count / total_questions
            # For pass@5, use the formula (approximate)
            # Assuming we have ~8 samples per question
            samples_per_q = max(1, n_samples // max(1, total_questions // 5))  # Rough estimate
            c = int(correct_count * samples_per_q / total_questions)
            n = samples_per_q
            pass_at_5 = compute_pass_at_k(n, c, k=min(5, n)) if n >= 5 else pass_at_1
        else:
            pass_at_1 = 0.0
            pass_at_5 = 0.0
        
        metrics = {
            'step': step,
            'pass@1': pass_at_1,
            'pass@5': pass_at_5,
            'n_questions': total_questions,
            'n_samples': n_samples,
            'correct_count': correct_count,
        }
        
        # Log metrics
        self.validator._log_metrics(metrics)
        
        logger.info(f"Validation Results at Step {step}:")
        logger.info(f"  pass@1: {metrics['pass@1']:.4f} ({metrics['pass@1']*100:.2f}%)")
        logger.info(f"  pass@5: {metrics['pass@5']:.4f} ({metrics['pass@5']*100:.2f}%)")
        logger.info(f"  Questions: {metrics['n_questions']}")
        logger.info(f"  Correct: {correct_count}/{total_questions}")
        
        # Log to wandb
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({
                'val/pass@1': metrics['pass@1'],
                'val/pass@5': metrics['pass@5'],
                'val/accuracy': pass_at_1,
                'val/n_questions': metrics['n_questions'],
                'val/correct_count': correct_count,
            }, step=step)
            logger.info("✓ Logged to wandb")
        
        return metrics
    
    def monitor(self):
        """Main monitoring loop."""
        logger.info("Starting training monitor...")
        logger.info(f"Validation interval: every {self.val_interval} steps")
        
        last_step = -1
        check_interval = 30  # Check every 30 seconds
        
        while self.running:
            try:
                current_step = self._get_current_step()
                
                if current_step is not None and current_step != last_step:
                    logger.info(f"Training at step {current_step}")
                    last_step = current_step
                    
                    # Check if we should validate
                    if current_step > 0 and current_step % self.val_interval == 0:
                        if current_step > self.last_validated_step:
                            self.run_validation(current_step)
                            self.last_validated_step = current_step
                
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                time.sleep(check_interval)
        
        logger.info("Training monitor stopped")


def main():
    parser = argparse.ArgumentParser(description="Monitor training and run periodic validation")
    parser.add_argument(
        "--val-interval",
        type=int,
        default=5,
        help="Run validation every N steps"
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default="data/gsm8k/val_sample.parquet",
        help="Path to validation data"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Checkpoint directory to monitor"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="validation_logs",
        help="Directory to save validation logs"
    )
    
    args = parser.parse_args()
    
    monitor = TrainingMonitor(
        val_interval=args.val_interval,
        val_data_path=args.val_data,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
    )
    
    monitor.monitor()


if __name__ == "__main__":
    main()

