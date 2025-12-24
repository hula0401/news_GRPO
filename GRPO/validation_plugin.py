"""
Validation plugin for GRPO training.

This plugin runs in a separate thread/process and monitors training progress,
triggering validation at configured intervals.
"""

import os
import time
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, Any
import signal

from compute_passatk import analyze_log, print_metrics

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)


class ValidationPlugin:
    """
    Validation plugin that monitors training and runs validation periodically.
    
    This runs in a background thread and doesn't interfere with training.
    """
    
    def __init__(
        self,
        val_interval: int = 5,
        reward_log: str = "reward_samples.log",
        training_log: str = "training.log",
        val_log_dir: str = "validation_logs",
        enable_wandb: bool = True,
        check_interval: int = 30,
    ):
        """
        Initialize validation plugin.
        
        Args:
            val_interval: Run validation every N steps
            reward_log: Path to reward samples log
            training_log: Path to training log  
            val_log_dir: Directory to save validation logs
            enable_wandb: Whether to log to wandb
            check_interval: How often to check for new steps (seconds)
        """
        self.val_interval = val_interval
        self.reward_log = Path(reward_log)
        self.training_log = Path(training_log)
        self.val_log_dir = Path(val_log_dir)
        self.enable_wandb = enable_wandb and WANDB_AVAILABLE
        self.check_interval = check_interval
        
        self.val_log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.val_log_dir / "validation_history.jsonl"
        
        self.last_validated_step = -1
        self.running = False
        self.thread = None
        
        logger.info(f"Validation plugin initialized:")
        logger.info(f"  - Interval: every {val_interval} steps")
        logger.info(f"  - Wandb: {'enabled' if self.enable_wandb else 'disabled'}")
        logger.info(f"  - Check interval: {check_interval}s")
    
    def _get_current_step(self) -> Optional[int]:
        """
        Extract current training step from logs.
        
        Looks for patterns like "Training Progress: 15/25" in training log.
        """
        if not self.training_log.exists():
            return None
        
        try:
            # Read last 100 lines for efficiency
            with open(self.training_log, 'r') as f:
                lines = f.readlines()
                
                # Search from end to find most recent step
                for line in reversed(lines[-100:]):
                    # Look for "Training Progress: XX/YY"
                    import re
                    match = re.search(r'Training Progress:.*?(\d+)/(\d+)', line)
                    if match:
                        current = int(match.group(1))
                        return current
                    
                    # Alternative pattern: look for step numbers
                    match = re.search(r'step[:\s]+(\d+)', line, re.IGNORECASE)
                    if match:
                        return int(match.group(1))
        
        except Exception as e:
            logger.debug(f"Could not parse training log: {e}")
        
        return None
    
    def _run_validation(self, step: int) -> Dict[str, Any]:
        """
        Run validation at current step.
        
        Args:
            step: Current training step
            
        Returns:
            Dictionary with validation metrics
        """
        logger.info("=" * 70)
        logger.info(f"VALIDATION AT STEP {step}")
        logger.info("=" * 70)
        
        # Analyze reward log
        if not self.reward_log.exists():
            logger.warning(f"Reward log not found: {self.reward_log}")
            return {'step': step, 'pass@1': 0.0, 'pass@5': 0.0}
        
        metrics = analyze_log(self.reward_log)
        metrics['step'] = step
        
        # Print metrics
        print_metrics(metrics)
        
        # Save to file
        self._save_metrics(metrics)
        
        # Log to wandb
        if self.enable_wandb:
            try:
                if wandb.run is not None:
                    wandb.log({
                        'val/pass@1': metrics['pass@1'],
                        'val/pass@5': metrics['pass@5'],
                        'val/accuracy': metrics['accuracy'],
                        'val/n_questions': metrics['n_questions'],
                        'val/n_samples': metrics['n_samples'],
                        'val/correct_samples': metrics['correct_samples'],
                    }, step=step)
                    logger.info("✓ Logged to wandb")
                else:
                    logger.debug("wandb.run is None, skipping wandb logging")
            except Exception as e:
                logger.warning(f"Could not log to wandb: {e}")
        
        return metrics
    
    def _save_metrics(self, metrics: Dict[str, Any]):
        """Save metrics to JSONL file."""
        import json
        
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
    
    def _monitoring_loop(self):
        """Main monitoring loop that runs in background thread."""
        logger.info("Validation monitoring started")
        
        last_logged_step = -1
        
        while self.running:
            try:
                # Get current training step
                current_step = self._get_current_step()
                
                if current_step is not None:
                    # Log progress occasionally (not too verbose)
                    if current_step != last_logged_step and current_step % 5 == 0:
                        next_val = ((current_step // self.val_interval) + 1) * self.val_interval
                        logger.info(f"Training at step {current_step} (next validation at step {next_val})")
                        last_logged_step = current_step
                    
                    # Check if we should validate
                    if current_step > 0 and current_step % self.val_interval == 0:
                        if current_step > self.last_validated_step:
                            self._run_validation(current_step)
                            self.last_validated_step = current_step
                
                # Sleep before next check
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in validation monitoring loop: {e}", exc_info=True)
                time.sleep(self.check_interval)
        
        logger.info("Validation monitoring stopped")
    
    def start(self):
        """Start validation monitoring in background thread."""
        if self.running:
            logger.warning("Validation plugin already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.thread.start()
        
        logger.info("✓ Validation plugin started (running in background)")
    
    def stop(self):
        """Stop validation monitoring."""
        if not self.running:
            return
        
        logger.info("Stopping validation plugin...")
        self.running = False
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        
        logger.info("✓ Validation plugin stopped")
    
    def get_metrics_history(self):
        """Load validation metrics history from file."""
        import json
        
        if not self.metrics_file.exists():
            return []
        
        metrics = []
        with open(self.metrics_file, 'r') as f:
            for line in f:
                try:
                    metrics.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        
        return metrics


def create_validation_plugin(config: Dict[str, Any]) -> Optional[ValidationPlugin]:
    """
    Create validation plugin from configuration.
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        ValidationPlugin instance or None if disabled
    """
    trainer_cfg = config.get('trainer', {})
    
    # Check if validation is enabled
    validation_cfg = trainer_cfg.get('validation', {})
    enabled = validation_cfg.get('enabled', True)
    
    if not enabled:
        logger.info("Validation plugin disabled in config")
        return None
    
    # Get configuration
    val_interval = validation_cfg.get('interval', 5)
    reward_log = validation_cfg.get('reward_log', 'reward_samples.log')
    training_log = validation_cfg.get('training_log', 'training.log')
    val_log_dir = validation_cfg.get('log_dir', 'validation_logs')
    check_interval = validation_cfg.get('check_interval', 30)
    
    # Check if wandb is enabled
    enable_wandb = False
    if 'wandb' in trainer_cfg.get('logger', []):
        wandb_cfg = trainer_cfg.get('wandb', {})
        enable_wandb = wandb_cfg.get('enabled', False)
    
    # Create plugin
    plugin = ValidationPlugin(
        val_interval=val_interval,
        reward_log=reward_log,
        training_log=training_log,
        val_log_dir=val_log_dir,
        enable_wandb=enable_wandb,
        check_interval=check_interval,
    )
    
    return plugin

