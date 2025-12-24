"""
Validation callback for computing pass@k metrics during GRPO training.

This module implements pass@1 and pass@5 metrics for GSM8K evaluation.
Pass@k measures the probability that at least one of k generated samples is correct.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def extract_answer(text: str) -> Optional[str]:
    """
    Extract numerical answer from model output.
    Looks for answer in <answer> tags or final numerical value.
    """
    # Try to extract from <answer> tags
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    if answer_match:
        answer_text = answer_match.group(1).strip()
        # Extract last number from answer
        numbers = re.findall(r'-?\d+\.?\d*', answer_text)
        if numbers:
            return numbers[-1]
    
    # Fallback: extract last number from entire text
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]
    
    return None


def extract_ground_truth(ground_truth_str: str) -> Optional[str]:
    """
    Extract the final answer from ground truth string.
    GSM8K format: "explanation #### answer"
    """
    # Look for #### marker
    if '####' in ground_truth_str:
        parts = ground_truth_str.split('####')
        if len(parts) >= 2:
            answer = parts[-1].strip()
            # Extract number
            numbers = re.findall(r'-?\d+\.?\d*', answer)
            if numbers:
                return numbers[0]
    
    # Fallback: extract last number
    numbers = re.findall(r'-?\d+\.?\d*', ground_truth_str)
    if numbers:
        return numbers[-1]
    
    return None


def check_correctness(model_answer: str, ground_truth: str) -> bool:
    """Check if model answer matches ground truth."""
    model_ans = extract_answer(model_answer)
    gt_ans = extract_ground_truth(ground_truth)
    
    if model_ans is None or gt_ans is None:
        return False
    
    try:
        # Compare as floats to handle different formats (3.0 vs 3)
        return abs(float(model_ans) - float(gt_ans)) < 1e-6
    except (ValueError, TypeError):
        # Compare as strings if not numbers
        return model_ans.strip() == gt_ans.strip()


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculate pass@k metric.
    
    Args:
        n: Total number of samples generated
        c: Number of correct samples
        k: Number of samples to consider (k <= n)
    
    Returns:
        pass@k: Probability that at least one of k samples is correct
    
    Formula: pass@k = 1 - C(n-c, k) / C(n, k)
    Where C(n, k) is the binomial coefficient "n choose k"
    """
    if n < k:
        return 0.0
    if c >= k:
        return 1.0
    if c == 0:
        return 0.0
    
    # Calculate using binomial coefficients
    # pass@k = 1 - (n-c choose k) / (n choose k)
    from math import comb
    
    try:
        pass_at_k = 1.0 - comb(n - c, k) / comb(n, k)
    except (ValueError, ZeroDivisionError):
        pass_at_k = 0.0
    
    return pass_at_k


class ValidationMetrics:
    """Tracks validation metrics across steps."""
    
    def __init__(self, log_dir: str = "validation_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.log_dir / "validation_metrics.jsonl"
        
    def compute_metrics(
        self,
        val_data: List[Dict[str, Any]],
        step: int
    ) -> Dict[str, float]:
        """
        Compute pass@1 and pass@5 metrics for validation data.
        
        Args:
            val_data: List of validation examples with 'outputs' and 'ground_truth'
            step: Current training step
        
        Returns:
            Dictionary with pass@1 and pass@5 metrics
        """
        # Group by question (for pass@k we need multiple samples per question)
        question_results = defaultdict(list)
        
        for item in val_data:
            question_id = item.get('question_id', item.get('index', 0))
            ground_truth = item['ground_truth']
            
            # Check each output
            for output in item.get('outputs', [item.get('output', '')]):
                is_correct = check_correctness(output, ground_truth)
                question_results[question_id].append(is_correct)
        
        # Calculate pass@1 and pass@5
        pass_1_scores = []
        pass_5_scores = []
        
        for question_id, results in question_results.items():
            n = len(results)  # Number of samples generated
            c = sum(results)  # Number of correct samples
            
            # pass@1: Use the first sample only
            if n >= 1:
                pass_1_scores.append(1.0 if results[0] else 0.0)
            
            # pass@5: Probability that at least one of 5 samples is correct
            if n >= 5:
                pass_5 = compute_pass_at_k(n, c, k=5)
                pass_5_scores.append(pass_5)
        
        metrics = {
            'step': step,
            'pass@1': np.mean(pass_1_scores) if pass_1_scores else 0.0,
            'pass@5': np.mean(pass_5_scores) if pass_5_scores else 0.0,
            'n_questions': len(question_results),
            'n_samples_total': sum(len(r) for r in question_results.values()),
        }
        
        # Log to file
        self._log_metrics(metrics)
        
        logger.info(f"Step {step} Validation Metrics:")
        logger.info(f"  pass@1: {metrics['pass@1']:.4f}")
        logger.info(f"  pass@5: {metrics['pass@5']:.4f}")
        logger.info(f"  Questions: {metrics['n_questions']}")
        
        return metrics
    
    def _log_metrics(self, metrics: Dict[str, Any]):
        """Append metrics to log file."""
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
    
    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """Load all metrics from log file."""
        if not self.metrics_file.exists():
            return []
        
        metrics = []
        with open(self.metrics_file, 'r') as f:
            for line in f:
                metrics.append(json.loads(line))
        
        return metrics


def parse_reward_log(log_file: Path = Path("reward_samples.log")) -> List[Dict[str, Any]]:
    """
    Parse reward_samples.log to extract validation data.
    
    Returns:
        List of validation examples with model outputs and ground truths
    """
    if not log_file.exists():
        logger.warning(f"Reward log file not found: {log_file}")
        return []
    
    examples = []
    current_example = {}
    
    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            # New sample
            if line.startswith('📊 ROLLOUT SAMPLE'):
                if current_example:
                    examples.append(current_example)
                current_example = {}
            
            # Ground truth
            elif line.startswith('✓ GROUND TRUTH:'):
                current_example['ground_truth'] = line.replace('✓ GROUND TRUTH:', '').strip()
            
            # Model output
            elif line.startswith('📝 FULL MODEL OUTPUT:'):
                # Read next lines until separator
                output_lines = []
                continue
            
            # Check if we're in output section
            elif 'ground_truth' in current_example and '====' not in line and line:
                if 'output' not in current_example:
                    current_example['output'] = line
                else:
                    current_example['output'] += '\n' + line
    
    if current_example:
        examples.append(current_example)
    
    return examples


def run_validation_step(step: int, log_dir: str = "validation_logs") -> Dict[str, float]:
    """
    Run validation at a specific training step.
    
    Args:
        step: Current training step
        log_dir: Directory to save validation logs
    
    Returns:
        Dictionary with validation metrics
    """
    logger.info(f"Running validation at step {step}")
    
    # Parse reward log to get recent samples
    val_data = parse_reward_log()
    
    if not val_data:
        logger.warning("No validation data found in reward log")
        return {'step': step, 'pass@1': 0.0, 'pass@5': 0.0}
    
    # Compute metrics
    validator = ValidationMetrics(log_dir=log_dir)
    metrics = validator.compute_metrics(val_data, step=step)
    
    # Log to wandb if available
    try:
        import wandb
        if wandb.run is not None:
            wandb.log({
                'val/pass@1': metrics['pass@1'],
                'val/pass@5': metrics['pass@5'],
                'val/n_questions': metrics['n_questions'],
                'val/step': step,
            }, step=step)
            logger.info("Logged validation metrics to wandb")
    except ImportError:
        logger.debug("wandb not available")
    except Exception as e:
        logger.warning(f"Could not log to wandb: {e}")
    
    return metrics


if __name__ == "__main__":
    # Test validation
    logging.basicConfig(level=logging.INFO)
    metrics = run_validation_step(step=0)
    print(f"\nValidation Results:")
    print(f"  pass@1: {metrics['pass@1']:.2%}")
    print(f"  pass@5: {metrics['pass@5']:.2%}")

