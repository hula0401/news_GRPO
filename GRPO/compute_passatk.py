"""
Compute pass@k metrics from reward_samples.log.

This script analyzes the reward samples log to compute pass@1 and pass@5 metrics.
"""

import re
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import argparse

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


def extract_answer_from_text(text: str) -> Optional[str]:
    """Extract numerical answer from model output."""
    # Try <answer> tags first
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    if answer_match:
        answer_text = answer_match.group(1).strip()
        numbers = re.findall(r'-?\d+\.?\d*', answer_text)
        if numbers:
            return numbers[-1]
    
    # Fallback: last number in text
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]
    
    return None


def extract_ground_truth_answer(gt_str: str) -> Optional[str]:
    """Extract answer from ground truth (GSM8K format with #### marker)."""
    if '####' in gt_str:
        parts = gt_str.split('####')
        if len(parts) >= 2:
            answer = parts[-1].strip()
            numbers = re.findall(r'-?\d+\.?\d*', answer)
            if numbers:
                return numbers[0]
    
    numbers = re.findall(r'-?\d+\.?\d*', gt_str)
    if numbers:
        return numbers[-1]
    
    return None


def check_answer_correctness(model_output: str, ground_truth: str) -> bool:
    """Check if model answer matches ground truth."""
    model_ans = extract_answer_from_text(model_output)
    gt_ans = extract_ground_truth_answer(ground_truth)
    
    if model_ans is None or gt_ans is None:
        return False
    
    try:
        return abs(float(model_ans) - float(gt_ans)) < 1e-6
    except (ValueError, TypeError):
        return model_ans.strip() == gt_ans.strip()


def parse_reward_log(log_file: Path) -> List[Dict]:
    """
    Parse reward_samples.log to extract samples.
    
    Returns list of samples with:
    - sample_id: ROLLOUT SAMPLE number
    - call_number: Call number
    - ground_truth: Ground truth answer
    - model_output: Model's full output
    - model_answer: Extracted model answer
    - correct: Whether answer is correct
    - reward: Total reward
    """
    if not log_file.exists():
        logger.error(f"Log file not found: {log_file}")
        return []
    
    samples = []
    current_sample = {}
    in_output_section = False
    output_lines = []
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.rstrip()
            
            # Start of new sample
            if '📊 ROLLOUT SAMPLE' in line:
                # Save previous sample
                if current_sample and 'ground_truth' in current_sample:
                    if output_lines:
                        current_sample['model_output'] = '\n'.join(output_lines).strip()
                    samples.append(current_sample)
                
                # Parse sample ID and call number
                # Format: "📊 ROLLOUT SAMPLE #X (Call #Y)"
                match = re.search(r'SAMPLE #(\d+).*Call #(\d+)', line)
                if match:
                    sample_id = int(match.group(1))
                    call_num = int(match.group(2))
                else:
                    sample_id = len(samples) + 1
                    call_num = 0
                
                current_sample = {
                    'sample_id': sample_id,
                    'call_number': call_num,
                }
                in_output_section = False
                output_lines = []
            
            # Ground truth
            elif line.startswith('✓ GROUND TRUTH:'):
                current_sample['ground_truth'] = line.replace('✓ GROUND TRUTH:', '').strip()
            
            # Model answer line
            elif line.startswith('🤖 MODEL ANSWER:'):
                answer_part = line.replace('🤖 MODEL ANSWER:', '').strip()
                # Remove correctness marker
                answer_part = re.sub(r'✗ INCORRECT|✓ CORRECT', '', answer_part).strip()
                current_sample['model_answer'] = answer_part
                current_sample['correct'] = '✓ CORRECT' in line or '✗ INCORRECT' not in line
            
            # Reward line
            elif 'TOTAL REWARD:' in line:
                match = re.search(r'TOTAL REWARD:\s*([\d.]+)', line)
                if match:
                    current_sample['reward'] = float(match.group(1))
            
            # Full model output section
            elif line.startswith('📝 FULL MODEL OUTPUT:'):
                in_output_section = True
                output_lines = []
            
            # Separator - end of output section
            elif '=====' in line:
                if in_output_section and output_lines:
                    current_sample['model_output'] = '\n'.join(output_lines).strip()
                in_output_section = False
            
            # Collect output lines
            elif in_output_section and line and not line.startswith('─'):
                output_lines.append(line)
    
    # Don't forget the last sample
    if current_sample and 'ground_truth' in current_sample:
        if output_lines:
            current_sample['model_output'] = '\n'.join(output_lines).strip()
        samples.append(current_sample)
    
    return samples


def compute_pass_at_k_from_samples(samples: List[Dict], k: int) -> float:
    """
    Compute pass@k from samples grouped by question.
    
    Assumes samples with the same sample_id are for the same question.
    """
    from math import comb
    
    # Group by sample_id (question)
    questions = defaultdict(list)
    for sample in samples:
        # Verify correctness by comparing answers
        if 'model_output' in sample and 'ground_truth' in sample:
            is_correct = check_answer_correctness(
                sample.get('model_output', ''),
                sample['ground_truth']
            )
            questions[sample['sample_id']].append(is_correct)
    
    # Calculate pass@k for each question
    pass_k_values = []
    
    for question_id, results in questions.items():
        n = len(results)  # Total samples
        c = sum(results)  # Correct samples
        
        if n < k:
            # Not enough samples, use what we have
            pass_k = 1.0 if c > 0 else 0.0
        elif c >= k:
            pass_k = 1.0
        elif c == 0:
            pass_k = 0.0
        else:
            # pass@k = 1 - C(n-c, k) / C(n, k)
            try:
                pass_k = 1.0 - comb(n - c, k) / comb(n, k)
            except (ValueError, ZeroDivisionError):
                pass_k = 1.0 if c > 0 else 0.0
        
        pass_k_values.append(pass_k)
    
    return sum(pass_k_values) / len(pass_k_values) if pass_k_values else 0.0


def analyze_log(log_file: Path = Path("reward_samples.log")) -> Dict:
    """Analyze reward log and compute metrics."""
    logger.info(f"Parsing {log_file}...")
    samples = parse_reward_log(log_file)
    
    if not samples:
        logger.error("No samples found in log file")
        return {
            'n_samples': 0,
            'pass@1': 0.0,
            'pass@5': 0.0,
            'accuracy': 0.0,
        }
    
    logger.info(f"Found {len(samples)} samples")
    
    # Group by question
    questions = defaultdict(list)
    for sample in samples:
        if 'model_output' in sample and 'ground_truth' in sample:
            is_correct = check_answer_correctness(
                sample.get('model_output', ''),
                sample['ground_truth']
            )
            questions[sample['sample_id']].append(is_correct)
    
    # Compute metrics
    n_questions = len(questions)
    n_samples = len(samples)
    
    # Overall accuracy (treat each sample independently)
    correct_samples = sum(1 for s in samples 
                         if 'model_output' in s and 'ground_truth' in s 
                         and check_answer_correctness(s.get('model_output', ''), s['ground_truth']))
    accuracy = correct_samples / n_samples if n_samples > 0 else 0.0
    
    # pass@1: First sample per question
    pass_1_scores = []
    for q_id, results in questions.items():
        if results:
            pass_1_scores.append(1.0 if results[0] else 0.0)
    pass_at_1 = sum(pass_1_scores) / len(pass_1_scores) if pass_1_scores else 0.0
    
    # pass@5: Compute using formula
    pass_at_5 = compute_pass_at_k_from_samples(samples, k=5)
    
    # Samples per question distribution
    samples_per_q = [len(results) for results in questions.values()]
    avg_samples_per_q = sum(samples_per_q) / len(samples_per_q) if samples_per_q else 0
    
    metrics = {
        'n_samples': n_samples,
        'n_questions': n_questions,
        'accuracy': accuracy,
        'pass@1': pass_at_1,
        'pass@5': pass_at_5,
        'correct_samples': correct_samples,
        'avg_samples_per_question': avg_samples_per_q,
    }
    
    return metrics


def print_metrics(metrics: Dict):
    """Print metrics in a nice format (pass@1 and pass@5 only)."""
    print()
    print("=" * 70)
    print("VALIDATION METRICS")
    print("=" * 70)
    print(f"pass@1:      {metrics['pass@1']:.4f} ({metrics['pass@1']*100:.2f}%)")
    print(f"pass@5:      {metrics['pass@5']:.4f} ({metrics['pass@5']*100:.2f}%)")
    print()
    print(f"Questions:   {metrics['n_questions']}")
    print(f"Samples:     {metrics['n_samples']}")
    print("=" * 70)
    print()


def main():
    parser = argparse.ArgumentParser(description="Compute pass@k metrics from reward log")
    parser.add_argument(
        "--log-file",
        type=str,
        default="reward_samples.log",
        help="Path to reward samples log file"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log metrics to wandb"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Training step number (for wandb logging)"
    )
    
    args = parser.parse_args()
    
    # Analyze log
    metrics = analyze_log(Path(args.log_file))
    
    # Print results
    print_metrics(metrics)
    
    # Log to wandb
    if args.wandb and WANDB_AVAILABLE:
        if wandb.run is not None:
            step = args.step if args.step is not None else 0
            wandb.log({
                'val/pass@1': metrics['pass@1'],
                'val/pass@5': metrics['pass@5'],
                'val/accuracy': metrics['accuracy'],
                'val/n_questions': metrics['n_questions'],
                'val/n_samples': metrics['n_samples'],
            }, step=step)
            logger.info(f"Logged to wandb at step {step}")
        else:
            logger.warning("wandb.run is None. Initialize wandb first with wandb.init()")


if __name__ == "__main__":
    main()

