"""
GSM8K Dataset Preparation Script

This script downloads and prepares the GSM8K dataset for GRPO training.
Data is formatted according to VERL requirements and saved in parquet format.
"""

import os
import sys
import logging
from pathlib import Path
from datasets import load_dataset

# Add parent directory to path to import prompts
sys.path.insert(0, str(Path(__file__).parent.parent))
from GRPO.prompts import get_gsm8k_prompt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def format_gsm8k_example(example: dict, idx: int, split: str, prompt_format: str = "grpo") -> dict:
    """
    Format a single GSM8K example for GRPO training.

    Args:
        example: Raw dataset example containing 'question' and 'answer'
        idx: Index of the example
        split: Dataset split ('train' or 'test')
        prompt_format: Format type for the prompt ('grpo', 'partial', or 'default')

    Returns:
        Formatted example with prompt and reward model configuration
    """
    # Get the system instruction based on format type
    system_instruction = get_gsm8k_prompt(prompt_format)
    
    # Create prompt with system message and user question (cleaner separation)
    prompt_messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": example['question']}
    ]

    return {
        "data_source": "gsm8k",
        "prompt": prompt_messages,
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": example["answer"]
        },
        "extra_info": {
            "split": split,
            "index": idx
        }
    }


def prepare_gsm8k_dataset(
    output_dir: str = "data/gsm8k",
    val_split_ratio: float = 0.1,
    prompt_format: str = "grpo"
) -> None:
    """
    Download and prepare GSM8K dataset for GRPO training.

    Creates train and validation splits from the training data,
    and keeps test set separate for final evaluation.

    Args:
        output_dir: Directory to save prepared dataset files
        val_split_ratio: Ratio of training data to use for validation (default: 0.1)
        prompt_format: Format type for prompts ('grpo', 'partial', or 'default')
    """
    logger.info("Starting GSM8K dataset preparation")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading GSM8K dataset from HuggingFace")
    dataset = load_dataset("openai/gsm8k", "main")

    logger.info(f"Splitting training data (val_ratio={val_split_ratio})")
    # Split training data into train and validation
    train_val_split = dataset["train"].train_test_split(
        test_size=val_split_ratio,
        seed=42
    )

    logger.info(f"Formatting training split with prompt format: {prompt_format}")
    train_dataset = train_val_split["train"].map(
        lambda x, i: format_gsm8k_example(x, i, "train", prompt_format),
        with_indices=True,
        desc="Formatting train examples"
    )

    logger.info("Formatting validation split")
    val_dataset = train_val_split["test"].map(
        lambda x, i: format_gsm8k_example(x, i, "val", prompt_format),
        with_indices=True,
        desc="Formatting val examples"
    )

    logger.info("Formatting test split")
    test_dataset = dataset["test"].map(
        lambda x, i: format_gsm8k_example(x, i, "test", prompt_format),
        with_indices=True,
        desc="Formatting test examples"
    )

    train_output = output_path / "train.parquet"
    val_output = output_path / "val.parquet"
    test_output = output_path / "test.parquet"

    logger.info(f"Saving training data to {train_output}")
    train_dataset.to_parquet(train_output)

    logger.info(f"Saving validation data to {val_output}")
    val_dataset.to_parquet(val_output)

    logger.info(f"Saving test data to {test_output}")
    test_dataset.to_parquet(test_output)

    logger.info(
        f"Dataset preparation complete. "
        f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    logger.info("Sample prompt:")
    logger.info(train_dataset[0]['prompt'][0]['content'])


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare GSM8K dataset with configurable prompt format")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/gsm8k",
        help="Directory to save prepared dataset files"
    )
    parser.add_argument(
        "--val-split-ratio",
        type=float,
        default=0.1,
        help="Ratio of training data to use for validation"
    )
    parser.add_argument(
        "--prompt-format",
        type=str,
        default="grpo",
        choices=["grpo", "partial", "default"],
        help="Format type for prompts (grpo=XML format, partial=#### format)"
    )
    
    args = parser.parse_args()
    
    logger.info(f"Preparing dataset with prompt format: {args.prompt_format}")
    logger.info(f"Prompt template:\n{get_gsm8k_prompt(args.prompt_format)}\n")
    
    prepare_gsm8k_dataset(
        output_dir=args.output_dir,
        val_split_ratio=args.val_split_ratio,
        prompt_format=args.prompt_format
    )
