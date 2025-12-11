"""
GSM8K Dataset Preparation Script

This script downloads and prepares the GSM8K dataset for GRPO training.
Data is formatted according to VERL requirements and saved in parquet format.
"""

import os
import logging
from pathlib import Path
from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def format_gsm8k_example(example: dict, idx: int, split: str) -> dict:
    """
    Format a single GSM8K example for GRPO training.

    Args:
        example: Raw dataset example containing 'question' and 'answer'
        idx: Index of the example
        split: Dataset split ('train' or 'test')

    Returns:
        Formatted example with prompt and reward model configuration
    """
    instruction = (
        f"{example['question']}\n"
        "Answer the above math problem. "
        "Think step by step. Output the final answer after ####."
    )

    return {
        "data_source": "gsm8k",
        "prompt": [{"role": "user", "content": instruction}],
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


def prepare_gsm8k_dataset(output_dir: str = "data/gsm8k") -> None:
    """
    Download and prepare GSM8K dataset for GRPO training.

    Args:
        output_dir: Directory to save prepared dataset files
    """
    logger.info("Starting GSM8K dataset preparation")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading GSM8K dataset from HuggingFace")
    dataset = load_dataset("openai/gsm8k", "main")

    logger.info("Formatting training split")
    train_dataset = dataset["train"].map(
        lambda x, i: format_gsm8k_example(x, i, "train"),
        with_indices=True,
        desc="Formatting train examples"
    )

    logger.info("Formatting test split")
    test_dataset = dataset["test"].map(
        lambda x, i: format_gsm8k_example(x, i, "test"),
        with_indices=True,
        desc="Formatting test examples"
    )

    train_output = output_path / "train.parquet"
    test_output = output_path / "test.parquet"

    logger.info(f"Saving training data to {train_output}")
    train_dataset.to_parquet(train_output)

    logger.info(f"Saving test data to {test_output}")
    test_dataset.to_parquet(test_output)

    logger.info(
        f"Dataset preparation complete. "
        f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}"
    )

    logger.info("Sample prompt:")
    logger.info(train_dataset[0]['prompt'][0]['content'])


if __name__ == "__main__":
    prepare_gsm8k_dataset()
