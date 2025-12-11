"""
VERL Reward Function Wrapper

This module provides a wrapper to integrate custom reward functions
with VERL's training pipeline. It converts between VERL's expected
format and our custom reward function interface.
"""

import sys
from pathlib import Path

# Add GRPO directory to Python path for imports
grpo_dir = Path(__file__).parent
if str(grpo_dir) not in sys.path:
    sys.path.insert(0, str(grpo_dir))

import logging
from typing import List, Dict, Any
import torch
from reward_function import get_reward_function

logger = logging.getLogger(__name__)


class VERLRewardWrapper:
    """
    Wrapper to integrate custom reward functions with VERL.

    VERL expects reward functions to accept batch data and return
    reward tensors. This wrapper handles the conversion.
    """

    def __init__(self, task_type: str = "gsm8k", **reward_kwargs):
        """
        Initialize VERL reward wrapper.

        Args:
            task_type: Type of task ('gsm8k' or 'news')
            **reward_kwargs: Arguments for reward function
        """
        self.task_type = task_type
        self.reward_fn = get_reward_function(task_type, **reward_kwargs)
        logger.info(f"Initialized VERL reward wrapper for task: {task_type}")

    def __call__(
        self,
        prompts: List[str],
        responses: List[str],
        metadata: List[Dict[str, Any]],
    ) -> torch.Tensor:
        """
        Calculate rewards for a batch of responses.

        Args:
            prompts: List of input prompts
            responses: List of model generated responses
            metadata: List of metadata dicts containing ground truth, etc.

        Returns:
            Tensor of reward scores, shape (batch_size,)
        """
        batch_size = len(responses)
        rewards = []

        for i in range(batch_size):
            try:
                if self.task_type == "gsm8k":
                    # Extract ground truth from metadata
                    ground_truth = metadata[i].get("reward_model", {}).get("ground_truth", "")
                    result = self.reward_fn.calculate_reward(
                        response=responses[i],
                        ground_truth=ground_truth
                    )
                elif self.task_type == "news":
                    # Extract prompt and reference from metadata
                    prompt = prompts[i]
                    reference = metadata[i].get("reference", None)
                    result = self.reward_fn.calculate_reward(
                        response=responses[i],
                        prompt=prompt,
                        reference=reference
                    )
                else:
                    raise ValueError(f"Unknown task type: {self.task_type}")

                rewards.append(result.score)

                # Log detailed results at debug level
                logger.debug(
                    f"Sample {i}: reward={result.score:.3f}, "
                    f"details={result.details}"
                )

            except Exception as e:
                logger.warning(f"Error calculating reward for sample {i}: {e}")
                rewards.append(0.0)

        # Convert to tensor
        reward_tensor = torch.tensor(rewards, dtype=torch.float32)

        # Log batch statistics
        logger.info(
            f"Batch rewards - mean: {reward_tensor.mean():.3f}, "
            f"std: {reward_tensor.std():.3f}, "
            f"min: {reward_tensor.min():.3f}, "
            f"max: {reward_tensor.max():.3f}"
        )

        return reward_tensor


def create_gsm8k_reward_fn(**kwargs):
    """
    Create GSM8K reward function for VERL.

    This returns a simple callable that VERL can serialize with Ray.

    Args:
        **kwargs: Arguments passed by VERL (may include extra params)

    Returns:
        Callable reward function
    """
    from reward_function import GSM8KRewardFunction

    # Only pass through kwargs that GSM8KRewardFunction actually accepts
    valid_keys = {
        'correct_answer_score', 'wrong_answer_score',
        'partial_credit', 'reasoning_weight'
    }
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}

    logger.info(f"Creating GSM8K reward function (filtered {len(kwargs) - len(filtered_kwargs)} VERL-specific kwargs)")
    if filtered_kwargs:
        logger.info(f"Using custom reward params: {filtered_kwargs}")

    # Create reward function instance
    reward_fn = GSM8KRewardFunction(**filtered_kwargs)

    # Return a simple function that VERL can call
    def compute_reward(prompts, responses, metadata):
        """Compute rewards for a batch of responses"""
        import torch
        rewards = []

        for i in range(len(responses)):
            try:
                ground_truth = metadata[i].get("reward_model", {}).get("ground_truth", "")
                result = reward_fn.calculate_reward(
                    response=responses[i],
                    ground_truth=ground_truth
                )
                rewards.append(result.score)
                logger.debug(f"Sample {i}: reward={result.score:.3f}, details={result.details}")
            except Exception as e:
                logger.warning(f"Error calculating reward for sample {i}: {e}")
                rewards.append(0.0)

        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        logger.info(
            f"Batch rewards - mean: {reward_tensor.mean():.3f}, "
            f"std: {reward_tensor.std():.3f}, "
            f"min: {reward_tensor.min():.3f}, "
            f"max: {reward_tensor.max():.3f}"
        )
        return reward_tensor

    return compute_reward


def create_news_reward_fn(**kwargs):
    """
    Create news reward function for VERL.

    Args:
        **kwargs: Arguments passed by VERL (may include extra params)

    Returns:
        VERLRewardWrapper instance
    """
    # Only pass through kwargs that NewsRewardFunction actually accepts
    valid_keys = {
        'coherence_weight', 'factuality_weight',
        'relevance_weight', 'style_weight'
    }
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}

    logger.info(f"Creating News reward function (filtered {len(kwargs) - len(filtered_kwargs)} VERL-specific kwargs)")
    if filtered_kwargs:
        logger.info(f"Using custom reward params: {filtered_kwargs}")

    return VERLRewardWrapper(task_type="news", **filtered_kwargs)


# Example usage for VERL integration
if __name__ == "__main__":
    # Test GSM8K reward function
    print("Testing GSM8K Reward Function")
    print("=" * 50)

    gsm8k_reward = create_gsm8k_reward_fn(
        partial_credit=True,
        reasoning_weight=0.2
    )

    # Example data
    prompts = [
        "Janet's ducks lay 16 eggs per day. She eats three for breakfast..."
    ]
    responses = [
        "Let me solve this step by step.\n"
        "First, eggs laid per day: 16\n"
        "Eggs eaten: 3\n"
        "Eggs used for muffins: 4\n"
        "Remaining: 16 - 3 - 4 = 9\n"
        "Price per egg: $2\n"
        "Total: 9 × 2 = 18\n"
        "#### 18"
    ]
    metadata = [
        {
            "reward_model": {
                "style": "rule",
                "ground_truth": "Janet sells 16 - 3 - 4 = 9 eggs\n#### 18"
            }
        }
    ]

    rewards = gsm8k_reward(prompts, responses, metadata)
    print(f"Reward: {rewards[0]:.3f}")
    print()

    # Test News reward function
    print("Testing News Reward Function")
    print("=" * 50)

    news_reward = create_news_reward_fn(
        coherence_weight=0.3,
        factuality_weight=0.4,
        relevance_weight=0.2,
        style_weight=0.1
    )

    prompts = [
        "Write a news article about recent tech developments"
    ]
    responses = [
        "Recent reports indicate significant advances in artificial intelligence. "
        "Tech companies announced breakthrough developments in machine learning. "
        "According to industry sources, these innovations will transform computing."
    ]
    metadata = [
        {
            "reference": "AI developments announced by tech companies"
        }
    ]

    rewards = news_reward(prompts, responses, metadata)
    print(f"Reward: {rewards[0]:.3f}")
