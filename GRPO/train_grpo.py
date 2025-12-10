"""
GRPO Training Script with VERL

This script trains a language model using Group Relative Policy Optimization (GRPO)
with the VERL framework on the GSM8K dataset.

The script is designed to run on GPU servers with proper CUDA support.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional
import subprocess

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GRPOTrainer:
    """GRPO trainer using VERL framework"""

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
        experiment_name: str = "qwen-grpo-gsm8k",
        checkpoint_dir: str = "checkpoints",
        data_dir: str = "data/gsm8k",
        n_gpus: int = 1,
        train_batch_size: int = 4,
        learning_rate: float = 1e-6,
    ):
        """
        Initialize GRPO trainer.

        Args:
            model_path: HuggingFace model path or local path
            experiment_name: Name for this training experiment
            checkpoint_dir: Directory to save model checkpoints
            data_dir: Directory containing prepared GSM8K data
            n_gpus: Number of GPUs to use for training
            train_batch_size: Training batch size
            learning_rate: Learning rate for actor model
        """
        self.model_path = model_path
        self.experiment_name = experiment_name
        self.checkpoint_dir = Path(checkpoint_dir) / experiment_name
        self.data_dir = Path(data_dir)
        self.n_gpus = n_gpus
        self.train_batch_size = train_batch_size
        self.learning_rate = learning_rate

        self._validate_setup()

    def _validate_setup(self) -> None:
        """Validate that required files and directories exist"""
        logger.info("Validating setup")

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}. "
                "Run prepare_gsm8k.py first to prepare the dataset."
            )

        train_file = self.data_dir / "train.parquet"
        test_file = self.data_dir / "test.parquet"

        if not train_file.exists():
            raise FileNotFoundError(f"Training data not found: {train_file}")
        if not test_file.exists():
            raise FileNotFoundError(f"Test data not found: {test_file}")

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Setup validation complete")

    def _set_environment_variables(self) -> None:
        """Set environment variables for optimal training performance"""
        logger.info("Setting environment variables")

        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

        logger.info("Environment variables configured")

    def _build_training_command(self) -> list[str]:
        """
        Build the VERL training command with all hyperparameters.

        Returns:
            List of command arguments for subprocess execution
        """
        train_file = str(self.data_dir / "train.parquet")
        val_file = str(self.data_dir / "test.parquet")
        checkpoint_path = str(self.checkpoint_dir)

        cmd = [
            sys.executable, "-m", "verl.trainer.main_ppo",

            # Algorithm configuration
            "algorithm.adv_estimator=grpo",
            "reward.type=custom",
            "reward.custom_class=verl.utils.reward_score.gsm8k_reward.GSM8KReward",

            # Data configuration
            f"data.train_files={train_file}",
            f"data.val_files={val_file}",
            f"data.train_batch_size={self.train_batch_size}",
            f"data.val_batch_size={self.train_batch_size}",
            "data.max_prompt_length=512",
            "data.max_response_length=512",
            "data.dataloader_num_workers=2",

            # Model configuration
            f"actor_rollout_ref.model.path={self.model_path}",
            "actor_rollout_ref.model.use_remove_padding=True",
            "actor_rollout_ref.model.enable_gradient_checkpointing=True",
            "++actor_rollout_ref.model.override_config.attn_implementation=sdpa",


            # Rollout configuration
            "actor_rollout_ref.rollout.name=vllm",
            "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
            "actor_rollout_ref.rollout.dtype=float16",
            "actor_rollout_ref.rollout.n=4",
            "actor_rollout_ref.rollout.temperature=0.8",
            "actor_rollout_ref.rollout.log_prob_micro_batch_size=4",

            # Actor training configuration
            f"actor_rollout_ref.actor.optim.lr={self.learning_rate}",
            "actor_rollout_ref.actor.ppo_mini_batch_size=4",
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4",

            # Hardware configuration
            f"trainer.n_gpus_per_node={self.n_gpus}",
            "trainer.nnodes=1",

            # Logging and checkpointing
            "trainer.logger=['console']",
            f"trainer.project_name={self.experiment_name}",
            f"trainer.experiment_name={self.experiment_name}",
            f"trainer.default_local_dir={checkpoint_path}",
            "++trainer.val_before_train=False",
        ]

        return cmd

    def train(self) -> None:
        """Execute GRPO training"""
        logger.info(f"Starting GRPO training for experiment: {self.experiment_name}")
        logger.info(f"Model: {self.model_path}")
        logger.info(f"Checkpoints will be saved to: {self.checkpoint_dir}")

        self._set_environment_variables()

        cmd = self._build_training_command()

        logger.info("Launching VERL trainer")
        logger.info(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                text=True,
                stdout=sys.stdout,
                stderr=sys.stderr
            )
            logger.info("Training completed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed with error: {e}")
            raise
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            raise


def main():
    """Main training entrypoint"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a language model with GRPO using VERL"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Model path (HuggingFace or local)"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="qwen-grpo-gsm8k",
        help="Experiment name for logging and checkpoints"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/gsm8k",
        help="Directory containing prepared GSM8K data"
    )
    parser.add_argument(
        "--n-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-6,
        help="Learning rate for actor model"
    )

    args = parser.parse_args()

    trainer = GRPOTrainer(
        model_path=args.model,
        experiment_name=args.experiment_name,
        checkpoint_dir=args.checkpoint_dir,
        data_dir=args.data_dir,
        n_gpus=args.n_gpus,
        train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    trainer.train()


if __name__ == "__main__":
    main()
