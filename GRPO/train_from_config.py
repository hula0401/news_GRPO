"""
GRPO Training Script with YAML Configuration

This script trains a language model using GRPO with settings from a YAML config file.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import subprocess
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_environment_variables(config: dict) -> None:
    """Set environment variables from config"""
    env_config = config.get('environment', {})

    # PyTorch and tokenizer settings
    os.environ["PYTORCH_ALLOC_CONF"] = env_config.get('pytorch_alloc_conf', 'expandable_segments:True')
    os.environ["TOKENIZERS_PARALLELISM"] = env_config.get('tokenizers_parallelism', 'true')

    logger.info("Environment variables configured")


def build_training_command(config: dict) -> list[str]:
    """Build VERL training command from config"""

    model_cfg = config['model']
    data_cfg = config['data']
    rollout_cfg = config['rollout']
    actor_cfg = config['actor']
    algorithm_cfg = config['algorithm']
    reward_cfg = config['reward']
    trainer_cfg = config['trainer']

    cmd = [
        sys.executable, "-m", "verl.trainer.main_ppo",

        # Algorithm configuration
        f"algorithm.adv_estimator={algorithm_cfg['adv_estimator']}",
        f"custom_reward_function.path={reward_cfg['path']}",
        f"custom_reward_function.name={reward_cfg['function_name']}",

        # Data configuration
        f"data.train_files={data_cfg['train_files']}",
        f"data.val_files={data_cfg['val_files']}",
        f"data.train_batch_size={data_cfg['train_batch_size']}",
        f"data.val_batch_size={data_cfg['val_batch_size']}",
        f"data.max_prompt_length={data_cfg['max_prompt_length']}",
        f"data.max_response_length={data_cfg['max_response_length']}",
        f"data.dataloader_num_workers={data_cfg['dataloader_num_workers']}",

        # Model configuration
        f"actor_rollout_ref.model.path={model_cfg['path']}",
        f"actor_rollout_ref.model.use_remove_padding={str(model_cfg['use_remove_padding']).lower()}",
        f"actor_rollout_ref.model.enable_gradient_checkpointing={str(model_cfg['enable_gradient_checkpointing']).lower()}",
        f"++actor_rollout_ref.model.override_config.attn_implementation={model_cfg['attn_implementation']}",

        # Rollout configuration
        f"actor_rollout_ref.rollout.name={rollout_cfg['name']}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={rollout_cfg['tensor_model_parallel_size']}",
        f"actor_rollout_ref.rollout.dtype={rollout_cfg['dtype']}",
        f"actor_rollout_ref.rollout.n={rollout_cfg['n']}",
        f"actor_rollout_ref.rollout.temperature={rollout_cfg['temperature']}",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size={rollout_cfg['log_prob_micro_batch_size']}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={rollout_cfg['gpu_memory_utilization']}",

        # Actor training configuration
        f"actor_rollout_ref.actor.optim.lr={actor_cfg['learning_rate']}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={actor_cfg['ppo_mini_batch_size']}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={actor_cfg['ppo_micro_batch_size_per_gpu']}",
        f"actor_rollout_ref.actor.ppo_epochs={actor_cfg['ppo_epochs']}",

        # Hardware configuration
        f"trainer.n_gpus_per_node={trainer_cfg['n_gpus_per_node']}",
        f"trainer.nnodes={trainer_cfg['nnodes']}",

        # Logging and checkpointing
        f"trainer.logger={trainer_cfg['logger']}",
        f"trainer.project_name={trainer_cfg['project_name']}",
        f"trainer.experiment_name={trainer_cfg['experiment_name']}",
        f"trainer.default_local_dir={trainer_cfg['default_local_dir']}",
        f"++trainer.val_before_train={str(trainer_cfg['val_before_train']).lower()}",
        f"trainer.save_freq={trainer_cfg['save_freq']}",
        f"trainer.total_epochs={trainer_cfg['total_epochs']}",

        # Optimization settings
        f"++trainer.mixed_precision={trainer_cfg['mixed_precision']}",
        f"++trainer.max_grad_norm={trainer_cfg['max_grad_norm']}",
    ]

    return cmd


def validate_config(config: dict) -> None:
    """Validate configuration and check for required files"""
    logger.info("Validating configuration")

    # Check data files exist
    data_cfg = config['data']
    train_file = Path(data_cfg['train_files'])
    val_file = Path(data_cfg['val_files'])

    if not train_file.exists():
        raise FileNotFoundError(f"Training data not found: {train_file}")
    if not val_file.exists():
        raise FileNotFoundError(f"Validation data not found: {val_file}")

    # Check reward function file exists
    reward_file = Path(config['reward']['path'])
    if not reward_file.exists():
        raise FileNotFoundError(f"Reward function not found: {reward_file}")

    # Create checkpoint directory
    checkpoint_dir = Path(config['trainer']['default_local_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Configuration validation complete")


def main():
    parser = argparse.ArgumentParser(
        description="Train a language model with GRPO using configuration file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="GRPO/config.yaml",
        help="Path to training configuration YAML file"
    )

    args = parser.parse_args()

    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    # Validate configuration
    validate_config(config)

    # Set environment variables
    set_environment_variables(config)

    # Build training command
    cmd = build_training_command(config)

    # Log training information
    logger.info("=" * 70)
    logger.info("Starting GRPO Training")
    logger.info("=" * 70)
    logger.info(f"Model: {config['model']['path']}")
    logger.info(f"Experiment: {config['trainer']['experiment_name']}")
    logger.info(f"Checkpoints: {config['trainer']['default_local_dir']}")
    logger.info(f"GPU Configuration:")
    logger.info(f"  - GPUs: {config['trainer']['n_gpus_per_node']}")
    logger.info(f"  - Memory utilization: {config['rollout']['gpu_memory_utilization']}")
    logger.info(f"  - Precision: {config['trainer']['mixed_precision']}")
    logger.info(f"Training Configuration:")
    logger.info(f"  - Batch size: {config['data']['train_batch_size']}")
    logger.info(f"  - Rollout samples: {config['rollout']['n']}")
    logger.info(f"  - PPO mini-batch: {config['actor']['ppo_mini_batch_size']}")
    logger.info(f"  - Learning rate: {config['actor']['learning_rate']}")
    logger.info(f"  - Total epochs: {config['trainer']['total_epochs']}")
    logger.info("=" * 70)

    # Execute training
    try:
        subprocess.run(
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


if __name__ == "__main__":
    main()
