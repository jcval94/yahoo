"""Reinforcement learning agent utilities"""
import logging
from typing import Dict


def train_rl_agent(env: str, config: Dict) -> str:
    """Placeholder RL agent trainer."""
    logger = logging.getLogger(__name__)
    logger.info("Training RL agent on %s", env)
    model_path = "rl_agent.zip"
    logger.info("RL agent saved to %s", model_path)
    return model_path
