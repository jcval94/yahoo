"""Reinforcement Learning agent utilities."""
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def train_rl_agent(env, **kwargs) -> Any:
    start = time.perf_counter()
    logger.info("Training RL agent")
    try:
        # Placeholder for RL training
        agent = None
    except Exception:
        logger.exception("Error while training RL agent")
        raise
    finally:
        duration = time.perf_counter() - start
        logger.info("RL agent training finished in %.2f seconds", duration)
    return agent
