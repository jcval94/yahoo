"""Remove all files under the daily model directory."""

import logging
from pathlib import Path

from .clean_models import delete_models
from .utils import load_config


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
CONFIG = load_config(CONFIG_PATH)

DAILY_DIR = Path(__file__).resolve().parents[1] / CONFIG.get(
    "model_dir", "models/daily"
)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    delete_models(DAILY_DIR)

