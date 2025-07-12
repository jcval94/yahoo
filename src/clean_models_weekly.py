"""Remove all files under the weekly model directory."""

import logging
from pathlib import Path

from .clean_models import delete_models


WEEKLY_DIR = Path(__file__).resolve().parents[1] / "models/weekly"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    delete_models(WEEKLY_DIR)

