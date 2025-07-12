"""Utility to delete saved model files."""
import logging
from pathlib import Path
import shutil
import yaml

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
with open(CONFIG_PATH) as cfg_file:
    CONFIG = yaml.safe_load(cfg_file)

MODEL_DIR = Path(__file__).resolve().parents[1] / CONFIG.get("model_dir", "models/daily")


def delete_models(model_dir: Path = MODEL_DIR) -> None:
    """Delete all model files inside ``model_dir``.

    The directory may contain nested subfolders. If it does not exist or no
    files are found, a warning or informational message is logged accordingly.
    """

    if not model_dir.exists():
        logger.warning("Model directory %s does not exist", model_dir)
        return

    deleted = False
    for item in model_dir.rglob("*"):
        if not item.is_file():
            continue
        try:
            item.unlink()
            logger.info("Deleted %s", item)
            deleted = True
        except Exception:
            logger.exception("Failed to delete %s", item)

    if not deleted:
        logger.info("No model files found in %s", model_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    delete_models()

