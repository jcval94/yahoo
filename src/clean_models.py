"""Utility to delete saved model files."""
import logging
from pathlib import Path
import shutil
import yaml

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
with open(CONFIG_PATH) as cfg_file:
    CONFIG = yaml.safe_load(cfg_file)

MODEL_DIR = Path(__file__).resolve().parents[1] / CONFIG.get("model_dir", "models")


def delete_models(model_dir: Path = MODEL_DIR) -> None:
    """Delete all files and subdirectories in the given directory."""
    for item in model_dir.iterdir():
        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
            logger.info("Deleted %s", item)
        except Exception:
            logger.error("Failed to delete %s", item)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    delete_models()

