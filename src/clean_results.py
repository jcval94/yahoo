"""Delete result files older than a certain age."""
from __future__ import annotations

import logging
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def delete_old_results(directory: Path = RESULTS_DIR, older_than: int = 90) -> None:
    """Delete all files older than ``older_than`` days inside ``directory``."""
    if not directory.exists():
        logger.warning("Results directory %s does not exist", directory)
        return

    cutoff = datetime.now() - timedelta(days=older_than)
    deleted = False
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        try:
            if datetime.fromtimestamp(path.stat().st_mtime) < cutoff:
                path.unlink()
                logger.info("Deleted %s", path)
                deleted = True
        except Exception:  # pragma: no cover - log but continue
            logger.exception("Failed to delete %s", path)

    if not deleted:
        logger.info("No result files older than %d days in %s", older_than, directory)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    delete_old_results()
