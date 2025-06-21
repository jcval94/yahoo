import logging
import time
from contextlib import contextmanager

@contextmanager
def timed_stage(name: str):
    """Context manager to log start/end time of a stage."""
    logger = logging.getLogger(__name__)
    start = time.perf_counter()
    logger.info("Starting %s", name)
    try:
        yield
    except Exception:
        logger.exception("Exception in %s", name)
        raise
    finally:
        duration = time.perf_counter() - start
        logger.info("Finished %s in %.2f seconds", name, duration)
