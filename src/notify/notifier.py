"""Utility functions to send email notifications."""
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def send_email(message: str, **kwargs) -> None:
    start = time.perf_counter()
    logger.info("Sending email notification")
    try:
        # Placeholder for email sending logic
        pass
    except Exception:
        logger.exception("Error while sending email")
        raise
    finally:
        duration = time.perf_counter() - start
        logger.info("Email notification finished in %.2f seconds", duration)


