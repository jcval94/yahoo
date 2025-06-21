"""Utilities to send email and Telegram notifications."""
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


def send_telegram(message: str, **kwargs) -> None:
    start = time.perf_counter()
    logger.info("Sending Telegram notification")
    try:
        # Placeholder for telegram sending logic
        pass
    except Exception:
        logger.exception("Error while sending telegram message")
        raise
    finally:
        duration = time.perf_counter() - start
        logger.info("Telegram notification finished in %.2f seconds", duration)
