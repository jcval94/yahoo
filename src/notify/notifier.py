"""Notification utilities"""
import logging


def send_notification(message: str) -> None:
    """Placeholder notification via logging."""
    logger = logging.getLogger(__name__)
    logger.info("Notification: %s", message)
