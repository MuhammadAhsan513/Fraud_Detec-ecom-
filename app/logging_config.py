"""Logging setup: one stdout handler, key=value messages, level from LOG_LEVEL."""
import logging
import sys

LOGGER_NAME = "app"
FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def setup_logging(level: str) -> logging.Logger:
    """Configure the `app` logger (idempotent: safe to call once per app instance)."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    if not any(getattr(h, "_app_handler", False) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(FORMAT))
        handler._app_handler = True
        logger.addHandler(handler)
    logger.propagate = True        # lets pytest's caplog see records too
    return logger


def get_logger(name: str = "") -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME)
