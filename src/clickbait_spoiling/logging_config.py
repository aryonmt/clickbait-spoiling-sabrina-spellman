import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Union


def setup_logging(
    log_dir: Union[str, Path], run_name: str, level: int = logging.INFO
) -> logging.Logger:
    """Configure a logger with a console handler (INFO+) and a rotating file handler
    (DEBUG+, max 5MB x 3 backups) writing to {log_dir}/{run_name}.log. Idempotent.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{run_name}.log"

    logger = logging.getLogger("clickbait_spoiling")

    # Check if handlers are already configured to avoid duplication
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addFilter(logging.Filter("clickbait_spoiling"))
    logger.addHandler(console_handler)

    # File Handler
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
