"""
src/logger.py
Configures project-wide logging (console + rotating file).
"""

import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logger(log_dir: str = "logs", level: int = logging.INFO) -> None:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "pipeline.log")

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    root.setLevel(level)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(fmt, datefmt))

    # File handler (max 5 MB, keep 3 backups)
    fh = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3)
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(fmt, datefmt))

    root.addHandler(ch)
    root.addHandler(fh)
