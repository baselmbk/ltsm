import logging
import sys
from datetime import datetime


def setup_logger(name, log_file=None, level=logging.INFO):
    """Set up logger with file and console handlers."""
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logger = logging.getLogger(name)
    logger.setLevel(level)

    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def get_logger(name):
    """Get or create a logger with the given name."""
    if name in logging.Logger.manager.loggerDict:
        return logging.getLogger(name)
    else:
        log_file = f"logs/{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        return setup_logger(name, log_file)
