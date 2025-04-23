import logging
from logging.handlers import RotatingFileHandler
from typing import Union
from pathlib import Path
from datetime import datetime
import gurobipy as gp


def setup_logger(
    name: Union[str, None] = None, 
    log_file: Union[str, Path, None] = None, 
    level: int = logging.INFO, 
    console: bool = True,
    console_level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with specified configurations.

    Args:
        name (str): Name of the logger.
        log_file (Union[str, Path, None], optional): Path to the log file. If not specified, logging to file is disabled.
        level: Logging level, e.g., logging.INFO, logging.DEBUG.
        console (bool): Whether to log to console. Defaults to True.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid logging duplication
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(str(log_file), maxBytes=1048576, backupCount=5)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_level = console_level if console_level is not None else level
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    return logger

class MyLogger(object):
    def __init__(self, filename):
        self.file = open(filename, 'w')

    def log(self, message):
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] 
        message = f"[{time_str}]{message}"
        self.file.write(message + '\n')
        self.file.flush()
    
    def dispose(self):
        self.file.close()

    def info(self, message):
        self.log(f"[INFO] - {message}")

    def debug(self, message):
        self.log(f"[DEBUG] - {message}")

    def warning(self, message):
        self.log(f"[WARNING] - {message}")

    def error(self, message):
        self.log(f"[ERROR] - {message}")
        
    def critical(self, message):
        self.log(f"[CRITICAL] - {message}")
        
        
gurobi_env = gp.Env()
gurobi_env.setParam('LogToConsole', 0)