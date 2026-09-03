import os
from pathlib import Path
import logging
from dotenv import load_dotenv

# Centralized Project Paths
BASE_DIR = Path(__file__).resolve().parent.parent

# Always resolve .env relative to project root — works regardless of cwd
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Ensure data directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# API Configurations
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def setup_logger(name: str) -> logging.Logger:
    """Configures and returns a standard logger for the project."""
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
