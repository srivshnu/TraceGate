import sys
from pathlib import Path

# Ensure root directory is in sys.path so 'src' imports resolve cleanly on all platforms
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Launch main dashboard
from src.app.main import *
