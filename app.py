import runpy
import sys
from pathlib import Path

# Ensure the project root is importable so 'src.*' imports resolve on all platforms
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Streamlit re-executes this script on every interaction. A plain `import` only runs
# the dashboard once (modules are cached), leaving later reruns blank, so run the
# file fresh each time instead.
runpy.run_path(str(ROOT_DIR / "src" / "app" / "main.py"), run_name="__main__")
