"""Package-local configuration; no dependencies on benchmark directories."""
from pathlib import Path
PACKAGE_DIR = Path(__file__).resolve().parent
MODE_FILE = PACKAGE_DIR / "mode"
CALIBRATION_FILE = PACKAGE_DIR / "calibration.json"
TAG = "QWEN_XPU_FP8_TEST"
MISSING = object()
