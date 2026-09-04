import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.main import app  # noqa: F401
