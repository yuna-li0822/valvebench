"""Shared pytest setup for heart_valve_project tests.

The project uses namespace-package imports (`heart_valve_project.X`), which
requires the `aipv/` directory (the parent of this package) to be on
sys.path. Centralizing that here means individual test files don't need
their own sys.path hacks.
"""
import sys
from pathlib import Path

_AIPV_DIR = Path(__file__).resolve().parents[2]
if str(_AIPV_DIR) not in sys.path:
    sys.path.insert(0, str(_AIPV_DIR))
