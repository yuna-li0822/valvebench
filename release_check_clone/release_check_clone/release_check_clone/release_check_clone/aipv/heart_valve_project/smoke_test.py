"""Top-level smoke test entrypoint for ValveBench."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from valvebench.evaluation.smoke_test import smoke_test


if __name__ == '__main__':
    smoke_test()
