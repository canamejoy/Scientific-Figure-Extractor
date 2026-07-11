"""Pytest configuration for the test suite.

Adds the repository root (and the ``tests`` directory) to ``sys.path`` so
tests import ``src.*`` and their shared helpers without installation, and
disables the layout-detection model so the deterministic tests run without
downloading weights.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_TESTS = Path(__file__).resolve().parent
for path in (_ROOT, _TESTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Deterministic tests use the caption-anchor heuristics, not the layout model.
os.environ.setdefault("LAYOUT_DETECTION", "off")
