"""Run all project tests.

Usage:
  python tests/test.py
  python tests/test.py -q
  python tests/test.py test_prepare.py -k merge
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent


def main() -> int:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    args = sys.argv[1:]
    # If no explicit test path given, run the whole tests/ directory.
    has_path = any(not a.startswith("-") for a in args)
    if not has_path:
        args = [str(TESTS), *args]
    return pytest.main(["-v", *args])


if __name__ == "__main__":
    raise SystemExit(main())
