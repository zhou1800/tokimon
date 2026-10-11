"""Include the repository workspace in the Python distribution's data directory."""

from pathlib import Path
import sys

from setuptools import setup

SOURCE_ROOT = Path(__file__).resolve().parent
# PEP 517 metadata hooks need not put the source directory on sys.path.
sys.path.insert(0, str(SOURCE_ROOT))

from build_support import workspace_data_files


setup(data_files=workspace_data_files(SOURCE_ROOT))
