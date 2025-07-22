"""
Test suite for the receipt processing application.
"""

import sys
import os
from pathlib import Path

# Add src directory to path for testing
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

__version__ = "1.0.0"
__author__ = "Receipt Processing Team"

