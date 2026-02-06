"""
Script to clean electoral data.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.cleaner import main as cleaner_main

if __name__ == '__main__':
    cleaner_main()
